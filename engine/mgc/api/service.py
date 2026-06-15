"""Engine facade — UI-agnostic orchestration over all modules.

This is the single entry point the Typer CLI uses now and the Tauri sidecar
will reuse in Phase 2. It owns the Store + active Embedder and sequences the
modules; it contains no UI and no business logic that belongs in a module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from mgc.config import Config
from mgc.store import Store
from mgc.types import Suggestion


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.store = Store.open(config.db_path)
        self._embedder = None  # lazy

    # ---- lifecycle ----------------------------------------------------------
    @property
    def model(self) -> str:
        return self.config.active_model

    @property
    def embedder(self):
        if self._embedder is None:
            from mgc.embed import get_embedder
            self._embedder = get_embedder(self.config.active_model)
        return self._embedder

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- ingestion + embedding ---------------------------------------------
    def scan(self, root: Optional[str] = None) -> list[int]:
        from mgc.ingest.scanner import scan
        root = root or self.config.library_root
        if not root:
            raise ValueError("No library_root configured and no root passed.")
        return scan(self.store, root, self.config.extensions)

    def embed_all(self, force: bool = False, progress: Optional[Callable] = None) -> int:
        from mgc.embed.cache import embed_track
        n = 0
        tracks = self.store.iter_tracks()
        for i, t in enumerate(tracks):
            try:
                embed_track(self.store, self.embedder, t,
                            window_seconds=self.config.window_seconds,
                            hop_seconds=self.config.window_hop_seconds,
                            max_windows=self.config.max_windows, force=force)
                n += 1
            except Exception as e:  # keep the batch going; bad files are skipped
                self.store.set_track_status(t.id, "decode_error")
                if progress:
                    progress(f"skip {t.path}: {e}")
            if progress:
                progress(f"embedded {i + 1}/{len(tracks)}")
        return n

    # ---- taxonomy + genres --------------------------------------------------
    def seed_taxonomy(self, refs_dir: str, limit: Optional[int] = None) -> int:
        from mgc.taxonomy.rym import seed_taxonomy
        return seed_taxonomy(self.store, refs_dir, limit=limit)

    def add_genre_by_example(self, name: str, track_ids: list[int],
                             parent_id: Optional[int] = None, level: str = "subgenre") -> int:
        from mgc.registry.centroids import create_genre_by_example
        return create_genre_by_example(self.store, name, track_ids, self.model,
                                       parent_id=parent_id, level=level)

    # ---- classification -----------------------------------------------------
    def suggest(self, track_id: int) -> list[Suggestion]:
        from mgc.classify.classifier import suggest
        return suggest(self.store, track_id, self.model,
                       top_k=self.config.top_k, threshold=self.config.confidence_threshold)

    def suggest_all(self, persist: bool = True) -> dict[int, list[Suggestion]]:
        from mgc.classify.classifier import suggest_all
        out = suggest_all(self.store, self.model,
                          top_k=self.config.top_k, threshold=self.config.confidence_threshold)
        if persist:
            for tid, suggestions in out.items():
                if suggestions:
                    top = suggestions[0]
                    self.store.set_assignment(tid, top.genre_id, top.confidence,
                                              top.method, status="suggested")
        return out

    # ---- clustering + similarity -------------------------------------------
    def cluster(self, min_cluster_size: int = 2):
        from mgc.cluster.cluster import cluster_tracks
        return cluster_tracks(self.store, self.model, min_cluster_size=min_cluster_size)

    def similar(self, track_id: int, n: int = 10):
        from mgc.similarity.similar import similar_tracks
        return similar_tracks(self.store, track_id, self.model, n=n)

    # ---- review / confirm (active learning) --------------------------------
    def review(self, limit: int = 20) -> list:
        """Lowest-confidence suggested assignments first — what to edit."""
        rows = self.store.conn.execute(
            """SELECT a.track_id, a.genre_id, a.confidence, t.path
               FROM assignments a JOIN tracks t ON t.id=a.track_id
               WHERE a.status='suggested'
               ORDER BY a.confidence ASC LIMIT ?""", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def confirm(self, track_id: int, genre_id: int) -> None:
        from mgc.registry.centroids import add_exemplar
        self.store.set_assignment(track_id, genre_id, 1.0, "manual", status="confirmed")
        add_exemplar(self.store, genre_id, track_id, self.model)

    # ---- output actions -----------------------------------------------------
    def write_tags(self, dry_run: bool = False) -> list[dict]:
        from mgc.actions.tags import write_genre
        plans = []
        for r in self.store.iter_assignments():
            if r["genre_id"] is None or r["status"] == "rejected":
                continue
            sub = self.store.get_genre(r["genre_id"])
            if sub is None:
                continue
            parent = self.store.get_genre(sub.parent_id) if sub.parent_id else None
            track = self.store.get_track(r["track_id"])
            plans.append(write_genre(self.store, track, sub.name,
                                     parent=parent.name if parent else None,
                                     write_parent_to_grouping=self.config.write_parent_to_grouping,
                                     dry_run=dry_run))
        return plans

    def organize(self, dry_run: bool = False) -> list[dict]:
        from mgc.actions.organize import plan_organize, execute_organize
        root = self.config.organize_root or str(Path(self.config.library_root or ".") / "_organized")
        plan = plan_organize(self.store, root)
        return execute_organize(self.store, plan, mode=self.config.organize_mode, dry_run=dry_run)

    def undo(self) -> dict:
        from mgc.actions.tags import undo_tags
        from mgc.actions.organize import undo_organize
        return {"tags": undo_tags(self.store), "organize": undo_organize(self.store)}

    # ---- evaluation ---------------------------------------------------------
    def project(self, method: str = "pca"):
        from mgc.eval.validate import project_embeddings
        return project_embeddings(self.store, self.model, method=method)
