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
from mgc.types import LEVEL_GENRE, LEVEL_SUBGENRE, GenreNode, Suggestion


class Engine:
    def __init__(self, config: Config, store: Optional[Store] = None,
                 check_same_thread: bool = True):
        self.config = config
        self.store = store or Store.open(config.db_path, check_same_thread=check_same_thread)
        self._embedder = None  # lazy

    # ---- lifecycle ----------------------------------------------------------
    @property
    def model(self) -> str:
        return self.config.active_model

    @property
    def classify_model(self) -> str:
        """Grouping/classification runs on the fused space once it's been built,
        otherwise on the base embedding model."""
        try:
            from mgc.fusion import FUSED_MODEL
            ids, _ = self.store.load_matrix(FUSED_MODEL)
            if ids:
                return FUSED_MODEL
        except Exception:
            pass
        return self.model

    def fuse(self, progress=None) -> int:
        """Build fused vectors (base + CLAP + AudioSet tags + tempo/energy) and
        rebuild every genre centroid in that richer space."""
        from mgc.fusion import FUSED_MODEL, build_fused
        from mgc.registry.centroids import recompute_centroid
        n = build_fused(self.store, self.model, progress=progress)
        if n:
            for g in self.store.iter_genres():
                try:
                    recompute_centroid(self.store, g.id, FUSED_MODEL)
                except Exception:
                    pass
        return n

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
        return create_genre_by_example(self.store, name, track_ids, self.classify_model,
                                       parent_id=parent_id, level=level)

    # ---- classification -----------------------------------------------------
    def suggest(self, track_id: int) -> list[Suggestion]:
        from mgc.classify.classifier import suggest
        return suggest(self.store, track_id, self.classify_model,
                       top_k=self.config.top_k, threshold=self.config.confidence_threshold)

    def suggest_all(self, persist: bool = True) -> dict[int, list[Suggestion]]:
        from mgc.classify.classifier import suggest_all
        out = suggest_all(self.store, self.classify_model,
                          top_k=self.config.top_k, threshold=self.config.confidence_threshold)
        if persist:
            for tid, suggestions in out.items():
                if suggestions:
                    top = suggestions[0]
                    self.store.set_assignment(tid, top.genre_id, top.confidence,
                                              top.method, status="suggested")
                    # keep the full blend (alternatives + scores) for relabeling
                    self.store.save_suggestions(
                        tid, [(s.genre_id, s.confidence, s.method) for s in suggestions])
        return out

    def track_suggestions(self, track_id: int) -> list:
        """The genre blend for a track: [{genre_id, name, confidence, rank, ...}], best first."""
        return self.store.get_suggestions(track_id)

    # ---- clustering + similarity -------------------------------------------
    def cluster(self, min_cluster_size: int = 2, n_clusters: Optional[int] = None):
        # Cluster on the pure SOUND embedding (timbre/texture), not the fused vector:
        # genre grouping should be by how a track sounds, not its tempo/energy (BPM
        # is metadata). Falls back to the fused/base space only if the raw model has
        # no embeddings.
        from mgc.cluster.cluster import cluster_tracks
        model = self.model
        if not self.store.load_matrix(model)[0]:
            model = self.classify_model
        return cluster_tracks(self.store, model,
                              min_cluster_size=min_cluster_size, n_clusters=n_clusters)

    def similar(self, track_id: int, n: int = 10):
        from mgc.similarity.similar import similar_tracks
        return similar_tracks(self.store, track_id, self.classify_model, n=n)

    def explain_similarity(self, a_id: int, b_id: int) -> dict:
        """Why two tracks are alike: shared vs differing sounds/mood/tempo/key."""
        from mgc.similarity.explain import explain_similarity
        return explain_similarity(self.store, a_id, b_id, self.classify_model)

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

    def auto_organize(self, n_groups: int = 14, dry_run: bool = True) -> dict:
        """Two-level auto-organize: major genre folders, subgenre subfolders, sorted.

        Drop the whole library in: cluster by sound into ``n_groups`` fine sound groups,
        name each by its dominant AudioSet genre (Techno, House, Disco, ...), then MERGE
        the fine groups that share a genre into one major folder, with each fine group a
        numbered subgenre inside it (``root/<Major>/<Major N>/file``). Major names are
        reliable genres; subgenres are real sound clusters with placeholder names (rename
        freely). Dry-run by default so the tree can be reviewed before any file moves.
        """
        import numpy as np
        from collections import Counter
        from sklearn.cluster import KMeans
        from mgc.cluster.cluster import _reduce
        from mgc.tagging import get_audioset_labels, top_tags

        model = self.model if self.store.load_matrix(self.model)[0] else self.classify_model
        ids, mat = self.store.load_matrix(model)
        if not ids:
            return {"tree": [], "plan": [], "count": 0}
        mat = np.asarray(mat, dtype=np.float32)
        labels = get_audioset_labels() or []

        # AudioSet fallback vocabulary: specific genres only. Umbrella tags ("Electronic
        # music", "Dance music") sit on every track and would collapse the whole library
        # into one folder, so they are excluded.
        AUDIOSET_GENRES = {
            "house music", "techno", "dubstep", "drum and bass", "ambient music",
            "trance music", "disco", "funk", "trip hop", "pop music", "hip hop music",
            "rock music", "jazz", "reggae", "soul music", "classical music",
        }
        # Broad parent for the major folder; longest names first so "minimal techno"
        # -> Techno and "tech house" -> House.
        BROAD = ["drum and bass", "minimal techno", "tech house", "deep house",
                 "techno", "house", "trance", "disco", "dubstep", "ambient",
                 "trip hop", "breakbeat", "garage", "hardcore", "downtempo",
                 "funk", "soul", "jazz", "hip hop", "reggae", "pop", "rock"]

        def clean(name: str) -> str:
            return name.replace(" music", "").replace(" Music", "").strip().title()

        # Per-track genres: AcoustID/MusicBrainz identity (authoritative + specific) when
        # available, else AudioSet's specific-genre tags. This is the AcoustID payoff: a
        # track recognised by its sound carries a real genre regardless of its filename.
        # Broad umbrellas appear on nearly every artist; drop them so the specific
        # genre (techno/french house) wins the vote instead of "electronic".
        UMBRELLA = {"electronic", "electronica", "edm", "dance", "club", "pop",
                    "instrumental", "experimental", "electro", "leftfield"}
        track_genres: dict[int, list[str]] = {}
        for tid in ids:
            idn = self.store.get_identity(tid)
            if idn and idn.get("genres"):
                specific = [g.lower() for g in idn["genres"] if g.lower() not in UMBRELLA]
                track_genres[tid] = specific or [g.lower() for g in idn["genres"]]
                continue
            u = self.store.get_understanding(tid)
            tags = []
            if u and u.get("audioset") is not None and labels:
                for tg in top_tags(u["audioset"], labels, k=5):
                    if tg["label"].lower() in AUDIOSET_GENRES:
                        tags.append(tg["label"].lower())
            track_genres[tid] = tags

        def dominant_genre(rows):
            gf: Counter = Counter()
            for r in rows:
                gf.update(set(track_genres.get(ids[r], [])))
            return clean(gf.most_common(1)[0][0]) if gf else "Electronic"

        def major_of(genre: str) -> str:
            g = genre.lower()
            for kw in BROAD:
                if kw in g:
                    return clean(kw)
            return clean(genre)  # already broad / unknown -> itself

        def kmeans_rows(rows, k):
            if k <= 1 or len(rows) <= k:
                return [list(rows)]
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(_reduce(mat[rows]))
            out: dict[int, list[int]] = {}
            for r, l in zip(rows, lab):
                out.setdefault(int(l), []).append(r)
            return list(out.values())

        # Fine sound clusters; each becomes a subgenre (its dominant real genre) under
        # that genre's broad major folder. Clusters that resolve to the same names merge.
        fine = kmeans_rows(list(range(len(ids))), min(n_groups, len(ids)))
        majors: dict[str, dict] = {}   # major -> {subgenre name -> [rows]}
        for frows in fine:
            sub = dominant_genre(frows)
            maj = major_of(sub)
            majors.setdefault(maj, {})
            sname = sub if clean(sub) != maj else f"{maj} {len(majors[maj]) + 1}"
            majors[maj].setdefault(sname, []).extend(frows)

        tree = []
        for maj, subs in majors.items():
            parent_id = self.store.upsert_genre(
                GenreNode(name=maj, level=LEVEL_GENRE, source="custom"))
            subs_out, size = [], 0
            for sname, rows in sorted(subs.items(), key=lambda kv: -len(kv[1])):
                sub_id = self.store.upsert_genre(
                    GenreNode(name=sname, parent_id=parent_id,
                              level=LEVEL_SUBGENRE, source="custom"))
                for r in rows:
                    self.store.set_assignment(ids[r], sub_id, 0.5, "cluster", status="suggested")
                subs_out.append({"name": sname, "size": len(rows)})
                size += len(rows)
            tree.append({"major": maj, "size": size, "subgenres": subs_out})

        plan = self.organize(dry_run=dry_run)
        return {"tree": sorted(tree, key=lambda t: -t["size"]),
                "plan": plan, "count": len(plan)}

    def undo(self) -> dict:
        from mgc.actions.tags import undo_tags
        from mgc.actions.organize import undo_organize
        return {"tags": undo_tags(self.store), "organize": undo_organize(self.store)}

    # ---- evaluation ---------------------------------------------------------
    def project(self, method: str = "pca"):
        from mgc.eval.validate import project_embeddings
        return project_embeddings(self.store, self.model, method=method)

    # ---- analysis / set-builder / identify / radio -------------------------
    def analyze_all(self, progress=None) -> int:
        from mgc.analysis import analyze_all
        return analyze_all(self.store, progress=progress)

    def build_set(self, description: str, length: Optional[int] = None) -> dict:
        from mgc.setbuilder.builder import build_set
        return build_set(self.store, description, self.model, length=length)

    def identify(self, path: str, n: int = 5) -> list:
        from mgc.identify.identify import identify_in_library
        return identify_in_library(self.store, path, self.model, n=n)

    def identify_mix(self, path: str, window_seconds: float = 15.0,
                     hop_seconds: float = 7.0) -> list:
        from mgc.identify.identify import identify_mix
        return identify_mix(self.store, path, self.model,
                            window_seconds=window_seconds, hop_seconds=hop_seconds)

    def region(self, artist: str, title: Optional[str] = None) -> dict:
        from mgc.identify.identify import lookup_region
        return lookup_region(artist, title)

    # ---- AudioSet tagging + open-vocab search ------------------------------
    def tag_all(self, progress=None) -> int:
        from mgc.tagging import tag_all
        return tag_all(self.store, progress=progress)

    def search(self, query: str, n: int = 50, threshold=None) -> dict:
        from mgc.search import search
        return search(self.store, query, n=n, threshold=threshold)

    def understanding(self, track_id: int) -> Optional[dict]:
        from mgc.tagging import get_audioset_labels, top_tags
        from mgc.understanding import compile_record
        u = self.store.get_understanding(track_id)
        if not u or u.get("audioset") is None:
            return None
        labels = get_audioset_labels() or []
        vec = u["audioset"]
        tags = top_tags(vec, labels) if labels else []
        analysis = self.store.get_analysis(track_id) or {}
        rec = compile_record(vec, labels, analysis=analysis) if labels else {}
        vocal = dict(rec.get("vocal") or {})
        stored_vocal = u.get("vocal") if isinstance(u.get("vocal"), dict) else {}
        if stored_vocal and stored_vocal.get("language"):  # from the deep pass (Whisper)
            vocal["language"] = stored_vocal["language"]
            vocal["language_conf"] = stored_vocal.get("language_conf")
        return {"track_id": track_id, "top_tags": tags,
                "instruments": rec.get("instruments"), "vocal": vocal,
                "mood": rec.get("mood"), "caption": rec.get("caption"),
                "tags_canonical": rec.get("tags_canonical"), "deep_done": u.get("deep_done")}

    def deep_analyze(self, track_id: int) -> dict:
        from mgc.deep import deep_analyze
        return deep_analyze(self.store, track_id)

    def deep_analyze_all(self, progress=None) -> int:
        from mgc.deep import deep_analyze_all
        return deep_analyze_all(self.store, progress=progress)

    # ---- MusicBrainz metadata + genre seeding ------------------------------
    def mb_lookup(self, artist: str, title: Optional[str] = None) -> dict:
        from mgc.metadata import mb_lookup
        return mb_lookup(artist, title)

    def identify_all(self, key: Optional[str] = None, force: bool = False,
                     progress: Optional[Callable] = None, limit: Optional[int] = None,
                     use_discogs: bool = True) -> dict:
        """Recognise each track and store an authoritative genre/artist/title/region.

        Two sources, best of both: (1) AcoustID audio fingerprint -> MusicBrainz (exact,
        but only covers released music); (2) Discogs search on the parsed artist/title
        (its 'styles' are the DJ-grade subgenres, and it covers underground/label
        releases far better). Resumable, best-effort: a miss leaves the track for the
        AudioSet fallback in auto_organize."""
        import os

        from mgc.metadata import acoustid as aid, discogs, mb_lookup_by_mbid
        from mgc.metadata.parse import parse_artist_title

        if self.config.fpcalc_path:
            os.environ["MGC_FPCALC"] = self.config.fpcalc_path
        acoustid_key = aid.api_key(key or self.config.acoustid_key)
        dk, ds = discogs.creds()
        have_discogs = use_discogs and bool(dk and ds)
        if not acoustid_key and not have_discogs:
            return {"error": "no_identify_source", "identified": 0, "total": 0}

        tracks = self.store.iter_tracks()
        if limit:
            tracks = tracks[:limit]
        identified = 0
        for i, t in enumerate(tracks):
            if not force and self.store.has_identity(t.id):
                identified += 1
                if progress:
                    progress(i + 1, len(tracks))
                continue

            tags = t.existing_tags or {}
            mbid = artist = title = area = year = None
            genres: list = []
            score = None
            if acoustid_key:  # 1) fingerprint -> MusicBrainz
                res = aid.identify(t.path, key=acoustid_key)
                mbid = res.get("recording_mbid")
                if mbid:
                    meta = mb_lookup_by_mbid(mbid)
                    genres = meta.get("genres") or []
                    artist = meta.get("artist") or res.get("artist")
                    title = meta.get("title") or res.get("title")
                    area, year, score = meta.get("area"), meta.get("year"), res.get("score")
            if not genres and have_discogs:  # 2) Discogs on parsed artist/title
                pa, pt = parse_artist_title(tags.get("title"), tags.get("artist"), t.path)
                d = discogs.lookup(pa, pt)
                dg = (d.get("styles") or []) + (d.get("genres") or [])
                if dg:
                    genres = dg
                    artist, title = artist or pa, title or pt
                    year = year or d.get("year")

            if genres or mbid:
                self.store.save_identity(t.id, recording_mbid=mbid, artist=artist,
                                         title=title, genres=genres, area=area,
                                         year=year, score=score)
                identified += 1
            if progress:
                progress(i + 1, len(tracks))
        return {"identified": identified, "total": len(tracks)}

    def seed_from_musicbrainz(self, min_examples: int = 3, progress=None) -> dict:
        from mgc.metadata import seed_genres_from_mb
        return seed_genres_from_mb(self.store, self.classify_model,
                                   min_examples=min_examples, progress=progress)

    def related_genres(self, genre: str, n: int = 25) -> list:
        from mgc.metadata import get_graph
        return get_graph().related(genre, limit=n)

    # ---- segment-level similarity (waveform region -> matching parts) ------
    def index_segments(self, progress=None) -> int:
        from mgc.segments import build_segment_index
        return build_segment_index(self.store, self.model, progress=progress)

    def search_by_segment(self, track_id: int, start: float, end: float, n: int = 20) -> list:
        from mgc.segments import embed_segment, find_similar_segments
        track = self.store.get_track(track_id)
        if not track:
            return []
        q = embed_segment(track.path, start, end, self.model)
        return find_similar_segments(self.store, q, self.model, n=n, exclude_track_id=track_id)

    def save_segment(self, track_id: int, start: float, end: float, label: Optional[str] = None,
                     note: Optional[str] = None, genre_id: Optional[int] = None) -> dict:
        from mgc.segments import embed_segment
        track = self.store.get_track(track_id)
        if not track:
            return {"ok": False, "error": "no such track"}
        q = embed_segment(track.path, start, end, self.model)
        sid = self.store.save_segment_exemplar(track_id, self.model, start, end, q,
                                               label=label, note=note, genre_id=genre_id)
        return {"ok": True, "segment_id": sid}

    def list_segments(self, genre_id: Optional[int] = None) -> list:
        rows = self.store.get_segment_exemplars(genre_id)
        for r in rows:
            r.pop("vector", None)  # don't ship the raw vector
        return rows

    def create_genre_from_segment(self, track_id: int, start: float, end: float, name: str,
                                  parent_id: Optional[int] = None, n: int = 8,
                                  level: str = "subgenre") -> dict:
        """Define a subgenre by a SOUND: embed the region, find the tracks that
        contain that part, and seed a by-example genre from them (+ the source)."""
        from mgc.segments import embed_segment, find_similar_segments
        track = self.store.get_track(track_id)
        if not track:
            return {"ok": False, "error": "no such track"}
        q = embed_segment(track.path, start, end, self.model)
        matches = find_similar_segments(self.store, q, self.model, n=n)
        examples, seen = [], set()
        for tid in [track_id, *[m["track_id"] for m in matches]]:
            if tid not in seen:
                seen.add(tid)
                examples.append(tid)
        gid = self.add_genre_by_example(name, examples, parent_id=parent_id, level=level)
        self.store.save_segment_exemplar(track_id, self.model, start, end, q,
                                         label=name, genre_id=gid)
        return {"ok": True, "genre_id": gid, "examples": examples, "matches": matches}

    def radio(self, track_id: int, n: int = 20) -> list:
        from mgc.similarity.similar import radio_queue
        ids = radio_queue(self.store, track_id, self.classify_model, n=n)
        out = []
        for tid in ids:
            t = self.store.get_track(tid)
            out.append({"track_id": tid, "name": Path(t.path).name if t else str(tid)})
        return out
