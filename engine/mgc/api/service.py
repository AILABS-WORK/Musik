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

    def auto_organize(self, n_groups: int = 14, dry_run: bool = True,
                      min_similarity: float = 0.5) -> dict:
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

        # Self-clean so every (re)sort is idempotent: drop the PREVIOUS auto-generated
        # genres + all assignments, but KEEP your by-example genres (those with
        # exemplars) so re-sorting builds on your labels instead of duplicating folders.
        keep = [g.id for g in self.store.iter_genres() if self.store.get_exemplars(g.id)]
        self.store.conn.execute("DELETE FROM assignments")
        self.store.conn.execute("DELETE FROM cluster_members")
        self.store.conn.execute("DELETE FROM clusters")
        if keep:
            ph = ",".join("?" * len(keep))
            self.store.conn.execute(f"DELETE FROM genres WHERE id NOT IN ({ph})", tuple(keep))
        else:
            self.store.conn.execute("DELETE FROM genres")
        self.store.conn.commit()

        # AudioSet fallback vocabulary: specific genres only. Umbrella tags ("Electronic
        # music", "Dance music") sit on every track and would collapse the whole library
        # into one folder, so they are excluded.
        AUDIOSET_GENRES = {
            "house music", "techno", "dubstep", "drum and bass", "ambient music",
            "trance music", "disco", "funk", "trip hop", "pop music", "hip hop music",
            "rock music", "jazz", "reggae", "soul music", "classical music",
        }
        # TOP-LEVEL genres only -> the major folder. A specific style maps to its
        # family ("minimal techno"/"deep techno" -> Techno, "tech house"/"deep house"
        # -> House) and is then used as the *subgenre* name inside it.
        # "garage" must be checked AFTER "house"/"rock" so "garage house" -> House and
        # "garage rock" -> Rock (a namesake, dropped by the electronic filter), while
        # "uk garage"/"speed garage" still map to Garage.
        BROAD = ["drum and bass", "techno", "house", "trance", "disco", "dubstep",
                 "ambient", "trip hop", "breakbeat", "hardcore", "downtempo",
                 "funk", "soul", "jazz", "hip hop", "reggae", "rock", "pop", "garage"]

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

        def major_of(genre: str) -> str:
            g = genre.lower()
            for kw in BROAD:
                if kw in g:
                    return clean(kw)
            return clean(genre)  # already broad / unknown -> itself

        def vote_names(rows):
            """A cluster's (major, subgenre): major = most-common top-level family;
            subgenre = most-common *specific* style within that family (else None)."""
            broad: Counter = Counter()
            specific: list = []
            for r in rows:
                for g in track_genres.get(ids[r], []):
                    m = major_of(g)
                    broad[m] += 1
                    specific.append((m, clean(g)))
            if not broad:
                return "Electronic", None
            major = broad.most_common(1)[0][0]
            spec = Counter(s for (m, s) in specific if m == major and s != major)
            return major, (spec.most_common(1)[0][0] if spec else None)

        def kmeans_rows(rows, k):
            if k <= 1 or len(rows) <= k:
                return [list(rows)]
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(_reduce(mat[rows]))
            out: dict[int, list[int]] = {}
            for r, l in zip(rows, lab):
                out.setdefault(int(l), []).append(r)
            return list(out.values())

        pos = {t: i for i, t in enumerate(ids)}

        # Electronic families. In a predominantly-electronic library we ignore obvious
        # namesake hits (a techno track that Discogs matched to a "Country" artist),
        # which would otherwise create nonsense folders and mis-file tracks.
        ELECTRONIC = {"Techno", "House", "Trance", "Disco", "Dubstep", "Drum And Bass",
                      "Ambient", "Trip Hop", "Breakbeat", "Garage", "Downtempo", "Hardcore"}
        fam = Counter(major_of(g) for tid in ids for g in track_genres.get(tid, []))
        fam_total = sum(fam.values())
        elec_lib = bool(fam_total) and sum(
            c for m, c in fam.items() if m in ELECTRONIC) >= 0.6 * fam_total

        # Plausible BPM range per genre (octave-tolerant). A track whose tempo can't fit
        # a genre is almost certainly a wrong/namesake match -- e.g. Discogs calling a
        # 139 BPM hard-techno track "Italo-Disco" -- so we drop that label, which also
        # stops a bad anchor pulling a whole cluster into the wrong folder.
        GENRE_BPM = {
            "italo": (112, 132), "nu disco": (110, 126), "disco": (108, 130),
            "deep house": (115, 126), "tech house": (120, 130), "acid house": (118, 130),
            "house": (118, 130), "minimal": (125, 134), "dub techno": (120, 134),
            "hard techno": (140, 165), "peak time": (138, 150), "techno": (125, 140),
            "hardcore": (150, 200), "drum and bass": (160, 180), "trance": (130, 145),
            "dubstep": (135, 150), "speed garage": (130, 140), "uk garage": (128, 138),
            "garage": (126, 138), "breakbeat": (125, 140), "trip hop": (80, 110),
            "downtempo": (60, 112), "ambient": (0, 110),
        }
        ana = self.store.load_analysis()

        def _bpm(t):
            v = (ana.get(t) or {}).get("bpm")
            return float(v) if v else None

        def _bpm_ok(genre: str, bpm) -> bool:
            if bpm is None:
                return True
            g = genre.lower()
            for kw, (lo, hi) in GENRE_BPM.items():
                if kw in g:
                    return any(lo - 4 <= b <= hi + 4 for b in (bpm, bpm / 2, bpm * 2))
            return True  # unknown genre -> allow

        # YOUR labels are ground truth: any track that is an exemplar of a user
        # subgenre (added via "use as examples") becomes that (major, subgenre), which
        # overrides Discogs/AudioSet and seeds the similarity fill. This is the
        # by-example loop: label a few, re-sort, and your labels propagate by sound.
        user_label: dict[int, tuple] = {}
        for g in self.store.iter_genres(level=LEVEL_SUBGENRE):
            ex = self.store.get_exemplars(g.id)
            if not ex:
                continue
            parent = self.store.get_genre(g.parent_id) if g.parent_id else None
            maj = parent.name if parent else major_of(g.name)
            for tid in ex:
                user_label[tid] = (maj, g.name)

        def own_label(tid):
            """(major, specific subgenre | None): your label first, then identity."""
            if tid in user_label:
                return user_label[tid]
            tb = _bpm(tid)
            broad = None
            for g in track_genres.get(tid, []):
                m, s = major_of(g), clean(g)
                if elec_lib and m not in ELECTRONIC:
                    continue  # drop a namesake non-electronic match
                if not _bpm_ok(g, tb):
                    continue  # genre implausible for this tempo -> almost certainly wrong
                if broad is None:
                    broad = m
                if s != m:
                    return m, s          # first specific style wins
            return broad, None

        # 1) Place each track by its OWN identity (not the cluster's vote). Tracks with a
        #    specific style become anchors; broad-only/unidentified get filled in below.
        place: dict[int, tuple] = {}
        conf: dict[int, float] = {}   # real per-track confidence (not a flat 0.5)
        anchors: list[int] = []
        for tid in ids:
            maj, sub = own_label(tid)
            if maj:
                place[tid] = (maj, sub)
                # your label = certain; an identified specific style = high; broad-only = medium
                conf[tid] = 1.0 if tid in user_label else (0.85 if sub else 0.6)
                if sub:
                    anchors.append(tid)

        if anchors:
            # 2) Similarity fill in the PCA-REDUCED space. Raw 95M/MuQ embeddings are
            #    near-uniform (~0.95 cosine to everything), so nearest-neighbour there is
            #    meaningless; PCA removes the common baseline and exposes real structure.
            #    A track only inherits a subgenre if it is genuinely close (>= min_similarity)
            #    to an anchor; otherwise it goes to "<Major> - Unsorted" rather than being
            #    forced into a wrong subgenre.
            Z = _reduce(mat)  # PCA-reduced + L2-normalised
            amat = np.stack([Z[pos[t]] for t in anchors])
            # BPM gate: a fill only inherits a subgenre whose anchor tempo is close
            # (<= 14 BPM) so a 143-BPM hard-techno track can't land in a 125-BPM
            # Italo-Disco folder just because MuQ thought them "similar".
            abpm = [_bpm(t) for t in anchors]
            for tid in ids:
                if tid in place and place[tid][1]:
                    continue  # already has its own specific style
                tb = _bpm(tid)
                sims = amat @ Z[pos[tid]]
                order = np.argsort(-sims)
                cur_major = place.get(tid, (None, None))[0]
                # confident AND tempo-compatible anchors, best similarity first
                cand = [j for j in order if sims[j] >= min_similarity
                        and (tb is None or abpm[j] is None or abs(tb - abpm[j]) <= 14.0)]
                chosen = None
                if cand:
                    same = [j for j in cand if cur_major and place[anchors[j]][0] == cur_major]
                    j = (same or cand)[0]
                    chosen = place[anchors[j]]
                    conf[tid] = round(float(sims[j]), 3)
                if chosen:
                    place[tid] = chosen
                else:  # nothing confident + tempo-compatible -> major only, no fake subgenre
                    place[tid] = (cur_major or place[anchors[order[0]]][0], None)
                    conf[tid] = round(float(sims[order[0]]), 3)
            grouping: dict[str, dict] = {}
            for tid, (maj, sub) in place.items():
                sname = sub or f"{maj} - Unsorted"
                grouping.setdefault(maj, {}).setdefault(sname, []).append(pos[tid])
        else:
            # 3) No identities anywhere -> name fine sound clusters by AudioSet vote.
            grouping = {}
            for frows in kmeans_rows(list(range(len(ids))), min(n_groups, len(ids))):
                maj, sub = vote_names(frows)
                grouping.setdefault(maj, {})
                sname = sub or f"{maj} {len(grouping[maj]) + 1}"
                grouping[maj].setdefault(sname, []).extend(frows)

        tree = []
        for maj, subs in grouping.items():
            parent_id = self.store.upsert_genre(
                GenreNode(name=maj, level=LEVEL_GENRE, source="custom"))
            subs_out, size = [], 0
            for sname, rows in sorted(subs.items(), key=lambda kv: -len(kv[1])):
                sub_id = self.store.upsert_genre(
                    GenreNode(name=sname, parent_id=parent_id,
                              level=LEVEL_SUBGENRE, source="custom"))
                for r in rows:
                    self.store.set_assignment(ids[r], sub_id, conf.get(ids[r], 0.5),
                                              "identity", status="suggested")
                # Give every auto genre a centroid (mean of its tracks) so Suggest /
                # Similar / Radio work on them — without it, Suggest finds no centroids
                # and appears to "wipe" the library.
                if rows:
                    cen = mat[rows].mean(axis=0)
                    nrm = float(np.linalg.norm(cen))
                    if nrm > 0:
                        self.store.set_centroid(sub_id, (cen / nrm).astype(np.float32),
                                                is_text=False)
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

    def learn_metric(self, min_per_class: int = 5, min_classes: int = 2) -> dict:
        """Learn a genre-discriminative projection from your CONFIRMED/exemplar labels
        (shrinkage LDA on the sound embedding) and store it as the 'learned' space, which
        the map + grouping then use. Unlike auto-naming, this is supervised by YOUR ears,
        so the more groups you label the sharper everything gets. No-op until there are
        enough labels (>= min_classes genres with >= min_per_class tracks each)."""
        import collections

        import numpy as np

        labels: dict[int, int] = {}
        for r in self.store.conn.execute("SELECT track_id, genre_id FROM exemplars").fetchall():
            labels[r["track_id"]] = r["genre_id"]
        for r in self.store.conn.execute(
                "SELECT track_id, genre_id FROM assignments "
                "WHERE status='confirmed' AND genre_id IS NOT NULL").fetchall():
            labels[r["track_id"]] = r["genre_id"]
        if not labels:
            return {"error": "no labels yet", "learned": 0}

        def major(gid):
            g = self.store.get_genre(gid)
            return None if g is None else (g.parent_id if g.parent_id is not None else g.id)

        X, y = [], []
        for tid, gid in labels.items():
            mj = major(gid)
            emb = self.store.get_embedding(tid, self.model)
            if mj is not None and emb is not None:
                X.append(np.asarray(emb, dtype=np.float64))
                y.append(mj)
        if not X:
            return {"error": "no embeddings for labelled tracks", "learned": 0}
        X = np.array(X)
        y = np.array(y)
        keep = {c for c, n in collections.Counter(y).items() if n >= min_per_class}
        if len(keep) < min_classes:
            return {"error": "need more labels", "learned": 0,
                    "ready_classes": len(keep), "labelled_tracks": len(y),
                    "need": f">= {min_classes} genres with >= {min_per_class} labelled tracks each"}
        mask = np.array([c in keep for c in y])
        X, y = X[mask], y[mask]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        n_comp = max(1, min(len(keep) - 1, 48, X.shape[1]))
        lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="eigen", shrinkage="auto")
        lda.fit(X, y)

        ids, mat = self.store.load_matrix(self.model)
        M = mat.astype(np.float64)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        Z = lda.transform(M)
        for tid, z in zip(ids, Z):
            self.store.save_embedding(tid, "learned", z.astype(np.float32))
        return {"learned": len(ids), "classes": len(keep), "dims": int(Z.shape[1])}

    def propagate_from_labels(self, min_per_class: int = 4) -> dict:
        """Sort the WHOLE library from YOUR labels: learn a subgenre space (shrinkage
        LDA) from your exemplars/confirmed tracks, then assign every un-labelled track to
        its nearest labelled subgenre in that space. Your labels are kept as-is. Genres
        with the same name are merged into one class (handles accidental duplicates)."""
        import collections

        import numpy as np

        rows = self.store.conn.execute(
            "SELECT e.track_id tid, g.id gid, g.name name FROM exemplars e "
            "JOIN genres g ON g.id=e.genre_id").fetchall()
        rows2 = self.store.conn.execute(
            "SELECT a.track_id tid, g.id gid, g.name name FROM assignments a "
            "JOIN genres g ON g.id=a.genre_id WHERE a.status='confirmed' AND a.genre_id IS NOT NULL").fetchall()
        lab: dict[int, str] = {}
        canon: dict[str, collections.Counter] = {}
        for r in list(rows) + list(rows2):
            nm = (r["name"] or "").strip().lower()
            if not nm:
                continue
            lab[r["tid"]] = nm
            canon.setdefault(nm, collections.Counter())[r["gid"]] += 1
        if not lab:
            return {"error": "no labels yet", "assigned": 0}
        cnt = collections.Counter(lab.values())
        keep = {nm for nm, n in cnt.items() if n >= min_per_class}
        if len(keep) < 2:
            return {"error": "need more labels per genre", "assigned": 0,
                    "ready_classes": len(keep),
                    "need": f">= 2 genres with >= {min_per_class} labelled tracks"}
        canon_id = {nm: canon[nm].most_common(1)[0][0] for nm in keep}

        ids, mat = self.store.load_matrix(self.model)
        idx = {t: i for i, t in enumerate(ids)}
        M = mat.astype(np.float64)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

        Xtr, ytr = [], []
        for tid, nm in lab.items():
            if nm in keep and tid in idx:
                Xtr.append(M[idx[tid]])
                ytr.append(nm)
        Xtr = np.array(Xtr)
        ytr = np.array(ytr)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        n_comp = max(1, min(len(keep) - 1, 48, Xtr.shape[1]))
        lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="eigen", shrinkage="auto")
        lda.fit(Xtr, ytr)
        Z = lda.transform(M)
        for tid in ids:
            self.store.save_embedding(tid, "learned", Z[idx[tid]].astype(np.float32))

        Ztr = lda.transform(Xtr)
        cnames = sorted(keep)
        C = np.array([Ztr[ytr == nm].mean(axis=0) for nm in cnames])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

        labeled = set(lab)
        assigned = 0
        for tid in ids:
            if tid in labeled:
                continue
            z = Z[idx[tid]]
            z = z / (np.linalg.norm(z) + 1e-9)
            sims = C @ z
            j = int(np.argmax(sims))
            conf = round(float((sims[j] + 1.0) / 2.0), 3)
            self.store.set_assignment(tid, canon_id[cnames[j]], conf, "propagate", status="suggested")
            assigned += 1
        for nm in keep:
            try:
                from mgc.registry.centroids import recompute_centroid
                recompute_centroid(self.store, canon_id[nm], self.classify_model)
            except Exception:
                pass
        return {"assigned": assigned, "classes": len(keep),
                "labelled": len(labeled), "learned": len(ids)}

    def merge_duplicate_genres(self) -> dict:
        """Collapse duplicate genres (same name under the same parent) into one, cascading:
        when two majors of the same name merge, their children reparent and same-name
        children then merge too. The kept genre prefers level=genre, then the one with the
        most labels. Assignments + exemplars are moved over; empties deleted."""
        conn = self.store.conn

        def norm(s):
            return (s or "").strip().lower()

        def weight(gid):
            ex = conn.execute("SELECT count(*) FROM exemplars WHERE genre_id=?", (gid,)).fetchone()[0]
            asg = conn.execute("SELECT count(*) FROM assignments WHERE genre_id=?", (gid,)).fetchone()[0]
            ch = conn.execute("SELECT count(*) FROM genres WHERE parent_id=?", (gid,)).fetchone()[0]
            return (ex, asg + ch)

        def merge_into(keep, drop):
            conn.execute("UPDATE assignments SET genre_id=? WHERE genre_id=?", (keep, drop))
            for r in conn.execute("SELECT track_id FROM exemplars WHERE genre_id=?", (drop,)).fetchall():
                conn.execute("INSERT OR IGNORE INTO exemplars(genre_id, track_id) VALUES(?,?)",
                             (keep, r["track_id"]))
            conn.execute("DELETE FROM exemplars WHERE genre_id=?", (drop,))
            # Reparent children to keep; if a same-name child already exists there, merge
            # into it (recurse) to respect the UNIQUE(name, parent_id) constraint.
            for ch in conn.execute("SELECT id, name FROM genres WHERE parent_id=?", (drop,)).fetchall():
                existing = conn.execute(
                    "SELECT id FROM genres WHERE parent_id=? AND lower(name)=lower(?) AND id!=?",
                    (keep, ch["name"], ch["id"])).fetchone()
                if existing:
                    merge_into(existing["id"], ch["id"])
                else:
                    conn.execute("UPDATE genres SET parent_id=? WHERE id=?", (keep, ch["id"]))
            conn.execute("DELETE FROM genres WHERE id=?", (drop,))

        merged = 0
        for _ in range(500):  # iterate until no duplicate (parent, name) remains
            rows = conn.execute("SELECT id, name, level, parent_id FROM genres").fetchall()
            groups: dict = {}
            for r in rows:
                groups.setdefault((r["parent_id"], norm(r["name"])), []).append(r)
            dup = next((g for g in groups.values() if len(g) > 1), None)
            if dup is None:
                break
            dup.sort(key=lambda r: (r["level"] == "genre", weight(r["id"])), reverse=True)
            keep = dup[0]["id"]
            for r in dup[1:]:
                merge_into(keep, r["id"])
                merged += 1
        conn.commit()
        return {"merged": merged}

    # ---- evaluation ---------------------------------------------------------
    def project(self, method: str = "pca"):
        from mgc.eval.validate import project_embeddings
        # Prefer the supervised 'learned' space when it exists — it's arranged by YOUR
        # labels, so same-genre tracks land together (project it directly, no fusion).
        if self.store.load_matrix("learned")[0]:
            return project_embeddings(self.store, "learned", method=method, fuse=False)
        return project_embeddings(self.store, self.model, method=method)

    # ---- analysis / set-builder / identify / radio -------------------------
    def analyze_all(self, progress=None) -> int:
        from mgc.analysis import analyze_all
        return analyze_all(self.store, progress=progress)

    def index_spectral(self, force: bool = False, progress=None) -> int:
        """Compute + store each track's frequency fingerprint (resumable, best-effort)."""
        from mgc.analysis.waveform import spectral_profile
        tracks = self.store.iter_tracks()
        n = 0
        for i, t in enumerate(tracks):
            if force or not self.store.has_spectral(t.id):
                try:
                    prof = spectral_profile(t.path)
                    if prof:
                        self.store.save_spectral(t.id, prof)
                        n += 1
                except Exception:
                    pass
            if progress:
                progress(i + 1, len(tracks))
        return n

    def similar_detailed(self, track_id: int, n: int = 25) -> list:
        """Rich similarity for a track: an OVERALL score plus a per-frequency-range
        breakdown (sub..highs) for each match. Overall is cosine in a discriminative
        space — the supervised 'learned' space if you've labelled, otherwise a
        PCA-reduced embedding (raw MuQ cosines sit ~0.95 to everything, so we reduce
        them to spread the scores and actually tell tracks apart)."""
        import os as _os

        import numpy as np
        from mgc.similarity.bands import band_breakdown

        model = "learned" if self.store.load_matrix("learned")[0] else self.model
        ids, mat = self.store.load_matrix(model)
        if track_id not in ids or mat.size == 0:
            return []
        M = mat.astype(np.float64)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        if model != "learned" and M.shape[1] > 60:
            try:
                from mgc.eval.validate import _pca
                M = _pca(M, min(40, M.shape[0], M.shape[1]))
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            except Exception:
                pass
        qi = ids.index(track_id)
        cos = M @ M[qi]

        sids, smat = self.store.load_spectral()
        sidx = {tid: i for i, tid in enumerate(sids)}
        qprof = self.store.get_spectral(track_id)

        out = []
        for j in np.argsort(-cos):
            tid = ids[j]
            if tid == track_id:
                continue
            t = self.store.get_track(tid)
            bands = []
            if qprof is not None and tid in sidx:
                bands = band_breakdown(qprof, smat[sidx[tid]].tolist()).get("bands", [])
            out.append({"track_id": tid,
                        "name": _os.path.basename(t.path) if t else str(tid),
                        "score": round(float(cos[j]), 3), "bands": bands})
            if len(out) >= n:
                break
        return out

    def index_groove(self, force: bool = False, progress=None) -> int:
        """Compute + store each track's per-band temporal (groove) features."""
        from mgc.analysis.groove import groove_features
        tracks = self.store.iter_tracks()
        n = 0
        for i, t in enumerate(tracks):
            if force or not self.store.has_groove(t.id):
                try:
                    feat = groove_features(t.path)
                    if feat:
                        self.store.save_groove(t.id, feat)
                        n += 1
                except Exception:
                    pass
            if progress:
                progress(i + 1, len(tracks))
        return n

    def spectral_similar(self, track_id: int, n: int = 20) -> list:
        """Tracks with the most similar frequency fingerprint (bassline/brightness)."""
        import os as _os

        import numpy as np
        ids, mat = self.store.load_spectral()
        if track_id not in ids or mat.size == 0:
            return []
        # Mean-centre first: every track in an electronic library shares the same
        # bass-heavy shape (cosine ~0.99 to everything), so similarity must key on the
        # DEVIATION from the library-average profile (extra sub-bass, brighter highs, ...).
        mat = mat - mat.mean(axis=0, keepdims=True)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ mat[ids.index(track_id)]
        out = []
        for j in np.argsort(-sims):
            if ids[j] == track_id:
                continue
            tk = self.store.get_track(ids[j])
            out.append({"track_id": ids[j],
                        "name": _os.path.basename(tk.path) if tk else str(ids[j]),
                        "score": round(float(sims[j]), 3)})
            if len(out) >= n:
                break
        return out

    def _set_candidates(self) -> list:
        """Compact per-track table for the LLM: id, name, bpm, Camelot key, energy, genre."""
        from mgc.setbuilder.builder import _to_camelot, _camelot_label
        analysis = self.store.load_analysis()
        # current assigned genre name per track
        genre_of = {}
        for row in self.store.iter_assignments():
            if row["genre_id"]:
                g = self.store.get_genre(row["genre_id"])
                if g:
                    genre_of[row["track_id"]] = g.name
        out = []
        for t in self.store.iter_tracks():
            a = analysis.get(t.id) or {}
            out.append({
                "id": t.id,
                "name": (t.existing_tags or {}).get("title") or t.path.split("\\")[-1],
                "bpm": a.get("bpm"),
                "key": _camelot_label(_to_camelot(a.get("music_key"))) or None,
                "energy": a.get("energy"),
                "genre": genre_of.get(t.id),
            })
        return out

    def _set_pool(self, cands: list, parsed: dict, want: int, cap: int = 80) -> list:
        """A focused, diverse candidate pool small enough for the LLM to reason over.

        Bias to the requested genres when that leaves enough tracks, then cap by
        sampling evenly across the energy range so the full arc is still reachable.
        """
        genres = [g.lower() for g in (parsed.get("genres") or [])]
        if genres:
            filt = [c for c in cands if c.get("genre")
                    and any(g in c["genre"].lower() for g in genres)]
            if len(filt) >= max(2 * want, 16):
                cands = filt
        if len(cands) <= cap:
            return cands
        cands = sorted(cands, key=lambda c: c.get("energy") or 0.5)
        step = (len(cands) - 1) / (cap - 1)
        idx = sorted({round(i * step) for i in range(cap)})
        return [cands[i] for i in idx]

    def build_set(self, description: str, length: Optional[int] = None) -> dict:
        """LLM-reasoned set ordering (local Ollama) with the heuristic as fallback."""
        import re

        from mgc.setbuilder.builder import build_set, parse_description

        parsed = parse_description(description)
        # duration in minutes -> a track count (avg track length), if the user gave a time
        minutes = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h\b)", description.lower())
        if m:
            minutes = int(float(m.group(1)) * 60)
        else:
            m = re.search(r"(\d{2,3})\s*(?:min|minutes?|mins)", description.lower())
            if m:
                minutes = int(m.group(1))
        target_len = length or parsed.get("length")
        if target_len is None and minutes:
            durs = [t.duration for t in self.store.iter_tracks() if t.duration]
            avg = (sum(durs) / len(durs) / 60.0) if durs else 5.5
            target_len = max(1, round(minutes / max(1.0, avg)))
        target_len = int(target_len or 12)

        # 1) LLM path (MuQ-grounded metadata, LLM reasons the order)
        try:
            from mgc.llm import ollama
            from mgc.llm.setbuild import llm_build_set
            if ollama.available():
                cands = self._set_candidates()
                pool = self._set_pool(cands, parsed, target_len)
                res = llm_build_set(description, pool, target_len, minutes=minutes,
                                    model=self.config.llm_model)
                if res and res["track_ids"]:
                    analysis = self.store.load_analysis()
                    names = {c["id"]: c["name"] for c in cands}
                    arc = [float((analysis.get(tid) or {}).get("energy") or 0.5)
                           for tid in res["track_ids"]]
                    return {"track_ids": res["track_ids"],
                            "names": [names.get(t, str(t)) for t in res["track_ids"]],
                            "arc": arc, "reasons": res["reasons"],
                            "parsed": parsed, "engine": f"llm:{res['model']}"}
        except Exception:
            pass

        # 2) heuristic fallback
        return build_set(self.store, description, self.model, length=target_len)

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
                     use_discogs: bool = True, use_musicbrainz: bool = True) -> dict:
        """Recognise each track and store an authoritative genre/artist/title/region.

        Three sources, best of each, tried in order until one hits: (1) AcoustID audio
        fingerprint -> MusicBrainz (exact, but only covers released music); (2) Discogs
        search on the parsed artist/title (its 'styles' are the DJ-grade subgenres, best
        underground coverage); (3) MusicBrainz text search on the parsed name (broader
        free text DB, fills Discogs gaps). Resumable, best-effort: a miss leaves the
        track for the AudioSet fallback in auto_organize."""
        import os

        from mgc.metadata import acoustid as aid, discogs, mb_lookup, mb_lookup_by_mbid
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
            if acoustid_key:  # 1) fingerprint -> MusicBrainz (exact)
                res = aid.identify(t.path, key=acoustid_key)
                mbid = res.get("recording_mbid")
                if mbid:
                    meta = mb_lookup_by_mbid(mbid)
                    genres = meta.get("genres") or []
                    artist = meta.get("artist") or res.get("artist")
                    title = meta.get("title") or res.get("title")
                    area, year, score = meta.get("area"), meta.get("year"), res.get("score")
            if not genres and (have_discogs or use_musicbrainz):
                pa, pt = parse_artist_title(tags.get("title"), tags.get("artist"), t.path)
                if have_discogs:  # 2) Discogs (best styles for electronic)
                    d = discogs.lookup(pa, pt)
                    dg = (d.get("styles") or []) + (d.get("genres") or [])
                    if dg:
                        genres = dg
                        artist, title = artist or pa, title or pt
                        year = year or d.get("year")
                if not genres and use_musicbrainz and pa:  # 3) MusicBrainz text search
                    m = mb_lookup(pa, pt)
                    mg = m.get("genres") or m.get("tags") or []
                    if mg:
                        genres = mg
                        artist, title = artist or pa, title or (m.get("title") or pt)
                        area = area or m.get("area")
                        year = year or m.get("year")
                        mbid = mbid or m.get("recording_mbid")

            if genres or mbid:
                self.store.save_identity(t.id, recording_mbid=mbid, artist=artist,
                                         title=title, genres=genres, area=area,
                                         year=year, score=score)
                identified += 1
            if progress:
                progress(i + 1, len(tracks))
        return {"identified": identified, "total": len(tracks)}

    def llm_classify_all(self, force: bool = False, progress=None, limit=None) -> dict:
        """Fill UN-identified tracks with a validated LLM genre guess (from the label in
        the filename + sound tags + BPM). Only confident, BPM-plausible guesses are kept,
        and existing Discogs/MB identities are preserved (unless ``force``). These become
        anchors for the next Re-sort. Best-effort; needs Ollama."""
        from mgc.llm import ollama
        from mgc.llm.genre import llm_genre
        from mgc.tagging import get_audioset_labels, top_tags

        if not ollama.available():
            return {"error": "no_ollama", "labeled": 0, "total": 0}
        labels = get_audioset_labels() or []
        ana = self.store.load_analysis()
        tracks = self.store.iter_tracks()
        if limit:
            tracks = tracks[:limit]
        labeled = 0
        for i, t in enumerate(tracks):
            idn = self.store.get_identity(t.id)
            if idn and idn.get("genres") and not force:
                if progress:
                    progress(i + 1, len(tracks))
                continue
            u = self.store.get_understanding(t.id) or {}
            tags = ([tg["label"] for tg in top_tags(u["audioset"], labels, k=4)]
                    if u.get("audioset") is not None and labels else [])
            bpm = (ana.get(t.id) or {}).get("bpm")
            res = llm_genre(t.path, tags, bpm, model=self.config.llm_model)
            if res:
                self.store.save_identity(
                    t.id, recording_mbid=(idn or {}).get("recording_mbid"),
                    artist=(idn or {}).get("artist"), title=(idn or {}).get("title"),
                    genres=[res["genre"]], area=(idn or {}).get("area"),
                    year=(idn or {}).get("year"), score=res["confidence"])
                labeled += 1
            if progress:
                progress(i + 1, len(tracks))
        return {"labeled": labeled, "total": len(tracks)}

    def llm_genre_one(self, track_id: int) -> dict:
        """A single-track LLM genre SUGGESTION (unvalidated) for the user to confirm.

        Tries a web-grounded guess first (search the real track, let the LLM name
        the subgenre from those snippets) to avoid memory hallucinations; falls
        back to the ungrounded filename/tags guess if grounding yields nothing."""
        from mgc.llm.genre import llm_genre, llm_genre_grounded
        from mgc.metadata.parse import parse_artist_title
        from mgc.tagging import get_audioset_labels, top_tags
        t = self.store.get_track(track_id)
        if not t:
            return {}
        u = self.store.get_understanding(track_id) or {}
        labels = get_audioset_labels() or []
        tags = ([tg["label"] for tg in top_tags(u["audioset"], labels, k=4)]
                if u.get("audioset") is not None and labels else [])
        bpm = (self.store.get_analysis(track_id) or {}).get("bpm")
        id3 = t.existing_tags or {}
        artist, title = parse_artist_title(id3.get("title"), id3.get("artist"), t.path)
        grounded = llm_genre_grounded(artist, title, id3.get("artist"), tags, bpm,
                                      model=self.config.llm_model)
        if grounded:
            return grounded
        return llm_genre(t.path, tags, bpm, model=self.config.llm_model, validate=False) or {}

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
