"""Local HTTP sidecar exposing the Engine to the Tauri/web frontend.

Single-user desktop app: one Engine, all access serialized by a lock, sqlite
opened with check_same_thread=False. Embedding runs on a background thread that
locks per-track (so the UI stays responsive) and reports progress.

Run: ``mgc serve`` (or ``uvicorn mgc.server:app``).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mgc.config import Config
from mgc.store import Store
from mgc.api.service import Engine


# ---- request models ---------------------------------------------------------
class ConfigIn(BaseModel):
    library_root: Optional[str] = None
    db_path: Optional[str] = None
    active_model: Optional[str] = None
    organize_root: Optional[str] = None
    organize_mode: Optional[str] = None
    confidence_threshold: Optional[float] = None


class ByExampleIn(BaseModel):
    name: str
    track_ids: list[int]
    parent_id: Optional[int] = None
    level: str = "subgenre"


class GenreIn(BaseModel):
    name: str
    parent_id: Optional[int] = None
    level: str = "genre"


class ConfirmIn(BaseModel):
    track_id: int
    genre_id: int


class ApplyIn(BaseModel):
    dry_run: bool = True


class SeedIn(BaseModel):
    refs_dir: str


class ImportIn(BaseModel):
    paths: list[str]          # files and/or folders (folders are walked recursively)
    embed: bool = True        # auto-embed the newly added tracks in the background


class UploadIn(BaseModel):
    # browser drag-drop: each file is {name, data_base64} (data URL prefix tolerated)
    files: list[dict]


class IdentifyUploadIn(BaseModel):
    # phone/mic recording: base64 audio identified against the library (NOT imported)
    name: str = "clip.wav"
    data_base64: str
    n: int = 5


class SetBuildIn(BaseModel):
    description: str
    length: Optional[int] = None


class IdentifyIn(BaseModel):
    path: str
    n: int = 5


class MixIn(BaseModel):
    path: str
    window_seconds: float = 15.0
    hop_seconds: float = 7.0


class RegionIn(BaseModel):
    artist: str
    title: Optional[str] = None


class SearchIn(BaseModel):
    query: str
    n: int = 50
    threshold: Optional[float] = None


class MBLookupIn(BaseModel):
    artist: str
    title: Optional[str] = None


class SegmentSearchIn(BaseModel):
    track_id: int
    start: float
    end: float
    n: int = 20


class SegmentSaveIn(BaseModel):
    track_id: int
    start: float
    end: float
    label: Optional[str] = None
    note: Optional[str] = None
    genre_id: Optional[int] = None


class SegmentGenreIn(BaseModel):
    track_id: int
    start: float
    end: float
    name: str
    parent_id: Optional[int] = None
    n: int = 8


def _track_dict(engine: Engine, t) -> dict:
    row = engine.store.get_assignment(t.id)
    genre_name, major_name, sub_name, confidence, status = None, None, None, None, None
    if row is not None and row["genre_id"] is not None:
        g = engine.store.get_genre(row["genre_id"])
        if g is not None:
            # Surface BOTH the major and the subgenre. When the assigned genre is a
            # subgenre, walk up to its parent so the major shows too (was: only the
            # subgenre appeared). `genre` is the combined "Major / Sub" label for the
            # table; `major`/`subgenre` are separate so the UI can sort/filter by each.
            if g.parent_id is not None:
                parent = engine.store.get_genre(g.parent_id)
                major_name = parent.name if parent else None
                sub_name = g.name
                genre_name = f"{major_name} / {sub_name}" if major_name else g.name
            else:
                major_name = g.name
                genre_name = g.name
        confidence = row["confidence"]
        status = row["status"]
    a = engine.store.get_analysis(t.id)
    return {
        "id": t.id, "name": Path(t.path).name, "path": t.path, "fmt": t.fmt,
        "duration": t.duration, "genre": genre_name,
        "major": major_name, "subgenre": sub_name, "confidence": confidence,
        "assignment_status": status, "status": t.status,
        "bpm": a["bpm"] if a else None,
        "music_key": a["music_key"] if a else None,
        "energy": a["energy"] if a else None,
    }


def create_app(config: Optional[Config] = None) -> FastAPI:
    app = FastAPI(title="mgc engine sidecar", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    app.state.config = config or Config()
    app.state.lock = threading.Lock()
    app.state.progress = {"running": False, "done": 0, "total": 0, "last": "", "error": None}
    app.state.tagger = None  # cached AudioSet tagger (lazy, loaded on first /api/tag)
    app.state.engine = Engine(app.state.config, check_same_thread=False)

    def eng() -> Engine:
        return app.state.engine

    # ---- config / health ---------------------------------------------------
    @app.get("/api/health")
    def health():
        e = eng()
        with app.state.lock:
            return {"status": "ok", "model": e.model, "db": e.config.db_path,
                    "library": e.config.library_root, "tracks": e.store.count_tracks(),
                    "genres": len(e.store.iter_genres())}

    @app.get("/api/config")
    def get_config():
        c = app.state.config
        return {"library_root": c.library_root, "db_path": c.db_path,
                "active_model": c.active_model, "organize_root": c.organize_root,
                "organize_mode": c.organize_mode, "confidence_threshold": c.confidence_threshold}

    @app.post("/api/config")
    def set_config(body: ConfigIn):
        c = app.state.config
        for k, v in body.model_dump(exclude_none=True).items():
            setattr(c, k, v)
        # reopen engine against the (possibly new) db/model
        with app.state.lock:
            try:
                app.state.engine.close()
            except Exception:
                pass
            app.state.engine = Engine(c, check_same_thread=False)
        return get_config()

    # ---- ingestion / embedding --------------------------------------------
    @app.post("/api/scan")
    def scan():
        with app.state.lock:
            ids = eng().scan()
            return {"scanned": len(ids), "total": eng().store.count_tracks()}

    def start_embedding(force: bool = False) -> bool:
        """Embed all tracks on a background thread (lock per-track). Returns False
        if an embed run is already in progress."""
        if app.state.progress["running"]:
            return False

        def worker():
            from mgc.embed.cache import embed_track
            e = eng()
            tracks = e.store.iter_tracks()
            p = app.state.progress
            p.update(running=True, done=0, total=len(tracks), last="", error=None)
            for t in tracks:
                try:
                    with app.state.lock:
                        embed_track(e.store, e.embedder, t,
                                    window_seconds=e.config.window_seconds,
                                    hop_seconds=e.config.window_hop_seconds,
                                    max_windows=e.config.max_windows, force=force)
                    p["last"] = Path(t.path).name
                except Exception as ex:  # skip bad files, keep going
                    p["error"] = f"{Path(t.path).name}: {ex}"
                    with app.state.lock:
                        e.store.set_track_status(t.id, "decode_error")
                p["done"] += 1
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/embed")
    def embed(force: bool = False):
        return {"started": start_embedding(force)}

    @app.post("/api/import")
    def import_paths(body: ImportIn):
        """Register dropped files/folders (folders walked recursively, filtered to
        supported audio), then auto-embed in the background. The organize step
        later copies/moves them into the configured Organized-library destination."""
        import os

        import soundfile as sf

        from mgc.ingest.scanner import content_hash, read_tags
        from mgc.types import Track

        exts = {x.lower() for x in app.state.config.extensions}
        files: list[str] = []
        for raw in body.paths:
            p = raw.strip().strip('"')
            if os.path.isdir(p):
                for root, _dirs, names in os.walk(p):
                    for n in names:
                        if os.path.splitext(n)[1].lower() in exts:
                            files.append(os.path.join(root, n))
            elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                files.append(p)

        with app.state.lock:
            e = eng()
            before = e.store.count_tracks()
            for f in files:
                try:
                    info = None
                    try:
                        info = sf.info(f)
                    except Exception:
                        pass
                    e.store.upsert_track(Track(
                        path=f, content_hash=content_hash(f),
                        fmt=os.path.splitext(f)[1].lstrip(".").lower(),
                        duration=getattr(info, "duration", None),
                        sample_rate=getattr(info, "samplerate", None),
                        existing_tags=read_tags(f),
                    ))
                except Exception:
                    pass
            added = e.store.count_tracks() - before

        embedding = start_embedding(force=False) if (body.embed and files) else False
        return {"added": added, "files_seen": len(files),
                "total": eng().store.count_tracks(), "embedding": embedding}

    @app.post("/api/upload")
    def upload(body: UploadIn):
        """Browser drag-drop: receive base64 file bytes, save them under
        <library>/_dropped, then register + embed them (reuses /api/import)."""
        import base64
        import os

        root = app.state.config.library_root or os.getcwd()
        drop = os.path.join(root, "_dropped")
        os.makedirs(drop, exist_ok=True)
        saved: list[str] = []
        for f in body.files:
            name = os.path.basename(str(f.get("name", "")).strip()) or "track"
            b64 = f.get("data_base64") or ""
            try:
                raw = base64.b64decode(b64.split(",")[-1])  # tolerate a data: URL prefix
            except Exception:
                continue
            dest = os.path.join(drop, name)
            try:
                with open(dest, "wb") as fh:
                    fh.write(raw)
                saved.append(dest)
            except Exception:
                continue
        if not saved:
            return {"added": 0, "files_seen": 0, "total": eng().store.count_tracks(), "embedding": False}
        return import_paths(ImportIn(paths=saved, embed=True))

    @app.post("/api/identify-upload")
    def identify_upload(body: IdentifyUploadIn):
        """Identify a recorded clip against the library WITHOUT importing it
        (the phone 'what's this / what are they mixing' primitive)."""
        import base64
        import os
        import tempfile

        try:
            raw = base64.b64decode((body.data_base64 or "").split(",")[-1])
        except Exception:
            return {"matches": []}
        ext = os.path.splitext(body.name)[1] or ".wav"
        fd, tmp = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(tmp, "wb") as fh:
            fh.write(raw)
        try:
            with app.state.lock:
                matches = eng().identify(tmp, n=body.n)
        except Exception as ex:
            return {"matches": [], "error": str(ex)[:160]}
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return {"matches": matches}

    @app.get("/api/progress")
    def progress():
        return app.state.progress

    # ---- tracks / audio ----------------------------------------------------
    @app.get("/api/tracks")
    def tracks(limit: int = 500, offset: int = 0):
        with app.state.lock:
            ts = eng().store.iter_tracks()[offset:offset + limit]
            return [_track_dict(eng(), t) for t in ts]

    @app.get("/api/audio/{track_id}")
    def audio(track_id: int):
        with app.state.lock:
            t = eng().store.get_track(track_id)
        if t is None or not os.path.exists(t.path):
            raise HTTPException(404, "track not found")
        return FileResponse(t.path)

    @app.get("/api/waveform/{track_id}")
    def waveform(track_id: int, bins: int = 480,
                 start: float | None = None, end: float | None = None):
        """RGB spectral waveform: bass/mid/high energy over time for the player bar.
        With start/end (seconds) returns a high-res slice for zoom. Reads the path via a
        throwaway connection so the player keeps working even while a long job holds the
        main lock."""
        import sqlite3
        row = None
        try:
            conn = sqlite3.connect(str(app.state.config.db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT path FROM tracks WHERE id=?", (track_id,)).fetchone()
            conn.close()
        except Exception:
            row = None
        if row is None or not os.path.exists(row["path"]):
            raise HTTPException(404, "track not found")
        from mgc.analysis.waveform import spectral_waveform  # CPU work, outside the lock
        return spectral_waveform(row["path"], bins=bins, start=start, end=end)

    @app.get("/api/spectral-similar/{track_id}")
    def spectral_similar(track_id: int, n: int = 20):
        """Tracks with the most similar frequency fingerprint (bassline/brightness)."""
        with app.state.lock:
            return {"matches": eng().spectral_similar(track_id, n=n)}

    # ---- genres ------------------------------------------------------------
    @app.get("/api/genres")
    def genres():
        with app.state.lock:
            rows = eng().store.conn.execute(
                "SELECT id, name, parent_id, level, source, (centroid IS NOT NULL) AS has_centroid "
                "FROM genres ORDER BY level, name").fetchall()
            return [dict(r) for r in rows]

    @app.post("/api/genres")
    def add_genre(body: GenreIn):
        from mgc.types import GenreNode
        with app.state.lock:
            gid = eng().store.upsert_genre(GenreNode(name=body.name, parent_id=body.parent_id,
                                                     level=body.level, source="custom"))
            return {"genre_id": gid}

    @app.post("/api/genres/by-example")
    def by_example(body: ByExampleIn):
        with app.state.lock:
            gid = eng().add_genre_by_example(body.name, body.track_ids,
                                             parent_id=body.parent_id, level=body.level)
            return {"genre_id": gid}

    # ---- classify / review / confirm --------------------------------------
    @app.post("/api/suggest")
    def suggest():
        with app.state.lock:
            out = eng().suggest_all(persist=True)
            known = sum(1 for v in out.values() if v and v[0].genre_id is not None)
            return {"count": len(out), "known": known}

    @app.get("/api/review")
    def review(limit: int = 50):
        with app.state.lock:
            return eng().review(limit=limit)

    @app.post("/api/confirm")
    def confirm(body: ConfirmIn):
        with app.state.lock:
            eng().confirm(body.track_id, body.genre_id)
            return {"ok": True}

    # ---- similarity / cluster / projection --------------------------------
    @app.get("/api/similar/{track_id}")
    def similar(track_id: int, n: int = 12):
        with app.state.lock:
            out = eng().similar(track_id, n=n)
            res = []
            for tid, score in out:
                t = eng().store.get_track(tid)
                res.append({"track_id": tid, "name": Path(t.path).name if t else str(tid),
                            "score": float(score)})
            return res

    @app.post("/api/cluster")
    def cluster(min_size: int = 2, n_clusters: Optional[int] = None):
        with app.state.lock:
            cl = eng().cluster(min_cluster_size=min_size, n_clusters=n_clusters)
            return [{"cluster_id": c.cluster_id, "size": len(c.member_track_ids),
                     "members": c.member_track_ids, "suggested_genre_id": c.suggested_genre_id}
                    for c in cl]

    @app.get("/api/project")
    def project(method: str = "pca"):
        with app.state.lock:
            ids, coords = eng().project(method=method)
            pts = []
            for i, tid in enumerate(ids):
                row = eng().store.get_assignment(tid)
                gname, major = None, None
                if row is not None and row["genre_id"] is not None:
                    g = eng().store.get_genre(row["genre_id"])
                    if g is not None:
                        gname = g.name
                        if g.parent_id is not None:
                            parent = eng().store.get_genre(g.parent_id)
                            major = parent.name if parent else g.name
                        else:
                            major = g.name
                pts.append({"track_id": tid, "x": float(coords[i][0]),
                            "y": float(coords[i][1]), "genre": gname, "major": major})
            return {"points": pts}

    # ---- output actions ----------------------------------------------------
    @app.post("/api/write-tags")
    def write_tags(body: ApplyIn):
        with app.state.lock:
            plans = eng().write_tags(dry_run=body.dry_run)
            return {"count": len(plans), "plans": plans[:200], "applied": not body.dry_run}

    @app.post("/api/organize")
    def organize(body: ApplyIn):
        with app.state.lock:
            plan = eng().organize(dry_run=body.dry_run)
            return {"count": len(plan), "plan": plan[:200], "applied": not body.dry_run}

    @app.post("/api/auto-organize")
    def auto_organize(dry_run: bool = True, n_groups: int = 14):
        """One-click: cluster the whole library into major genre folders + numbered
        subgenres and (optionally) copy every track into them. Dry-run returns the
        proposed tree without touching disk."""
        with app.state.lock:
            res = eng().auto_organize(n_groups=n_groups, dry_run=dry_run)
            return {"tree": res["tree"], "count": res["count"], "applied": not dry_run}

    @app.post("/api/undo")
    def undo():
        with app.state.lock:
            return eng().undo()

    @app.post("/api/seed-taxonomy")
    def seed(body: SeedIn):
        with app.state.lock:
            return {"seeded": eng().seed_taxonomy(body.refs_dir)}

    # ---- analysis / set-builder / identify / radio -------------------------
    def start_analysis() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            from mgc.analysis import analyze_all
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="analyzing…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                with app.state.lock:
                    analyze_all(eng().store, progress=prog)
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/analyze")
    def analyze():
        """Compute BPM / key / energy for tracks that lack it (background)."""
        return {"started": start_analysis()}

    @app.post("/api/set-build")
    def set_build(body: SetBuildIn):
        with app.state.lock:
            res = eng().build_set(body.description, length=body.length)
            res["names"] = [
                (lambda t: Path(t.path).name if t else str(tid))(eng().store.get_track(tid))
                for tid in res.get("track_ids", [])
            ]
            return res

    @app.post("/api/identify")
    def identify(body: IdentifyIn):
        with app.state.lock:
            return {"matches": eng().identify(body.path, n=body.n)}

    @app.get("/api/radio/{track_id}")
    def radio(track_id: int, n: int = 20):
        with app.state.lock:
            return {"queue": eng().radio(track_id, n=n)}

    @app.post("/api/identify-mix")
    def identify_mix(body: MixIn):
        """Tracklist a whole mix/set against the library, with timestamps."""
        with app.state.lock:
            return {"segments": eng().identify_mix(
                body.path, window_seconds=body.window_seconds, hop_seconds=body.hop_seconds)}

    @app.post("/api/region")
    def region(body: RegionIn):
        """Artist region/origin via MusicBrainz metadata (network, best-effort)."""
        return {"region": eng().region(body.artist, body.title)}

    # ---- AudioSet tagging + open-vocab search ------------------------------
    def start_tagging() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            from mgc.tagging import AudioSetTagger, tag_all
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="tagging…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                if app.state.tagger is None:
                    app.state.tagger = AudioSetTagger()
                with app.state.lock:
                    tag_all(eng().store, tagger=app.state.tagger, progress=prog)
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/tag")
    def tag():
        """Compute AudioSet-527 tags for untagged tracks (background)."""
        return {"started": start_tagging()}

    def start_identifying() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="identifying…", error=None)
            try:
                with app.state.lock:
                    res = eng().identify_all(progress=lambda d, t: p.update(done=d, total=t))
                if res.get("error") == "no_identify_source":
                    p["error"] = "Add ACOUSTID_API_KEY and/or DISCOGS_KEY+DISCOGS_SECRET to app/.env"
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/identify")
    def identify_all_ep():
        """Recognise tracks by audio fingerprint (AcoustID) -> MusicBrainz genre (background)."""
        return {"started": start_identifying()}

    def start_spectral() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="indexing frequencies…", error=None)
            try:
                with app.state.lock:
                    eng().index_spectral(progress=lambda d, t: p.update(done=d, total=t))
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/spectral/index")
    def spectral_index_ep():
        """Compute per-track frequency fingerprints for spectral-similarity search (background)."""
        return {"started": start_spectral()}

    def start_groove() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="analysing groove…", error=None)
            try:
                with app.state.lock:
                    eng().index_groove(progress=lambda d, t: p.update(done=d, total=t))
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/groove/index")
    def groove_index_ep():
        """Compute per-band temporal (groove) features for sharper genre grouping (background)."""
        return {"started": start_groove()}

    @app.post("/api/search")
    def search_ep(body: SearchIn):
        """Open-vocabulary attribute search ('songs with cowbells')."""
        with app.state.lock:
            return eng().search(body.query, n=body.n, threshold=body.threshold)

    @app.get("/api/understanding/{track_id}")
    def understanding_ep(track_id: int):
        with app.state.lock:
            return eng().understanding(track_id) or {"track_id": track_id, "top_tags": []}

    @app.get("/api/llm-genre/{track_id}")
    def llm_genre_ep(track_id: int):
        """A local-LLM genre SUGGESTION for one track (from its label/tags/BPM)."""
        with app.state.lock:
            return eng().llm_genre_one(track_id)

    # ---- deep pass (stem separation + sung-language ID) --------------------
    def start_deep() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            from mgc.deep import deep_analyze_all
            from mgc.tagging import AudioSetTagger
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="deep analysis…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                if app.state.tagger is None:
                    app.state.tagger = AudioSetTagger()
                with app.state.lock:
                    deep_analyze_all(eng().store, tagger=app.state.tagger, progress=prog)
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/deep")
    def deep():
        """Deep-analyze all tagged tracks (stems + language) in the background."""
        return {"started": start_deep()}

    @app.post("/api/deep/{track_id}")
    def deep_one(track_id: int):
        """Deep-analyze a single track now (blocking)."""
        with app.state.lock:
            return eng().deep_analyze(track_id)

    # ---- MusicBrainz: seed genres from authoritative labels ----------------
    def start_mb_seed() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            from mgc.metadata import seed_genres_from_mb
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="seeding genres from MusicBrainz…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                with app.state.lock:
                    created = seed_genres_from_mb(eng().store, eng().model, progress=prog)
                p["last"] = f"seeded {len(created)} genres from MusicBrainz"
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/mb/seed")
    def mb_seed():
        """Seed by-example genres from MusicBrainz labels on your own tracks (background)."""
        return {"started": start_mb_seed()}

    def start_fuse() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="fusing embeddings…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                with app.state.lock:
                    n = eng().fuse(progress=prog)
                p["last"] = f"fused {n} tracks"
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/fuse")
    def fuse_ep():
        """Build fused embeddings (base + CLAP + tags + tempo) for sharper grouping (background)."""
        return {"started": start_fuse()}

    @app.post("/api/mb/lookup")
    def mb_lookup_ep(body: MBLookupIn):
        """MusicBrainz metadata (genres/tags/year/region) for an artist [+ title]."""
        return {"result": eng().mb_lookup(body.artist, body.title)}

    @app.get("/api/genre/related")
    def genre_related(genre: str, n: int = 25):
        """Closely-related genres from the MusicBrainz genre graph (subgenre/fusion)."""
        return {"related": eng().related_genres(genre, n=n)}

    # ---- segment-level similarity (waveform region -> matching parts) ------
    def start_segment_index() -> bool:
        if app.state.progress["running"]:
            return False

        def worker():
            p = app.state.progress
            p.update(running=True, done=0, total=0, last="indexing segments…", error=None)

            def prog(done, total):
                p["done"], p["total"] = done, total

            try:
                with app.state.lock:
                    n = eng().index_segments(progress=prog)
                p["last"] = f"indexed {n} tracks for segment search"
            except Exception as ex:
                p["error"] = str(ex)
            p["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/segment/index")
    def segment_index():
        """Build the per-window segment index for 'find this part elsewhere' (background)."""
        return {"started": start_segment_index()}

    @app.post("/api/segment/search")
    def segment_search(body: SegmentSearchIn):
        """Find tracks containing a part that sounds like [start,end] of a track."""
        with app.state.lock:
            return {"matches": eng().search_by_segment(body.track_id, body.start, body.end, n=body.n)}

    @app.post("/api/segment/save")
    def segment_save(body: SegmentSaveIn):
        """Label a region ('this is the electroclash cowbell') as a segment exemplar."""
        with app.state.lock:
            return eng().save_segment(body.track_id, body.start, body.end,
                                      label=body.label, note=body.note, genre_id=body.genre_id)

    @app.get("/api/segments")
    def segments_list(genre_id: Optional[int] = None):
        with app.state.lock:
            return {"segments": eng().list_segments(genre_id)}

    @app.post("/api/segment/make-genre")
    def segment_make_genre(body: SegmentGenreIn):
        """Define a subgenre by a sound: seed a by-example genre from the tracks
        that contain a part like [start,end] of this track."""
        with app.state.lock:
            return eng().create_genre_from_segment(body.track_id, body.start, body.end,
                                                   body.name, parent_id=body.parent_id, n=body.n)

    @app.get("/api/track/{track_id}/suggestions")
    def track_suggestions(track_id: int):
        """The genre blend (top-N with scores) for a track."""
        with app.state.lock:
            return {"suggestions": eng().track_suggestions(track_id)}

    @app.get("/api/explain")
    def explain(a: int, b: int):
        """Explain why two tracks are similar: what they share and what differs."""
        with app.state.lock:
            return eng().explain_similarity(a, b)

    return app


app = create_app(Config.load(os.environ.get("MGC_CONFIG", "mgc.config.json")))
