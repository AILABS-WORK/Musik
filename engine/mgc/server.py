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


def _track_dict(engine: Engine, t) -> dict:
    row = engine.store.get_assignment(t.id)
    genre_name, confidence, status = None, None, None
    if row is not None and row["genre_id"] is not None:
        g = engine.store.get_genre(row["genre_id"])
        genre_name = g.name if g else None
        confidence = row["confidence"]
        status = row["status"]
    a = engine.store.get_analysis(t.id)
    return {
        "id": t.id, "name": Path(t.path).name, "path": t.path, "fmt": t.fmt,
        "duration": t.duration, "genre": genre_name, "confidence": confidence,
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
    def cluster(min_size: int = 2):
        with app.state.lock:
            cl = eng().cluster(min_cluster_size=min_size)
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
                gname = None
                if row is not None and row["genre_id"] is not None:
                    g = eng().store.get_genre(row["genre_id"])
                    gname = g.name if g else None
                pts.append({"track_id": tid, "x": float(coords[i][0]),
                            "y": float(coords[i][1]), "genre": gname})
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

    @app.post("/api/search")
    def search_ep(body: SearchIn):
        """Open-vocabulary attribute search ('songs with cowbells')."""
        with app.state.lock:
            return eng().search(body.query, n=body.n, threshold=body.threshold)

    @app.get("/api/understanding/{track_id}")
    def understanding_ep(track_id: int):
        with app.state.lock:
            return eng().understanding(track_id) or {"track_id": track_id, "top_tags": []}

    return app


app = create_app(Config.load(os.environ.get("MGC_CONFIG", "mgc.config.json")))
