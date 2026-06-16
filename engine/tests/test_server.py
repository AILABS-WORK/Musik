"""API sidecar smoke test via FastAPI TestClient (no running server needed)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mgc.config import Config  # noqa: E402
from mgc.server import create_app  # noqa: E402


def _flac(path, freq, sr=22050, harmonics=(1,), seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    sig = np.zeros_like(t)
    for h in harmonics:
        sig += np.sin(2 * np.pi * freq * h * t) / h
    sig = 0.4 * sig / np.max(np.abs(sig) + 1e-9) + 0.005 * rng.standard_normal(t.shape)
    sf.write(str(path), sig.astype("float32"), sr, format="FLAC")
    return str(path)


def test_server_full_flow(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    for i in range(3):
        _flac(lib / f"a{i}.flac", 180 + i, harmonics=(1, 2, 3, 4), seed=i)
    for i in range(3):
        _flac(lib / f"b{i}.flac", 2600 + i, harmonics=(1,), seed=100 + i)

    cfg = Config(db_path=str(tmp_path / "db.sqlite"), library_root=str(lib),
                 active_model="baseline", organize_root=str(tmp_path / "org"))
    client = TestClient(create_app(cfg))

    assert client.get("/api/health").json()["status"] == "ok"
    assert client.post("/api/scan").json()["scanned"] == 6

    assert client.post("/api/embed").json()["started"] is True
    for _ in range(150):
        p = client.get("/api/progress").json()
        if not p["running"] and p["done"] >= 6:
            break
        time.sleep(0.1)
    p = client.get("/api/progress").json()
    assert p["done"] == 6 and not p["running"]

    tracks = client.get("/api/tracks").json()
    assert len(tracks) == 6
    aids = [t["id"] for t in tracks if t["name"].startswith("a")]
    bids = [t["id"] for t in tracks if t["name"].startswith("b")]

    parent = client.post("/api/genres", json={"name": "Electronic", "level": "genre"}).json()["genre_id"]
    client.post("/api/genres/by-example",
                json={"name": "GroupA", "track_ids": aids[:2], "parent_id": parent, "level": "subgenre"})
    client.post("/api/genres/by-example",
                json={"name": "GroupB", "track_ids": bids[:2], "parent_id": parent, "level": "subgenre"})

    assert client.post("/api/suggest").json()["count"] == 6
    assert len(client.get("/api/genres").json()) == 3

    sim = client.get(f"/api/similar/{aids[2]}").json()
    assert len(sim) >= 1

    assert client.post("/api/write-tags", json={"dry_run": True}).json()["count"] >= 1
    assert client.post("/api/write-tags", json={"dry_run": False}).json()["applied"] is True
    assert client.post("/api/organize", json={"dry_run": False}).json()["count"] >= 1
    u = client.post("/api/undo").json()
    assert "tags" in u and "organize" in u

    assert client.get(f"/api/audio/{aids[0]}").status_code == 200
    assert client.get("/api/review").status_code == 200


def test_import_paths(tmp_path):
    """Drag-and-drop ingestion: /api/import registers files/folders + auto-embeds."""
    drop = tmp_path / "drop"
    drop.mkdir()
    f1 = _flac(drop / "x.flac", 200, harmonics=(1, 2))
    f2 = _flac(drop / "y.flac", 3000)
    empty = tmp_path / "empty"
    empty.mkdir()

    cfg = Config(db_path=str(tmp_path / "db2.sqlite"), library_root=str(empty),
                 active_model="baseline")
    client = TestClient(create_app(cfg))

    # import a FOLDER path -> recurses + filters to audio
    r = client.post("/api/import", json={"paths": [str(drop)]}).json()
    assert r["added"] == 2 and r["files_seen"] == 2 and r["embedding"] is True
    for _ in range(150):
        p = client.get("/api/progress").json()
        if not p["running"] and p["done"] >= 2:
            break
        time.sleep(0.1)
    assert len(client.get("/api/tracks").json()) == 2

    # re-importing the same files (by file path) is idempotent
    r2 = client.post("/api/import", json={"paths": [str(f1), str(f2)]}).json()
    assert r2["added"] == 0
