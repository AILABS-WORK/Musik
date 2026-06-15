"""End-to-end pipeline test on synthetic FLAC audio with the baseline embedder.

scan -> embed -> define 2 genres by example -> suggest -> write tags -> verify
read-back -> organize (copy) -> verify Genre/Subgenre tree -> undo -> verify.
No heavy models; runs entirely on the dependency-free baseline embedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from mgc.config import Config
from mgc.api.service import Engine
from mgc.types import GenreNode
from mgc.actions.tags import read_genre


def _flac(path, freq, seconds=2.0, sr=22050, harmonics=(1,), seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = np.zeros_like(t)
    for h in harmonics:
        sig += np.sin(2 * np.pi * freq * h * t) / h
    sig = 0.4 * sig / np.max(np.abs(sig) + 1e-9) + 0.005 * rng.standard_normal(t.shape)
    sf.write(str(path), sig.astype(np.float32), sr, format="FLAC")
    return str(path)


def test_full_pipeline(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    a = [_flac(lib / f"a{i}.flac", 180 + i, harmonics=(1, 2, 3, 4), seed=i) for i in range(4)]
    b = [_flac(lib / f"b{i}.flac", 2600 + i, harmonics=(1,), seed=100 + i) for i in range(4)]

    cfg = Config(db_path=str(tmp_path / "mgc.sqlite"), library_root=str(lib),
                 active_model="baseline", organize_root=str(tmp_path / "organized"),
                 organize_mode="copy")
    e = Engine(cfg)

    # scan + embed
    ids = e.scan()
    assert len(ids) == 8
    assert e.embed_all() == 8

    # taxonomy: Electronic (genre) > Tech House / Trance (subgenres), defined by example
    electronic = e.store.upsert_genre(GenreNode(name="Electronic", level="genre"))
    # resolve track ids by path
    path_to_id = {t.path: t.id for t in e.store.iter_tracks()}
    a_ids = [path_to_id[p] for p in a]
    b_ids = [path_to_id[p] for p in b]
    th = e.add_genre_by_example("Tech House", a_ids[:2], parent_id=electronic, level="subgenre")
    tr = e.add_genre_by_example("Trance", b_ids[:2], parent_id=electronic, level="subgenre")

    # suggest: non-exemplar tracks should classify to the right subgenre
    out = e.suggest_all(persist=True)
    assert out[a_ids[2]][0].genre_id == th
    assert out[b_ids[2]][0].genre_id == tr

    # write tags (dry-run then real) and verify read-back on a Tech House track
    e.write_tags(dry_run=True)
    assert read_genre(a[2]) in (None, "")  # dry-run wrote nothing
    e.write_tags(dry_run=False)
    assert read_genre(a[2]) == "Tech House"
    assert read_genre(b[2]) == "Trance"

    # organize into Genre/Subgenre tree
    e.organize(dry_run=False)
    organized = tmp_path / "organized"
    assert (organized / "Electronic" / "Tech House").exists()
    assert (organized / "Electronic" / "Trance").exists()
    copied = list(organized.rglob("*.flac"))
    assert len(copied) == 8

    # undo: tags restored, organized copies removed
    res = e.undo()
    assert res["tags"] >= 1
    assert read_genre(a[2]) in (None, "")
    assert list((tmp_path / "organized").rglob("*.flac")) == []

    e.close()
