"""WAV/AIFF tag support + fail-soft behavior (regressions from review)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from mgc.types import Track
from mgc.actions.tags import write_genre, read_genre, undo_tags


def _wav(path, sr=22050):
    t = np.linspace(0, 1, sr, endpoint=False)
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)
    return str(path)


def test_wav_tag_roundtrip_and_undo(tmp_store, tmp_path):
    p = _wav(tmp_path / "x.wav")
    tid = tmp_store.upsert_track(Track(path=p, content_hash="hw", fmt="wav"))
    track = tmp_store.get_track(tid)

    res = write_genre(tmp_store, track, "Tech House")
    assert "error" not in res
    assert read_genre(p) == "Tech House"

    assert undo_tags(tmp_store) >= 1
    assert read_genre(p) in (None, "")


def test_write_genre_failsoft_on_unsupported(tmp_store, tmp_path):
    p = tmp_path / "weird.xyz"
    p.write_bytes(b"not audio")
    tid = tmp_store.upsert_track(Track(path=str(p), content_hash="hx", fmt="xyz"))
    track = tmp_store.get_track(tid)

    res = write_genre(tmp_store, track, "Trance")  # must NOT raise
    assert "error" in res
    assert tmp_store.iter_actions(type="tag_write") == []  # nothing logged for a failed write
