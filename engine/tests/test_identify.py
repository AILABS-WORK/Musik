"""Unit tests for mgc.identify.

A file matched against a library of distinct synthetic tones should rank itself
first. Uses only the baseline embedder + synthetic audio + the temp store; no
heavy models, no network.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mgc.embed import embed_track
from mgc.embed.baseline import BaselineEmbedder
from mgc.identify import identify_external, identify_in_library
from mgc.types import Track

MODEL = "baseline"


@pytest.fixture
def library(tmp_store, tmp_path, make_tone):
    """Build a 3-track library in tmp_store and return (paths, track_ids)."""
    freqs = [200.0, 900.0, 3000.0]
    harmonics = [(1, 2, 3, 4), (1, 2), (1,)]
    embedder = BaselineEmbedder()
    paths: list[str] = []
    ids: list[int] = []
    for i, (freq, harm) in enumerate(zip(freqs, harmonics)):
        p = make_tone(
            tmp_path / f"track{i}.wav",
            freq=freq,
            seconds=3.0,
            harmonics=harm,
            seed=i,
        )
        tid = tmp_store.upsert_track(
            Track(path=p, content_hash=f"hash{i}")
        )
        embed_track(tmp_store, embedder, tmp_store.get_track(tid))
        paths.append(p)
        ids.append(tid)
    return paths, ids


def test_identify_matches_self_best(tmp_store, library):
    paths, ids = library

    result = identify_in_library(tmp_store, paths[0], MODEL)

    assert len(result) >= 1
    assert result[0]["track_id"] == ids[0]
    # name is the basename of the matched track's path.
    assert result[0]["name"] == os.path.basename(paths[0])
    # An exact self-match is essentially cosine 1.0.
    assert result[0]["score"] > 0.99


def test_identify_each_track_matches_itself(tmp_store, library):
    paths, ids = library
    for path, tid in zip(paths, ids):
        result = identify_in_library(tmp_store, path, MODEL)
        assert result[0]["track_id"] == tid


def test_identify_result_shape_and_order(tmp_store, library):
    paths, ids = library
    result = identify_in_library(tmp_store, paths[1], MODEL)

    # Each entry has the documented keys.
    for entry in result:
        assert set(entry.keys()) == {"track_id", "name", "score"}
    # Scores are sorted descending.
    scores = [e["score"] for e in result]
    assert scores == sorted(scores, reverse=True)


def test_identify_respects_n(tmp_store, library):
    paths, _ids = library
    result = identify_in_library(tmp_store, paths[0], MODEL, n=2)
    assert len(result) == 2


def test_identify_empty_library_returns_empty(tmp_store, tmp_path, make_tone):
    # A query file but no embeddings stored under the model.
    p = make_tone(tmp_path / "lonely.wav", freq=440.0, seconds=2.0)
    result = identify_in_library(tmp_store, p, MODEL)
    assert result == []


def test_identify_mix_tracklists_in_order(tmp_store, tmp_path):
    """A mix made by concatenating 3 library tracks is tracklisted in order."""
    import soundfile as sf

    from mgc.identify import identify_mix

    # 8 s tracks (longer than the embed window so there is no zero-padding) and
    # the SAME 4 s window length for both the library embedding and the mix scan,
    # so the baseline embedder's vectors are directly comparable.
    sr, secs, win = 22050, 8.0, 4.0
    embedder = BaselineEmbedder()
    specs = [(220.0, (1, 2, 3, 4)), (880.0, (1, 2)), (3200.0, (1,))]
    sigs, ids = [], []
    for i, (f, h) in enumerate(specs):
        t = np.linspace(0, secs, int(secs * sr), endpoint=False)
        s = sum(np.sin(2 * np.pi * f * k * t) / k for k in h)
        s = (0.4 * s / np.max(np.abs(s))).astype("float32")
        sigs.append(s)
        p = str(tmp_path / f"m{i}.wav")
        sf.write(p, s, sr)
        tid = tmp_store.upsert_track(Track(path=p, content_hash=f"mh{i}"))
        embed_track(tmp_store, embedder, tmp_store.get_track(tid),
                    window_seconds=win, hop_seconds=win)
        ids.append(tid)

    mix_path = str(tmp_path / "mix.wav")
    sf.write(mix_path, np.concatenate(sigs), sr)

    segs = identify_mix(tmp_store, mix_path, MODEL, window_seconds=win, hop_seconds=win)
    assert [s["track_id"] for s in segs] == ids          # 3 segments, in order
    assert segs[0]["start"] == 0.0
    assert 7.0 <= segs[1]["start"] <= 9.0                 # ~the first transition (~8 s)
    for s in segs:
        assert set(s.keys()) == {"start", "end", "track_id", "name", "score"}


def test_lookup_region_empty_offline():
    from mgc.identify import lookup_region

    assert lookup_region("") == {}                        # no artist -> no network, {}


def test_identify_external_raises_without_dependency():
    # pyacoustid is not installed (and/or no API key) in the test env, so this
    # should raise a RuntimeError carrying the install hint.
    with pytest.raises(RuntimeError) as excinfo:
        identify_external("/no/such/file.wav")
    msg = str(excinfo.value)
    assert "ACOUSTID_API_KEY" in msg
    assert "fpcalc" in msg
