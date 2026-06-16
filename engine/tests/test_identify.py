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


def test_identify_external_raises_without_dependency():
    # pyacoustid is not installed (and/or no API key) in the test env, so this
    # should raise a RuntimeError carrying the install hint.
    with pytest.raises(RuntimeError) as excinfo:
        identify_external("/no/such/file.wav")
    msg = str(excinfo.value)
    assert "ACOUSTID_API_KEY" in msg
    assert "fpcalc" in msg
