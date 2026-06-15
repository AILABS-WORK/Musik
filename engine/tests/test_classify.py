"""Unit tests for mgc.classify (centroid cosine + zero-shot blend, ancestors).

Inputs are built directly in the store as numpy arrays; no cross-module or
heavy ML deps are exercised.
"""

from __future__ import annotations

import numpy as np

from mgc.classify import ancestors, suggest, suggest_all
from mgc.types import (
    LEVEL_GENRE,
    LEVEL_SUBGENRE,
    LEVEL_SUBSET,
    METHOD_CENTROID,
    METHOD_ZEROSHOT,
    GenreNode,
    Track,
)

MODEL = "baseline"
DIMS = 8


def _unit(i: int, dims: int = DIMS) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    v[i] = 1.0
    return v


def _make_track(store, content_hash: str, vector) -> int:
    tid = store.upsert_track(Track(path=f"{content_hash}.wav", content_hash=content_hash))
    store.save_embedding(tid, MODEL, np.asarray(vector, dtype=np.float32))
    return tid


def _two_genres(store) -> tuple[int, int]:
    """Two genres with near-orthogonal unit centroids A=[1,0,..], B=[0,1,..]."""
    ga = store.upsert_genre(GenreNode(name="GenreA", level=LEVEL_GENRE))
    gb = store.upsert_genre(GenreNode(name="GenreB", level=LEVEL_GENRE))
    store.set_centroid(ga, _unit(0))
    store.set_centroid(gb, _unit(1))
    return ga, gb


def test_suggest_top1_centroid_above_threshold(tmp_store):
    s = tmp_store
    ga, _gb = _two_genres(s)
    # Track close to centroid A.
    v = _unit(0).copy()
    v[2] = 0.05  # tiny off-axis component
    tid = _make_track(s, "ta", v)

    sugs = suggest(s, tid, MODEL, top_k=3, threshold=0.35)
    assert sugs[0].genre_id == ga
    assert sugs[0].genre_name == "GenreA"
    assert sugs[0].method == METHOD_CENTROID
    assert sugs[0].confidence > 0.35
    # GenreB should rank below (near-orthogonal).
    assert sugs[0].confidence > sugs[1].confidence


def test_suggest_below_threshold_returns_unknown(tmp_store):
    s = tmp_store
    _two_genres(s)
    # Track nearly orthogonal to both A and B (points along axis 5).
    tid = _make_track(s, "tortho", _unit(5))

    sugs = suggest(s, tid, MODEL, top_k=3, threshold=0.35)
    assert len(sugs) == 1
    only = sugs[0]
    assert only.genre_id is None
    assert only.genre_name is None
    assert only.method == METHOD_CENTROID
    assert only.confidence < 0.35


def test_suggest_respects_top_k(tmp_store):
    s = tmp_store
    # Three genres with three orthogonal-ish centroids.
    ids = []
    for i in range(3):
        gid = s.upsert_genre(GenreNode(name=f"G{i}", level=LEVEL_GENRE))
        s.set_centroid(gid, _unit(i))
        ids.append(gid)
    # Track pointing mostly to G0 but with components on all axes (all positive
    # cosine, so all three are candidates above threshold).
    v = np.array([0.9, 0.6, 0.5, 0, 0, 0, 0, 0], dtype=np.float32)
    tid = _make_track(s, "tk", v)

    sugs = suggest(s, tid, MODEL, top_k=2, threshold=0.1)
    assert len(sugs) == 2
    assert sugs[0].genre_id == ids[0]


def test_suggest_zero_shot_blend_matching_name(tmp_store):
    s = tmp_store
    ga, gb = _two_genres(s)
    # Track near A: centroid cosine to A ~1.0, to B ~0.0.
    tid = _make_track(s, "tz", _unit(0))

    # Zero-shot strongly favors B; blended B = (0 + 0.9)/2 = 0.45,
    # blended A = (1.0 + 0.1)/2 = 0.55 -> A still wins but via centroid signal.
    zs = {"GenreA": 0.1, "GenreB": 0.9}
    sugs = suggest(s, tid, MODEL, top_k=3, threshold=0.35, zero_shot=zs)
    by_name = {x.genre_name: x for x in sugs}
    assert "GenreA" in by_name and "GenreB" in by_name
    assert by_name["GenreA"].confidence == 0.55
    assert by_name["GenreB"].confidence == 0.45
    # A's winning signal is the centroid (1.0 > 0.1).
    assert by_name["GenreA"].method == METHOD_CENTROID
    # B's stronger signal is the zero-shot one (0.9 > 0.0).
    assert by_name["GenreB"].method == METHOD_ZEROSHOT


def test_suggest_zero_shot_only_candidate(tmp_store):
    s = tmp_store
    _two_genres(s)
    # A genre with NO centroid, present only in the store by name.
    gc = s.upsert_genre(GenreNode(name="GenreC", level=LEVEL_GENRE))
    # Track orthogonal to A/B centroids so they stay below threshold.
    tid = _make_track(s, "tzonly", _unit(6))

    zs = {"GenreC": 0.8}
    sugs = suggest(s, tid, MODEL, top_k=3, threshold=0.35, zero_shot=zs)
    top = sugs[0]
    assert top.genre_name == "GenreC"
    assert top.genre_id == gc  # resolved by name even without a centroid
    assert top.method == METHOD_ZEROSHOT
    assert abs(top.confidence - 0.8) < 1e-6


def test_suggest_no_embedding_returns_unknown(tmp_store):
    s = tmp_store
    _two_genres(s)
    tid = s.upsert_track(Track(path="noemb.wav", content_hash="noemb"))
    sugs = suggest(s, tid, MODEL, threshold=0.35)
    assert len(sugs) == 1
    assert sugs[0].genre_id is None


def test_suggest_all_covers_embedded_tracks(tmp_store):
    s = tmp_store
    ga, gb = _two_genres(s)
    t1 = _make_track(s, "all1", _unit(0))
    t2 = _make_track(s, "all2", _unit(1))
    # A track without an embedding should be ignored by suggest_all.
    s.upsert_track(Track(path="bare.wav", content_hash="bare"))

    res = suggest_all(s, MODEL, top_k=1, threshold=0.35)
    assert set(res.keys()) == {t1, t2}
    assert res[t1][0].genre_id == ga
    assert res[t2][0].genre_id == gb


def test_ancestors_subgenre_chain(tmp_store):
    s = tmp_store
    subset = s.upsert_genre(GenreNode(name="Electronic", level=LEVEL_SUBSET))
    genre = s.upsert_genre(GenreNode(name="House", parent_id=subset, level=LEVEL_GENRE))
    subgenre = s.upsert_genre(GenreNode(name="Tech House", parent_id=genre, level=LEVEL_SUBGENRE))

    chain = ancestors(s, subgenre)
    assert [g.id for g in chain] == [genre, subset]
    assert [g.name for g in chain] == ["House", "Electronic"]


def test_ancestors_root_has_none(tmp_store):
    s = tmp_store
    root = s.upsert_genre(GenreNode(name="Electronic", level=LEVEL_SUBSET))
    assert ancestors(s, root) == []


def test_ancestors_unknown_genre(tmp_store):
    s = tmp_store
    assert ancestors(s, 999999) == []
