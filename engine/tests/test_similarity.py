"""Unit tests for mgc.similarity.

These exercise only the similarity module: inputs are vectors written directly
into the store via save_embedding. No heavy deps, no cross-module imports.
"""

from __future__ import annotations

import numpy as np

from mgc.similarity import nearest_to_vector, similar_tracks
from mgc.similarity.similar import radio_queue
from mgc.types import Track

MODEL = "testmodel"


def _add_track(store, hash_):
    return store.upsert_track(Track(path=f"/x/{hash_}.wav", content_hash=hash_))


def _setup_abc(store):
    """A near-duplicate B of A, plus a dissimilar C. Returns (a, b, c)."""
    a = _add_track(store, "a")
    b = _add_track(store, "b")
    c = _add_track(store, "c")

    va = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # B ~ A (cosine very close to 1) with a tiny perturbation.
    vb = np.array([1.0, 0.01, 0.0, 0.0], dtype=np.float32)
    # C is orthogonal / dissimilar to A.
    vc = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    store.save_embedding(a, MODEL, va)
    store.save_embedding(b, MODEL, vb)
    store.save_embedding(c, MODEL, vc)
    return a, b, c


def test_similar_tracks_ranks_near_duplicate_first(tmp_store):
    a, b, c = _setup_abc(tmp_store)
    res = similar_tracks(tmp_store, a, MODEL, n=10)

    # Query excluded.
    ids = [tid for tid, _ in res]
    assert a not in ids
    assert set(ids) == {b, c}

    # Near-duplicate B ranked first, above dissimilar C.
    assert res[0][0] == b
    score = dict(res)
    assert score[b] > score[c]
    # B is essentially identical to A.
    assert score[b] > 0.99


def test_similar_tracks_respects_n(tmp_store):
    a, b, c = _setup_abc(tmp_store)
    res = similar_tracks(tmp_store, a, MODEL, n=1)
    assert len(res) == 1
    assert res[0][0] == b


def test_similar_tracks_unknown_track_returns_empty(tmp_store):
    _setup_abc(tmp_store)
    assert similar_tracks(tmp_store, 99999, MODEL, n=10) == []


def test_similar_tracks_unknown_model_returns_empty(tmp_store):
    a, _b, _c = _setup_abc(tmp_store)
    assert similar_tracks(tmp_store, a, "no-such-model", n=10) == []


def test_nearest_to_vector_excludes_and_orders(tmp_store):
    a, b, c = _setup_abc(tmp_store)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # A's vector
    res = nearest_to_vector(tmp_store, query, MODEL, n=10, exclude={a})

    ids = [tid for tid, _ in res]
    assert a not in ids
    assert ids[0] == b  # near-duplicate first
    score = dict(res)
    assert score[b] > score[c]


def test_nearest_to_vector_without_exclude_includes_all(tmp_store):
    a, b, c = _setup_abc(tmp_store)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    res = nearest_to_vector(tmp_store, query, MODEL, n=10)
    ids = {tid for tid, _ in res}
    assert ids == {a, b, c}
    # A (exact match) should top the ranking.
    assert res[0][0] == a


def test_scores_are_sorted_descending(tmp_store):
    _setup_abc(tmp_store)
    query = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    res = nearest_to_vector(tmp_store, query, MODEL, n=10)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_returns_empty(tmp_store):
    res = nearest_to_vector(tmp_store, np.ones(4, dtype=np.float32), MODEL, n=5)
    assert res == []


def _setup_two_blobs(store):
    """Two well-separated blobs of 3 tracks each. Returns (blob1_ids, blob2_ids)."""
    blob1, blob2 = [], []
    # Blob 1 clusters tightly around axis 0.
    for i in range(3):
        tid = _add_track(store, f"b1_{i}")
        v = np.array([1.0, 0.02 * i, 0.0, 0.0], dtype=np.float32)
        store.save_embedding(tid, MODEL, v)
        blob1.append(tid)
    # Blob 2 clusters tightly around axis 2 (orthogonal to blob 1).
    for i in range(3):
        tid = _add_track(store, f"b2_{i}")
        v = np.array([0.0, 0.0, 1.0, 0.02 * i], dtype=np.float32)
        store.save_embedding(tid, MODEL, v)
        blob2.append(tid)
    return blob1, blob2


def test_radio_queue_visits_same_blob_first(tmp_store):
    blob1, blob2 = _setup_two_blobs(tmp_store)
    seed = blob1[0]

    queue = radio_queue(tmp_store, seed, MODEL, n=20)

    # Seed is first.
    assert queue[0] == seed
    # All six tracks queued exactly once (n is large enough).
    assert len(queue) == 6
    assert sorted(queue) == sorted(blob1 + blob2)
    # The whole of blob 1 is visited before any of blob 2.
    first_blob2_pos = min(queue.index(t) for t in blob2)
    last_blob1_pos = max(queue.index(t) for t in blob1)
    assert last_blob1_pos < first_blob2_pos


def test_radio_queue_respects_n(tmp_store):
    blob1, _blob2 = _setup_two_blobs(tmp_store)
    seed = blob1[0]
    queue = radio_queue(tmp_store, seed, MODEL, n=3)
    assert len(queue) == 3
    assert queue[0] == seed
    # No duplicates.
    assert len(set(queue)) == 3


def test_radio_queue_unknown_seed_returns_empty(tmp_store):
    _setup_two_blobs(tmp_store)
    assert radio_queue(tmp_store, 99999, MODEL, n=10) == []


def test_radio_queue_unknown_model_returns_empty(tmp_store):
    blob1, _ = _setup_two_blobs(tmp_store)
    assert radio_queue(tmp_store, blob1[0], "no-such-model", n=10) == []
