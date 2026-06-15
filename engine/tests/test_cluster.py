"""Unit tests for mgc.cluster — exercises only the cluster module.

Inputs are built directly in the store via save_embedding (no audio/embedding
modules involved). Two clearly separated blobs of vectors must produce exactly
two clusters whose members match the original blobs.
"""

from __future__ import annotations

import numpy as np

from mgc.cluster import cluster_tracks
from mgc.types import ClusterResult, GenreNode
from mgc.store.db import Store
from mgc.types import Track

MODEL = "baseline"
DIMS = 8


def _add_track(store: Store, name: str) -> int:
    """Insert a minimal track row and return its id."""
    return store.upsert_track(Track(path=f"/lib/{name}.wav", content_hash=name))


def _two_blobs(store: Store, n_per: int = 10, seed: int = 0):
    """Insert n_per vectors near e0 (blob A) and n_per near e1 (blob B).

    Returns (ids_a, ids_b): the track ids belonging to each blob.
    """
    rng = np.random.default_rng(seed)
    ids_a, ids_b = [], []
    centers = {
        "a": np.eye(DIMS, dtype=np.float32)[0],
        "b": np.eye(DIMS, dtype=np.float32)[1],
    }
    for i in range(n_per):
        for tag, bucket in (("a", ids_a), ("b", ids_b)):
            v = centers[tag] + 0.02 * rng.standard_normal(DIMS).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            tid = _add_track(store, f"{tag}{i}")
            store.save_embedding(tid, MODEL, v)
            bucket.append(tid)
    return ids_a, ids_b


def test_two_separated_blobs_form_two_clusters(tmp_store):
    ids_a, ids_b = _two_blobs(tmp_store, n_per=10)

    results = cluster_tracks(tmp_store, MODEL, run_id="run1", min_cluster_size=2)

    assert isinstance(results, list)
    assert all(isinstance(r, ClusterResult) for r in results)
    assert len(results) == 2, f"expected 2 clusters, got {len(results)}"

    # Members must be grouped by blob: each cluster is exactly one of the blobs.
    member_sets = [set(r.member_track_ids) for r in results]
    expected = [set(ids_a), set(ids_b)]
    for ms in member_sets:
        assert ms in expected, f"cluster {ms} does not match a blob"
    # And the two clusters are distinct blobs.
    assert member_sets[0] != member_sets[1]

    # No centroids inserted -> suggested_genre_id is None for every cluster.
    assert all(r.suggested_genre_id is None for r in results)


def test_results_are_persisted(tmp_store):
    _two_blobs(tmp_store, n_per=10)

    results = cluster_tracks(tmp_store, MODEL, run_id="runX", min_cluster_size=2)

    persisted = tmp_store.get_clusters(run_id="runX")
    assert len(persisted) == len(results) == 2

    got = sorted(tuple(sorted(c.member_track_ids)) for c in persisted)
    want = sorted(tuple(sorted(r.member_track_ids)) for r in results)
    assert got == want


def test_clear_clusters_between_runs(tmp_store):
    _two_blobs(tmp_store, n_per=10)

    cluster_tracks(tmp_store, MODEL, run_id="first", min_cluster_size=2)
    # A second run with a different run_id must wipe the previous clusters.
    cluster_tracks(tmp_store, MODEL, run_id="second", min_cluster_size=2)

    assert tmp_store.get_clusters(run_id="first") == []
    assert len(tmp_store.get_clusters(run_id="second")) == 2


def test_empty_store_returns_no_clusters(tmp_store):
    results = cluster_tracks(tmp_store, MODEL, run_id="empty")
    assert results == []
    assert tmp_store.get_clusters() == []


def test_kmeans_method_groups_blobs(tmp_store):
    ids_a, ids_b = _two_blobs(tmp_store, n_per=10)

    results = cluster_tracks(
        tmp_store, MODEL, run_id="km", min_cluster_size=2, method="kmeans"
    )

    assert len(results) == 2
    member_sets = [set(r.member_track_ids) for r in results]
    assert {frozenset(m) for m in member_sets} == {frozenset(ids_a), frozenset(ids_b)}


def test_suggested_genre_from_centroids(tmp_store):
    ids_a, ids_b = _two_blobs(tmp_store, n_per=10)

    # Insert one genre centroid aligned with blob A's center (e0) and one with
    # blob B's center (e1). Each cluster should be tagged with the nearer genre.
    ga = tmp_store.upsert_genre(GenreNode(name="GenreA"))
    gb = tmp_store.upsert_genre(GenreNode(name="GenreB"))
    e = np.eye(DIMS, dtype=np.float32)
    tmp_store.set_centroid(ga, e[0])
    tmp_store.set_centroid(gb, e[1])

    results = cluster_tracks(tmp_store, MODEL, run_id="cent", min_cluster_size=2)
    assert len(results) == 2

    by_members = {frozenset(r.member_track_ids): r.suggested_genre_id for r in results}
    assert by_members[frozenset(ids_a)] == ga
    assert by_members[frozenset(ids_b)] == gb
