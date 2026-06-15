"""Cluster track embeddings and persist the resulting groups.

Light deps only (numpy + scikit-learn). HDBSCAN is the default; KMeans is the
fallback when HDBSCAN is unavailable, errors out, or is explicitly requested.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from mgc.types import ClusterResult

# HDBSCAN's "noise" label — these points belong to no cluster and are dropped.
_NOISE_LABEL = -1


def _kmeans_labels(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Fallback partitioning with a data-driven choice of k.

    Picks the k in ``[2, k_max]`` with the best silhouette score so natural
    groupings are recovered rather than forced; ``k_max`` is bounded so every
    cluster can plausibly hold ``min_cluster_size`` members. Degenerates to a
    single cluster when there are too few points to split.
    """
    from sklearn.cluster import KMeans

    n = matrix.shape[0]
    # Largest k where each cluster could still hold ``min_cluster_size`` members.
    k_max = max(1, min(n - 1, n // max(2, int(min_cluster_size))))
    if n < 2 or k_max < 2:
        return np.zeros(n, dtype=int)

    from sklearn.metrics import silhouette_score

    best_labels: Optional[np.ndarray] = None
    best_score = -np.inf
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score, best_labels = score, labels

    if best_labels is None:
        return np.zeros(n, dtype=int)
    return best_labels


def _label_points(matrix: np.ndarray, min_cluster_size: int, method: str) -> np.ndarray:
    """Return a per-row cluster label array using HDBSCAN or KMeans."""
    if method == "kmeans":
        return _kmeans_labels(matrix, min_cluster_size)

    try:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), copy=True)
        labels = clusterer.fit_predict(matrix)
        # If HDBSCAN found no real clusters (everything is noise), fall back.
        if not np.any(labels != _NOISE_LABEL):
            return _kmeans_labels(matrix, min_cluster_size)
        return labels
    except Exception:
        return _kmeans_labels(matrix, min_cluster_size)


def _nearest_centroid_genre(
    member_vecs: np.ndarray,
    centroids: list[tuple[int, str, np.ndarray, bool]],
) -> Optional[int]:
    """Most common nearest-centroid genre id across a cluster's members.

    Uses cosine similarity (vectors are expected L2-normalized, but we normalize
    defensively). Returns None when no centroids are available.
    """
    if not centroids:
        return None

    cvecs = np.stack([c[2] for c in centroids]).astype(np.float32)
    cids = [c[0] for c in centroids]

    cnorm = cvecs / (np.linalg.norm(cvecs, axis=1, keepdims=True) + 1e-9)
    mnorm = member_vecs / (np.linalg.norm(member_vecs, axis=1, keepdims=True) + 1e-9)

    sims = mnorm @ cnorm.T  # [n_members, n_centroids]
    nearest = np.argmax(sims, axis=1)
    winner = Counter(int(i) for i in nearest).most_common(1)[0][0]
    return cids[winner]


def cluster_tracks(
    store,
    model: str,
    run_id: str = "run",
    min_cluster_size: int = 2,
    method: str = "hdbscan",
) -> list[ClusterResult]:
    """Cluster all embeddings of ``model`` and persist the groups.

    Loads the embedding matrix via ``store.load_matrix(model)``, clusters with
    HDBSCAN (or KMeans fallback), drops HDBSCAN noise (label -1), groups track
    ids by label, and for each cluster picks ``suggested_genre_id`` as the most
    common nearest-centroid genre among members (or None when no genre centroids
    exist). Persists via ``clear_clusters`` + ``add_cluster``/``add_cluster_member``.
    """
    ids, matrix = store.load_matrix(model)
    if not ids or matrix.size == 0:
        store.clear_clusters()
        return []

    matrix = np.asarray(matrix, dtype=np.float32)
    labels = _label_points(matrix, min_cluster_size, method)

    # Group row indices by label, preserving label order, excluding noise.
    groups: dict[int, list[int]] = {}
    for row_idx, label in enumerate(labels):
        lbl = int(label)
        if lbl == _NOISE_LABEL:
            continue
        groups.setdefault(lbl, []).append(row_idx)

    centroids = store.iter_centroids()

    store.clear_clusters()
    results: list[ClusterResult] = []
    for lbl in sorted(groups):
        rows = groups[lbl]
        member_ids = [ids[r] for r in rows]
        member_vecs = matrix[rows]

        suggested = _nearest_centroid_genre(member_vecs, centroids)

        cid = store.add_cluster(run_id, suggested)
        for tid in member_ids:
            store.add_cluster_member(cid, tid)

        results.append(
            ClusterResult(
                cluster_id=cid,
                member_track_ids=sorted(member_ids),
                suggested_genre_id=suggested,
            )
        )

    return results
