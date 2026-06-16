"""Cosine-similarity nearest-neighbour search over stored embeddings.

Vectors produced by the embed layer are L2-normalized, but we normalize again
here defensively so the cosine ranking is correct regardless of input scale.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (no division by zero)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _rank(
    query: np.ndarray,
    ids: list[int],
    mat: np.ndarray,
    n: int,
    exclude: set,
) -> list[tuple[int, float]]:
    """Cosine of ``query`` against every row of ``mat``; return top ``n``."""
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return []

    q = np.asarray(query, dtype=np.float32).ravel()
    qnorm = np.linalg.norm(q)
    if qnorm == 0.0:
        return []
    q = q / qnorm

    normed = _normalize_rows(mat.astype(np.float32))
    scores = normed @ q

    pairs = [
        (tid, float(score))
        for tid, score in zip(ids, scores)
        if tid not in exclude
    ]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[: max(0, n)]


def similar_tracks(
    store, track_id: int, model: str, n: int = 10
) -> list[tuple[int, float]]:
    """Return up to ``n`` (track_id, cosine_score) most similar to ``track_id``.

    Loads all ``model`` embeddings, scores the query row against every other
    row, excludes the query itself, and sorts by score descending.
    """
    ids, mat = store.load_matrix(model)
    if track_id not in ids:
        return []
    query = mat[ids.index(track_id)]
    return _rank(query, ids, mat, n, exclude={track_id})


def nearest_to_vector(
    store,
    vector,
    model: str,
    n: int = 10,
    exclude: Optional[set] = None,
) -> list[tuple[int, float]]:
    """Return up to ``n`` (track_id, cosine_score) nearest to an arbitrary vector.

    ``exclude`` is an optional set of track ids to omit from the results.
    """
    ids, mat = store.load_matrix(model)
    excl = set(exclude) if exclude else set()
    return _rank(vector, ids, mat, n, exclude=excl)


def radio_queue(
    store, seed_track_id: int, model: str, n: int = 20
) -> list[int]:
    """Build a "radio" play order as a greedy nearest-neighbour walk.

    Starting at ``seed_track_id``, repeatedly step to the cosine-nearest track
    that has not been queued yet, using the *current* track as the query each
    step. This traces a path through embedding space (so neighbours of the seed
    play before far-away clusters), rather than a flat ranking around the seed.

    The seed is included as the FIRST element of the returned list. The result
    has length up to ``n`` (including the seed); it is shorter only when fewer
    than ``n`` tracks with a ``model`` embedding exist. Returns an empty list if
    the seed has no embedding for ``model``.
    """
    if n <= 0:
        return []

    ids, mat = store.load_matrix(model)
    if seed_track_id not in ids:
        return []

    normed = _normalize_rows(mat.astype(np.float32))
    index = {tid: i for i, tid in enumerate(ids)}

    queue: list[int] = [seed_track_id]
    queued: set = {seed_track_id}
    current = seed_track_id

    while len(queue) < n and len(queued) < len(ids):
        q = normed[index[current]]
        if float(np.linalg.norm(q)) == 0.0:
            # Degenerate current vector: fall back to any remaining track.
            remaining = [tid for tid in ids if tid not in queued]
            if not remaining:
                break
            nxt = remaining[0]
        else:
            scores = normed @ q
            best_tid = None
            best_score = -np.inf
            for tid in ids:
                if tid in queued:
                    continue
                s = float(scores[index[tid]])
                if s > best_score:
                    best_score = s
                    best_tid = tid
            if best_tid is None:
                break
            nxt = best_tid
        queue.append(nxt)
        queued.add(nxt)
        current = nxt

    return queue
