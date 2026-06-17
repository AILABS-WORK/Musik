"""Chunk-based grouping ("bag of audio words"): group tracks by the characteristic
SOUNDS they share, not by their whole-track average.

A whole-track embedding averages a song's sound over its whole length, which blurs
the specific elements (a bassline, a hat pattern, a particular synth) that actually
define a subgenre. Here we use the per-window segment index instead: cluster every
window across the library into a vocabulary of "sound words", describe each track as
a histogram over that vocabulary, then group tracks with similar histograms. Two
tracks land together when they are built from the same sounds, which is finer and
more musical than averaging. (This is the classic bag-of-audio-words / VLAD idea.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mgc.types import ClusterResult


def track_histograms(store, model: str, n_words: int = 200, idf: bool = True):
    """Build per-track sound-word histograms from the segment index.

    Returns ``(track_ids, H)`` where ``H[i]`` is track ``track_ids[i]``'s
    L2-normalized (optionally TF-IDF weighted) histogram over a vocabulary of
    ``n_words`` sound-words learned by clustering every indexed window. Returns
    empty arrays when the segment index is empty (index the tracks first).
    """
    meta, mat = store.load_segment_index(model)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[0] == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1.0, norms)

    from sklearn.cluster import KMeans

    k = max(2, min(int(n_words), mat.shape[0]))
    words = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(mat)

    track_ids = sorted({m["track_id"] for m in meta})
    pos = {t: i for i, t in enumerate(track_ids)}
    H = np.zeros((len(track_ids), k), dtype=np.float32)
    for m, w in zip(meta, words):
        H[pos[m["track_id"]], int(w)] += 1.0

    # term frequency (per-track normalize so long tracks don't dominate)
    row = H.sum(axis=1, keepdims=True)
    H = H / np.where(row == 0, 1.0, row)

    # inverse document frequency: downweight ubiquitous sounds (a kick drum is in
    # everything and tells you nothing; a signature synth is rare and tells you a lot)
    if idf:
        df = (H > 0).sum(axis=0)
        idfw = np.log((len(track_ids) + 1.0) / (df + 1.0)) + 1.0
        H = H * idfw

    n2 = np.linalg.norm(H, axis=1, keepdims=True)
    H = H / np.where(n2 == 0, 1.0, n2)
    return track_ids, H


def chunk_group_tracks(store, model: str, run_id: str = "run",
                       n_clusters: int = 12, n_words: int = 200) -> list[ClusterResult]:
    """Group tracks by shared sound-words and persist the groups.

    Mirrors ``cluster.cluster_tracks`` (clears + re-persists clusters) but groups on
    the bag-of-audio-words histograms instead of whole-track vectors. Requires the
    segment index to be built for ``model`` first (see ``segments.build_segment_index``).
    """
    track_ids, H = track_histograms(store, model, n_words=n_words)
    if not track_ids:
        store.clear_clusters()
        return []

    if len(track_ids) >= 2:
        from sklearn.cluster import KMeans

        k = max(2, min(int(n_clusters), len(track_ids)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(H)
    else:
        labels = np.zeros(len(track_ids), dtype=int)

    groups: dict[int, list[int]] = {}
    for tid, lbl in zip(track_ids, labels):
        groups.setdefault(int(lbl), []).append(tid)

    store.clear_clusters()
    results: list[ClusterResult] = []
    for lbl in sorted(groups):
        cid = store.add_cluster(run_id, None)
        for tid in groups[lbl]:
            store.add_cluster_member(cid, tid)
        results.append(ClusterResult(cluster_id=cid,
                                     member_track_ids=sorted(groups[lbl]),
                                     suggested_genre_id=None))
    return results
