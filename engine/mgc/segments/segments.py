"""Segment-level similarity: define a sound by a REGION of a waveform, then find
every track that contains a part that sounds like it.

This is how you pin a subgenre to its actual signature (the cowbell, that synth,
the specific bassline) instead of a whole-track vibe: select a region, embed just
that slice, and rank tracks by their single best-matching window. A precomputed
per-window "segment index" turns the search into one cosine matmul; embedding only
runs at index time and for the query region.
"""

from __future__ import annotations

import os

import numpy as np


def embed_segment(path: str, start: float, end: float, model: str, embedder=None) -> np.ndarray:
    """Embed a [start, end] region (seconds) of a track into one vector."""
    from mgc.audio.decode import load_mono
    from mgc.embed import get_embedder

    embedder = embedder or get_embedder(model)
    sr = embedder.sample_rate
    samples, _ = load_mono(path, sr)
    a = max(0, int(float(start) * sr))
    b = min(len(samples), int(float(end) * sr))
    if b - a < int(0.2 * sr):  # too short to embed; widen to ~1 s
        b = min(len(samples), a + int(1.0 * sr))
    w = samples[a:b]
    if len(w) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(embedder.embed(w, sr), dtype=np.float32).ravel()


def index_track(store, track, model: str, embedder=None, window_s: float = 6.0,
                hop_s: float = 3.0, max_windows: int = 120) -> int:
    """Embed sliding windows of one track and store them in the segment index."""
    from mgc.audio.decode import load_mono
    from mgc.embed import get_embedder

    embedder = embedder or get_embedder(model)
    sr = embedder.sample_rate
    samples, _ = load_mono(track.path, sr)
    win = max(1, int(window_s * sr))
    hop = max(1, int(hop_s * sr))
    store.clear_segment_index(model, track.id)
    n = 0
    for i, s in enumerate(range(0, max(1, len(samples) - win // 2), hop)):
        if i >= max_windows:
            break
        w = samples[s:s + win]
        if len(w) < win * 0.4:
            break
        v = embedder.embed(w, sr)
        store.save_segment_embedding(track.id, model, s / sr, min(s + win, len(samples)) / sr, v)
        n += 1
    return n


def build_segment_index(store, model: str, embedder=None, window_s: float = 6.0,
                        hop_s: float = 3.0, progress=None) -> int:
    """Index every track's windows. Returns the number of tracks indexed."""
    from mgc.embed import get_embedder

    embedder = embedder or get_embedder(model)
    tracks = store.iter_tracks()
    done = 0
    for i, t in enumerate(tracks):
        try:
            index_track(store, t, model, embedder=embedder, window_s=window_s, hop_s=hop_s)
            done += 1
        except Exception:
            pass
        if progress:
            progress(i + 1, len(tracks))
    return done


def find_similar_segments(store, query_vec, model: str, n: int = 20,
                          exclude_track_id=None) -> list[dict]:
    """Rank tracks by their single best window match to ``query_vec``.

    Returns ``[{track_id, name, score, start, end}]`` (the start/end is WHERE in
    each track the matching part is), best first.
    """
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn == 0:
        return []
    q = q / qn
    meta, mat = store.load_segment_index(model)
    if mat.shape[0] == 0:
        return []
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    scores = (mat / np.where(norms == 0, 1.0, norms)) @ q

    best: dict[int, tuple] = {}
    for m, sc in zip(meta, scores):
        tid = m["track_id"]
        if exclude_track_id is not None and tid == exclude_track_id:
            continue
        if tid not in best or float(sc) > best[tid][0]:
            best[tid] = (float(sc), m["start"], m["end"])

    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[: max(0, n)]
    out = []
    for tid, (sc, st, en) in ranked:
        track = store.get_track(tid)
        out.append({"track_id": tid, "name": os.path.basename(track.path) if track else str(tid),
                    "score": round(sc, 3), "start": round(st, 1), "end": round(en, 1)})
    return out
