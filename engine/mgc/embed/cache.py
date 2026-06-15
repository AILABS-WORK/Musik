"""Per-track embedding with caching.

``embed_track`` is the single entry point used by higher layers: it returns a
cached embedding when present, otherwise decodes the track into windows, embeds
each, pools them and persists the result. The audio-decode import is LAZY to
keep the embed module independent of the audio module at import time.
"""

from __future__ import annotations

import numpy as np

from mgc.embed.base import pool_and_normalize
from mgc.types import Embedder, Track


def embed_track(
    store,
    embedder: Embedder,
    track: Track,
    window_seconds: float = 5.0,
    hop_seconds: float = 5.0,
    max_windows: int = 24,
    force: bool = False,
) -> np.ndarray:
    """Return the track's embedding under ``embedder.name``, computing if needed.

    If a cached embedding exists and ``force`` is False, it is returned directly.
    Otherwise the track is decoded into windows at ``embedder.sample_rate``, each
    window is embedded, the windows are mean-pooled + L2-normalized, the result
    is saved to the store and returned.
    """
    if not force and store.has_embedding(track.id, embedder.name):
        return store.get_embedding(track.id, embedder.name)

    # Lazy cross-module import: keeps embed independent of audio at import time.
    from mgc.audio.decode import load_windows

    windows = load_windows(
        track.path,
        target_sr=embedder.sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        max_windows=max_windows,
    )
    window_vecs = [embedder.embed(w, embedder.sample_rate) for w in windows]
    vec = pool_and_normalize(window_vecs)
    store.save_embedding(track.id, embedder.name, vec)
    return vec
