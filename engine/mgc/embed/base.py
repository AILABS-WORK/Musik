"""Vector helpers shared by all embedding backends.

Light deps only (numpy). No heavy ML imports here.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def l2_normalize(v) -> np.ndarray:
    """Return ``v`` as a 1-D float32 vector scaled to unit L2 norm.

    A zero (or near-zero) vector is returned unchanged (as float32) to avoid
    division by zero.
    """
    a = np.asarray(v, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(a))
    if norm < _EPS:
        return a
    return (a / norm).astype(np.float32)


def default_layer_weights(n_layers: int) -> np.ndarray:
    """Per-layer weights that emphasize the mid-to-late transformer layers.

    In music SSL models (MERT/MuQ) the low/mid layers carry timbre and the late
    layers carry genre/semantics; a flat mean over all layers blurs both. This is
    a smooth bump peaking around 70% depth with a floor so every layer still
    contributes. Returns a length-``n_layers`` float32 array.
    """
    if n_layers <= 1:
        return np.ones(max(1, n_layers), dtype=np.float32)
    i = np.arange(n_layers, dtype=np.float32)
    center = 0.7 * (n_layers - 1)
    width = max(1.0, n_layers / 3.0)
    return (np.exp(-((i - center) ** 2) / (2 * width * width)) + 0.3).astype(np.float32)


def layer_pool(hidden, weights=None) -> np.ndarray:
    """Pool a per-layer hidden-state stack ``[n_layers, time, dim]`` to one vector.

    Time is always mean-pooled. Layers are combined by ``weights`` (a per-layer
    weight vector) or, when ``weights`` is None, a uniform mean (the classic
    average-all-layers behavior). Returns a 1-D float32 vector of length ``dim``.
    """
    h = np.asarray(hidden, dtype=np.float32)
    if h.ndim != 3:
        raise ValueError(f"layer_pool expects [layers, time, dim], got shape {h.shape}")
    per_layer = h.mean(axis=1)  # [n_layers, dim]
    if weights is None:
        return per_layer.mean(axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)
    s = float(w.sum())
    w = w / s if s else w
    return (per_layer * w[:, None]).sum(axis=0).astype(np.float32)


def pool_and_normalize(window_vecs) -> np.ndarray:
    """Mean-pool per-window vectors then L2-normalize.

    ``window_vecs`` is a list of 1-D vectors or a 2-D array ``[n_windows, dims]``.
    Returns a 1-D float32 unit vector of length ``dims``.
    """
    arr = np.asarray(window_vecs, dtype=np.float32)
    if arr.ndim == 1:
        pooled = arr
    else:
        if arr.shape[0] == 0:
            raise ValueError("pool_and_normalize: no window vectors to pool")
        pooled = arr.mean(axis=0)
    return l2_normalize(pooled)
