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
