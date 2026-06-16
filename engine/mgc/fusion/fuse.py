"""Embedding fusion: combine everything we know about a track into one vector.

A single encoder (MERT/MuQ) captures timbre and structure, but it does not see
the named sounds (AudioSet tags), the text-aligned semantics (CLAP), or the
tempo/energy. Fusing them gives the by-example classifier and clustering far more
to separate fine subgenres on. We L2-normalize each block, scale it by a weight,
concatenate, and L2-normalize the whole, so cosine on the fused vector behaves
like a weighted blend of the per-block similarities.

Derived entirely from data already in the store (base embedding + CLAP embedding
+ the AudioSet vector + BPM/key/energy), so it recomputes with no model runs.
Stored under the model name ``"fused"`` and used transparently for classification.
"""

from __future__ import annotations

import numpy as np

FUSED_MODEL = "fused"

# Per-block weights. Base dominates; tags and CLAP add semantic separation;
# tempo/energy is a light nudge.
DEFAULT_WEIGHTS = {"base": 1.0, "clap": 0.6, "tag": 0.6, "meta": 0.3}
_META_DIM = 3  # [energy, danceability, bpm/200]


def _l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _meta_vector(analysis: dict | None) -> np.ndarray:
    a = analysis or {}
    energy = a.get("energy")
    dance = a.get("danceability")
    bpm = a.get("bpm")
    return np.array([
        float(energy) if energy is not None else 0.5,
        float(dance) if dance is not None else 0.5,
        min(1.0, (float(bpm) / 200.0)) if bpm else 0.6,
    ], dtype=np.float32)


def fuse_one(base, clap=None, audioset=None, analysis=None, *, clap_dim=0,
             tag_dim=0, weights=None) -> np.ndarray:
    """Fuse one track's signals into a fixed-length L2-normalized vector.

    ``clap_dim``/``tag_dim`` are the GLOBAL block sizes (0 disables that block);
    missing blocks are zero-filled so every track's fused vector is the same length.
    """
    w = weights or DEFAULT_WEIGHTS
    blocks = [_l2(base) * w["base"]]
    if clap_dim:
        cb = _l2(clap) if (clap is not None and len(clap)) else np.zeros(clap_dim, np.float32)
        blocks.append(cb * w["clap"])
    if tag_dim:
        tb = _l2(audioset) if (audioset is not None and len(audioset)) else np.zeros(tag_dim, np.float32)
        blocks.append(tb * w["tag"])
    blocks.append(_meta_vector(analysis) * w["meta"])
    return _l2(np.concatenate(blocks))


def build_fused(store, base_model: str, out_model: str = FUSED_MODEL,
                weights=None, progress=None) -> int:
    """Build + store a fused vector for every track that has a ``base_model``
    embedding. Globally decides which blocks exist (CLAP/tags only if present at
    all) so every fused vector is the same length. Returns the count written.
    """
    ids, _mat = store.load_matrix(base_model)
    if not ids:
        return 0

    # Decide global block sizes from what the library actually has.
    clap_dim, tag_dim = 0, 0
    for tid in ids:
        if clap_dim == 0:
            c = store.get_embedding(tid, "clap")
            if c is not None and len(c):
                clap_dim = len(c)
        u = store.get_understanding(tid) if hasattr(store, "get_understanding") else None
        if tag_dim == 0 and u and u.get("audioset") is not None:
            tag_dim = len(u["audioset"])
        if clap_dim and tag_dim:
            break

    n = 0
    total = len(ids)
    for i, tid in enumerate(ids):
        base = store.get_embedding(tid, base_model)
        if base is None:
            continue
        clap = store.get_embedding(tid, "clap") if clap_dim else None
        u = store.get_understanding(tid) if hasattr(store, "get_understanding") else None
        audioset = u.get("audioset") if (u and tag_dim) else None
        analysis = store.get_analysis(tid)
        fused = fuse_one(base, clap=clap, audioset=audioset, analysis=analysis,
                         clap_dim=clap_dim, tag_dim=tag_dim, weights=weights)
        store.save_embedding(tid, out_model, fused)
        n += 1
        if progress:
            progress(i + 1, total)
    return n
