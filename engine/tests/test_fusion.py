"""Embedding fusion: fixed-length unit vectors, and fusion fixing a grouping that
the base embedding alone gets wrong."""

from __future__ import annotations

import numpy as np

from mgc.fusion import FUSED_MODEL, build_fused, fuse_one
from mgc.types import Track


def test_fuse_one_is_unit_and_fixed_length():
    base = np.random.default_rng(0).standard_normal(8).astype("float32")
    audioset = np.zeros(4, np.float32)
    audioset[2] = 1.0
    v = fuse_one(base, audioset=audioset, analysis={"energy": 0.7, "bpm": 124}, tag_dim=4)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert v.shape[0] == 8 + 4 + 3  # base + tag + meta(3)


def test_fusion_improves_grouping_via_tags(tmp_store):
    # base(A1) == base(B), base(A2) is only ~near A1 -> base-only would call B the
    # closer track. But A1,A2 share AudioSet tag X while B has tag Y, so fusing the
    # tag vector should pull A1~A2 above A1~B.
    base = np.array([1, 0, 0, 0], np.float32)
    near = np.array([0.99, 0.14, 0, 0], np.float32)
    tag_x = np.array([1, 0], np.float32)
    tag_y = np.array([0, 1], np.float32)
    rows = [("A1", base, tag_x), ("A2", near, tag_x), ("B", base, tag_y)]
    ids = {}
    for name, b, tg in rows:
        tid = tmp_store.upsert_track(Track(path=f"/{name}.wav", content_hash=f"h{name}"))
        tmp_store.save_embedding(tid, "mert", b)
        tmp_store.save_understanding(tid, audioset=tg, audioset_model="ast")
        ids[name] = tid

    assert build_fused(tmp_store, "mert") == 3
    fids, mat = tmp_store.load_matrix(FUSED_MODEL)
    fv = {tid: mat[i] for i, tid in enumerate(fids)}

    def cos(a, b):
        x, y = fv[ids[a]], fv[ids[b]]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    # base-only would say A1~B (identical) > A1~A2; fusion flips it the right way.
    assert cos("A1", "A2") > cos("A1", "B")


def test_layer_pool_mean_vs_weighted():
    from mgc.embed.base import default_layer_weights, layer_pool

    h = np.zeros((3, 2, 4), np.float32)  # [layers, time, dim]
    h[0], h[1], h[2] = 1.0, 2.0, 3.0
    assert np.allclose(layer_pool(h, None), 2.0)        # uniform mean over layers
    w = default_layer_weights(3)
    assert w.shape == (3,) and w[2] > w[0]              # late layers weighted higher
    assert float(layer_pool(h, w).mean()) > 2.0         # weighted pulls toward late layers
