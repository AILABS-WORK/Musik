"""Heavy-model integration tests.

Skipped by default (they download model weights). Enable with:
    MGC_TEST_MODELS=1 pytest engine/tests/test_models_integration.py -v
Verified locally: MERT-95M loads + embeds through the engine (dims=768,
unit-norm, deterministic, similar audio closer than different).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MGC_TEST_MODELS") != "1",
    reason="set MGC_TEST_MODELS=1 to run heavy model downloads",
)


def _clip(path, freqs, sr=24000, secs=5.0, seed=0):
    import soundfile as sf

    rng = np.random.default_rng(seed)
    t = np.linspace(0, secs, int(secs * sr), endpoint=False)
    sig = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)
    sig = sig / np.max(np.abs(sig) + 1e-9) * 0.4 + 0.003 * rng.standard_normal(t.shape)
    sf.write(str(path), sig.astype("float32"), sr)
    return str(path)


def test_mert_embeds_through_engine(tmp_path):
    os.environ.setdefault("MGC_MERT_MODEL", "m-a-p/MERT-v1-95M")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from mgc.audio.decode import load_windows
    from mgc.embed import get_embedder
    from mgc.embed.base import pool_and_normalize

    emb = get_embedder("mert")
    a = _clip(tmp_path / "a.wav", [110, 220, 330], seed=1)
    b = _clip(tmp_path / "b.wav", [110, 220, 330], seed=2)  # similar to a
    c = _clip(tmp_path / "c.wav", [1500, 3000], seed=3)     # different

    def emb_file(p):
        ws = load_windows(p, emb.sample_rate, 5.0, 5.0, 2)
        return pool_and_normalize([emb.embed(w, emb.sample_rate) for w in ws])

    va, vb, vc = emb_file(a), emb_file(b), emb_file(c)
    assert va.shape == (emb.dims,)
    assert abs(float(np.linalg.norm(va)) - 1.0) < 1e-3
    assert np.allclose(va, emb_file(a), atol=1e-3)              # deterministic
    assert float(np.dot(va, vb)) > float(np.dot(va, vc))        # similar closer than different
