"""Unit tests for the embed module (baseline + dispatch + caching).

Heavy backends are never imported here; cross-module audio decoding is
monkeypatched so the tests exercise only the embed module.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgc.embed import BaselineEmbedder, get_embedder, embed_track
from mgc.embed.base import l2_normalize, pool_and_normalize
from mgc.types import Track

SR = 22050


def _tone(freq, seconds=2.0, sr=SR, harmonics=(1,), seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = np.zeros_like(t)
    for h in harmonics:
        sig += np.sin(2 * np.pi * freq * h * t) / h
    sig = 0.4 * sig / np.max(np.abs(sig) + 1e-9)
    sig += 0.005 * rng.standard_normal(t.shape)
    return sig.astype(np.float32)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---- base helpers -----------------------------------------------------------

def test_l2_normalize_unit_norm():
    v = l2_normalize(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v.dtype == np.float32


def test_l2_normalize_zero_vector_safe():
    v = l2_normalize(np.zeros(5))
    assert np.allclose(v, 0.0)  # no NaNs / div-by-zero


def test_pool_and_normalize_mean_then_unit():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    pooled = pool_and_normalize([a, b])
    assert np.isclose(np.linalg.norm(pooled), 1.0)
    assert np.allclose(pooled, l2_normalize(np.array([0.5, 0.5, 0.0])))


# ---- baseline embedder ------------------------------------------------------

def test_baseline_shape_and_norm():
    emb = BaselineEmbedder()
    v = emb.embed(_tone(440.0), SR)
    assert v.shape == (emb.dims,)
    assert v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_baseline_deterministic():
    emb = BaselineEmbedder()
    sig = _tone(440.0)
    v1 = emb.embed(sig, SR)
    v2 = emb.embed(sig, SR)
    assert np.array_equal(v1, v2)


def test_baseline_separability():
    """Two low rich tones must be closer to each other than to a high pure tone."""
    emb = BaselineEmbedder()
    low_a = emb.embed(_tone(180.0, harmonics=(1, 2, 3, 4), seed=1), SR)
    low_b = emb.embed(_tone(185.0, harmonics=(1, 2, 3, 4), seed=2), SR)
    high = emb.embed(_tone(2600.0, harmonics=(1,), seed=3), SR)
    assert _cos(low_a, low_b) > _cos(low_a, high)


def test_baseline_handles_short_signal():
    emb = BaselineEmbedder()
    v = emb.embed(np.zeros(100, dtype=np.float32), SR)
    assert v.shape == (emb.dims,)


# ---- dispatch ---------------------------------------------------------------

def test_get_embedder_baseline():
    e = get_embedder("baseline")
    assert isinstance(e, BaselineEmbedder)
    assert e.name == "baseline"


def test_get_embedder_heavy_missing_raises_runtime():
    with pytest.raises(RuntimeError):
        get_embedder("mert")


def test_get_embedder_unknown_raises_value():
    with pytest.raises(ValueError):
        get_embedder("nope")


# ---- caching ----------------------------------------------------------------

def test_embed_track_caches(tmp_store, monkeypatch):
    """First call decodes + saves; second call returns cache without decoding."""
    store = tmp_store
    tid = store.upsert_track(Track(path="x.wav", content_hash="h1"))
    track = store.get_track(tid)

    emb = BaselineEmbedder()
    window = _tone(440.0, seconds=1.0)
    calls = {"n": 0}

    import mgc.audio.decode as decode_mod

    def fake_load_windows(path, target_sr, window_seconds, hop_seconds, max_windows):
        calls["n"] += 1
        return [window]

    monkeypatch.setattr(decode_mod, "load_windows", fake_load_windows, raising=False)

    v1 = embed_track(store, emb, track)
    assert calls["n"] == 1
    assert v1.shape == (emb.dims,)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert store.has_embedding(tid, "baseline")

    v2 = embed_track(store, emb, track)
    assert calls["n"] == 1  # NOT decoded again
    assert np.allclose(v1, v2)


def test_embed_track_force_recomputes(tmp_store, monkeypatch):
    store = tmp_store
    tid = store.upsert_track(Track(path="x.wav", content_hash="h2"))
    track = store.get_track(tid)
    emb = BaselineEmbedder()
    calls = {"n": 0}

    import mgc.audio.decode as decode_mod

    def fake_load_windows(path, target_sr, window_seconds, hop_seconds, max_windows):
        calls["n"] += 1
        return [_tone(440.0, seconds=1.0)]

    monkeypatch.setattr(decode_mod, "load_windows", fake_load_windows, raising=False)

    embed_track(store, emb, track)
    embed_track(store, emb, track, force=True)
    assert calls["n"] == 2
