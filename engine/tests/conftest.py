"""Shared test fixtures: synthetic audio + a temp Store.

No heavy ML deps. Synthetic wavs in two well-separated timbral groups let the
baseline embedder, clustering and similarity be tested deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mgc.store import Store


def write_tone(path, freq=440.0, seconds=2.0, sr=22050, harmonics=(1,), seed=0):
    """Write a deterministic mono wav: sum of harmonics of ``freq`` + tiny noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = np.zeros_like(t)
    for h in harmonics:
        sig += np.sin(2 * np.pi * freq * h * t) / h
    sig = 0.4 * sig / np.max(np.abs(sig) + 1e-9)
    sig += 0.005 * rng.standard_normal(t.shape)
    sf.write(str(path), sig.astype(np.float32), sr)
    return str(path)


@pytest.fixture
def make_tone():
    return write_tone


@pytest.fixture
def tmp_store(tmp_path):
    s = Store.open(tmp_path / "test.sqlite")
    yield s
    s.close()


@pytest.fixture
def tmp_library(tmp_path):
    """A small library: group A (low, rich harmonics) vs group B (high, pure)."""
    lib = tmp_path / "lib"
    lib.mkdir()
    paths = {"a": [], "b": []}
    for i in range(4):
        paths["a"].append(write_tone(lib / f"a{i}.wav", freq=180.0 + i, seconds=2.0,
                                     harmonics=(1, 2, 3, 4), seed=i))
    for i in range(4):
        paths["b"].append(write_tone(lib / f"b{i}.wav", freq=2600.0 + i, seconds=2.0,
                                     harmonics=(1,), seed=100 + i))
    return lib, paths
