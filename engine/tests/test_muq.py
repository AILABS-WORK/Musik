"""Unit tests for the MuQ embedding backend (lazy + hinted, no downloads).

These tests NEVER trigger a real model download: the heavy ``_load`` path is
never exercised. We only check the dispatch/lazy-deps contract and the static
metadata, mirroring how the MERT backend is tested in ``test_embed.py``.
"""

from __future__ import annotations

import pytest

from mgc.embed import get_embedder


def test_muq_is_known():
    """``muq`` is registered as a known backend."""
    from mgc.embed import _KNOWN

    assert "muq" in _KNOWN


def test_get_embedder_muq_missing_deps_raises_runtime(monkeypatch):
    """When the heavy deps are missing, ``get_embedder("muq")`` surfaces a
    RuntimeError carrying the install hint. Monkeypatched so it's deterministic
    whether or not torch/muq are installed."""
    from mgc.embed import muq

    def _boom():
        raise RuntimeError(muq._INSTALL_HINT)

    monkeypatch.setattr(muq, "_check_deps", _boom)
    with pytest.raises(RuntimeError, match=r"pip install muq"):
        get_embedder("muq")


def test_get_embedder_muq_instantiates_when_deps_ok(monkeypatch):
    """With deps reported present (monkeypatched), the embedder instantiates
    without loading/downloading a model and exposes the right metadata."""
    from mgc.embed import muq

    monkeypatch.setattr(muq, "_check_deps", lambda: None)
    emb = get_embedder("muq")

    assert isinstance(emb, muq.MuQEmbedder)
    assert emb.name == "muq"
    assert emb.sample_rate == 24000
    # Constructed lazily: no model touched / downloaded.
    assert emb._model is None


def test_muq_default_model_id_and_hint():
    """Static contract: default model id + the documented install hint."""
    from mgc.embed import muq

    assert muq._DEFAULT_MODEL == "OpenMuQ/MuQ-large-msd-iter"
    assert muq._INSTALL_HINT == "MuQ backend needs torch + muq: pip install muq"


def test_muq_model_id_overridable_via_env(monkeypatch):
    """``$MGC_MUQ_MODEL`` overrides the model id at construction time."""
    monkeypatch.setenv("MGC_MUQ_MODEL", "OpenMuQ/MuQ-base")
    import importlib

    from mgc.embed import muq

    muq = importlib.reload(muq)
    try:
        assert muq.MuQEmbedder()._model_id == "OpenMuQ/MuQ-base"
    finally:
        monkeypatch.delenv("MGC_MUQ_MODEL", raising=False)
        importlib.reload(muq)
