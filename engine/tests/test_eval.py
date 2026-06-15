"""Unit tests for mgc.eval.validate.

These exercise ONLY the eval module. The cross-module classifier dependency
(`mgc.classify.classifier.suggest`) is stubbed via sys.modules so the tests
never require the classify module to exist or any heavy deps.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mgc.eval import accuracy_report, project_embeddings
from mgc.types import METHOD_CENTROID, Suggestion


# --------------------------------------------------------------------------- #
# project_embeddings
# --------------------------------------------------------------------------- #
def test_project_embeddings_shape(tmp_store):
    """6 random embeddings project to a (6, 2) coordinate array, ids aligned."""
    rng = np.random.default_rng(0)
    ids = []
    for i in range(6):
        tid = tmp_store.upsert_track(_track(f"t{i}", f"h{i}"))
        ids.append(tid)
        tmp_store.save_embedding(tid, "baseline", rng.standard_normal(32).astype(np.float32))

    got_ids, coords = project_embeddings(tmp_store, "baseline")

    assert got_ids == sorted(ids)            # load_matrix orders by track_id
    assert coords.shape == (6, 2)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()


def test_project_embeddings_n_components(tmp_store):
    """n_components is honored (here 3)."""
    rng = np.random.default_rng(1)
    for i in range(5):
        tid = tmp_store.upsert_track(_track(f"x{i}", f"xh{i}"))
        tmp_store.save_embedding(tid, "baseline", rng.standard_normal(16).astype(np.float32))

    ids, coords = project_embeddings(tmp_store, "baseline", n_components=3)
    assert coords.shape == (5, 3)


def test_project_embeddings_empty(tmp_store):
    """No embeddings -> empty ids and (0, k) coords, no crash."""
    ids, coords = project_embeddings(tmp_store, "baseline")
    assert ids == []
    assert coords.shape == (0, 2)


def test_project_embeddings_unknown_method_falls_back(tmp_store):
    """An unknown / optional method silently falls back to PCA."""
    rng = np.random.default_rng(2)
    for i in range(6):
        tid = tmp_store.upsert_track(_track(f"u{i}", f"uh{i}"))
        tmp_store.save_embedding(tid, "baseline", rng.standard_normal(8).astype(np.float32))

    ids, coords = project_embeddings(tmp_store, "baseline", method="umap")
    assert coords.shape == (6, 2)  # falls back to PCA when umap is unavailable


# --------------------------------------------------------------------------- #
# accuracy_report
# --------------------------------------------------------------------------- #
def test_accuracy_report_perfect(tmp_store, monkeypatch):
    """When the stubbed classifier always nails it, top1 == top3 == 1.0."""
    t_rock = tmp_store.upsert_track(_track("r", "hr"))
    t_jazz = tmp_store.upsert_track(_track("j", "hj"))
    labeled = {t_rock: "Rock", t_jazz: "Jazz"}

    def fake_suggest(store, track_id, model, top_k):
        true = labeled[track_id]
        return [Suggestion(track_id, None, true, 0.9, METHOD_CENTROID)]

    _install_suggest(monkeypatch, fake_suggest)

    rep = accuracy_report(tmp_store, labeled, "baseline")
    assert rep["n"] == 2
    assert rep["top1"] == 1.0
    assert rep["top3"] == 1.0
    assert rep["per_genre"]["Rock"] == {"precision": 1.0, "recall": 1.0}
    assert rep["per_genre"]["Jazz"] == {"precision": 1.0, "recall": 1.0}


def test_accuracy_report_top3_but_not_top1(tmp_store, monkeypatch):
    """True genre is 2nd in the ranking -> counts for top3, not top1."""
    tid = tmp_store.upsert_track(_track("a", "ha"))
    labeled = {tid: "Rock"}

    def fake_suggest(store, track_id, model, top_k):
        return [
            Suggestion(track_id, None, "Pop", 0.6, METHOD_CENTROID),
            Suggestion(track_id, None, "Rock", 0.5, METHOD_CENTROID),
            Suggestion(track_id, None, "Jazz", 0.4, METHOD_CENTROID),
        ]

    _install_suggest(monkeypatch, fake_suggest)

    rep = accuracy_report(tmp_store, labeled, "baseline")
    assert rep["top1"] == 0.0
    assert rep["top3"] == 1.0
    assert rep["n"] == 1


def test_accuracy_report_threshold_filters(tmp_store, monkeypatch):
    """Low-confidence suggestions below threshold are dropped before scoring."""
    tid = tmp_store.upsert_track(_track("a", "ha"))
    labeled = {tid: "Rock"}

    def fake_suggest(store, track_id, model, top_k):
        return [Suggestion(track_id, None, "Rock", 0.1, METHOD_CENTROID)]

    _install_suggest(monkeypatch, fake_suggest)

    rep = accuracy_report(tmp_store, labeled, "baseline", threshold=0.5)
    assert rep["top1"] == 0.0
    assert rep["top3"] == 0.0
    # Rock is never predicted, only the true label -> recall 0, precision 0.
    assert rep["per_genre"]["Rock"] == {"precision": 0.0, "recall": 0.0}


def test_accuracy_report_precision_recall_mix(tmp_store, monkeypatch):
    """A wrong prediction is a false positive for the predicted genre."""
    t1 = tmp_store.upsert_track(_track("1", "h1"))
    t2 = tmp_store.upsert_track(_track("2", "h2"))
    labeled = {t1: "Rock", t2: "Jazz"}

    # t1 (Rock) predicted Rock (correct); t2 (Jazz) predicted Rock (wrong).
    preds = {t1: "Rock", t2: "Rock"}

    def fake_suggest(store, track_id, model, top_k):
        return [Suggestion(track_id, None, preds[track_id], 0.8, METHOD_CENTROID)]

    _install_suggest(monkeypatch, fake_suggest)

    rep = accuracy_report(tmp_store, labeled, "baseline")
    assert rep["top1"] == 0.5
    # Rock: TP=1 (t1), FP=1 (t2) -> precision 0.5; FN=0 -> recall 1.0
    assert rep["per_genre"]["Rock"] == {"precision": 0.5, "recall": 1.0}
    # Jazz: never predicted, one true -> precision 0.0, recall 0.0
    assert rep["per_genre"]["Jazz"] == {"precision": 0.0, "recall": 0.0}


def test_accuracy_report_empty(tmp_store, monkeypatch):
    """No labeled tracks -> zeros, no division by zero."""
    _install_suggest(monkeypatch, lambda *a, **k: [])
    rep = accuracy_report(tmp_store, {}, "baseline")
    assert rep == {"top1": 0.0, "top3": 0.0, "n": 0, "per_genre": {}}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _track(path, h):
    from mgc.types import Track

    return Track(path=path, content_hash=h)


def _install_suggest(monkeypatch, fn):
    """Inject a fake mgc.classify.classifier.suggest into sys.modules."""
    pkg = types.ModuleType("mgc.classify")
    mod = types.ModuleType("mgc.classify.classifier")
    mod.suggest = fn
    pkg.classifier = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mgc.classify", pkg)
    monkeypatch.setitem(sys.modules, "mgc.classify.classifier", mod)
