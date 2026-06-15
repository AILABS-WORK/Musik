"""Evaluation utilities for the mgc engine.

Two light, dependency-frugal helpers:

* :func:`project_embeddings` reduces stored embeddings to 2-D (or N-D) for
  visualization. PCA (scikit-learn) is the default and only hard requirement;
  UMAP / t-SNE are used lazily when requested and importable, otherwise we fall
  back to PCA so the call never fails for lack of an optional dependency.
* :func:`accuracy_report` measures top-1 / top-3 classification accuracy and
  per-genre precision/recall against a hand-labeled set, delegating the actual
  scoring to :mod:`mgc.classify` (imported lazily to keep this module light and
  free of circular imports).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def project_embeddings(
    store,
    model: str,
    method: str = "pca",
    n_components: int = 2,
) -> tuple[list[int], np.ndarray]:
    """Reduce stored embeddings for ``model`` to ``n_components`` dimensions.

    Loads the embedding matrix via ``store.load_matrix(model)`` and projects it
    down with the requested ``method``:

    * ``"pca"`` (default) — scikit-learn PCA.
    * ``"umap"`` / ``"tsne"`` — used lazily if the library imports; otherwise we
      transparently fall back to PCA.

    Returns ``(track_ids, coords)`` where ``coords`` has shape
    ``(n_tracks, n_components)``. An empty store yields ``([], zeros((0, k)))``.
    """
    ids, mat = store.load_matrix(model)
    n = mat.shape[0]
    if n == 0:
        return list(ids), np.zeros((0, n_components), dtype=np.float32)

    # Can't ask for more components than samples or features.
    k = max(1, min(n_components, n, mat.shape[1]))

    coords = _reduce(mat.astype(np.float64), method, k)

    # Pad to the requested width if we were forced to use fewer components.
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]), dtype=coords.dtype)
        coords = np.hstack([coords, pad])

    return list(ids), coords.astype(np.float32)


def _reduce(mat: np.ndarray, method: str, k: int) -> np.ndarray:
    """Project ``mat`` to ``k`` dims using ``method`` with a PCA fallback."""
    method = (method or "pca").lower()

    if method == "umap":
        reduced = _try_umap(mat, k)
        if reduced is not None:
            return reduced
    elif method == "tsne":
        reduced = _try_tsne(mat, k)
        if reduced is not None:
            return reduced

    return _pca(mat, k)


def _pca(mat: np.ndarray, k: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=k, random_state=0).fit_transform(mat)


def _try_umap(mat: np.ndarray, k: int) -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    # n_neighbors must be < n_samples for UMAP to be well-defined.
    n_neighbors = max(2, min(15, mat.shape[0] - 1))
    reducer = umap.UMAP(n_components=k, n_neighbors=n_neighbors, random_state=0)
    return np.asarray(reducer.fit_transform(mat))


def _try_tsne(mat: np.ndarray, k: int) -> Optional[np.ndarray]:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    # perplexity must be < n_samples.
    perplexity = max(1.0, min(30.0, float(mat.shape[0] - 1)))
    reducer = TSNE(n_components=k, perplexity=perplexity, random_state=0, init="pca")
    return np.asarray(reducer.fit_transform(mat))


def accuracy_report(
    store,
    labeled: dict[int, str],
    model: str,
    top_k: int = 3,
    threshold: float = 0.0,
) -> dict:
    """Score classifier suggestions against a hand-labeled set.

    ``labeled`` maps ``track_id -> true_genre_name``. For each track we ask the
    classifier (``mgc.classify.classifier.suggest``, imported lazily) for a
    ranked list of suggestions, then measure:

    * ``top1`` — fraction of tracks whose #1 suggestion matches the true genre.
    * ``top3`` — fraction whose true genre appears in the top 3 suggestions.
    * ``n`` — number of scored tracks.
    * ``per_genre`` — ``{name: {"precision": float, "recall": float}}`` computed
      from the top-1 prediction.

    Suggestions with confidence below ``threshold`` are ignored. Tracks for
    which the classifier yields nothing count as misses but are still scored.
    """
    from mgc.classify.classifier import suggest  # lazy: avoids circular import

    n = 0
    top1_hits = 0
    top3_hits = 0

    # Per-genre tallies for precision/recall over the top-1 prediction.
    genres = set(labeled.values())
    tp: dict[str, int] = {g: 0 for g in genres}
    fp: dict[str, int] = {g: 0 for g in genres}
    fn: dict[str, int] = {g: 0 for g in genres}

    for track_id, true_name in labeled.items():
        n += 1
        names = _ranked_names(suggest(store, track_id, model, top_k), threshold, top_k)

        pred1 = names[0] if names else None
        if pred1 == true_name:
            top1_hits += 1
        if true_name in names[:3]:
            top3_hits += 1

        # Per-genre confusion accounting (top-1).
        if pred1 == true_name:
            tp[true_name] = tp.get(true_name, 0) + 1
        else:
            fn[true_name] = fn.get(true_name, 0) + 1
            if pred1 is not None:
                fp[pred1] = fp.get(pred1, 0) + 1

    per_genre: dict[str, dict] = {}
    for g in sorted(set(tp) | set(fp) | set(fn)):
        t, f_p, f_n = tp.get(g, 0), fp.get(g, 0), fn.get(g, 0)
        precision = t / (t + f_p) if (t + f_p) else 0.0
        recall = t / (t + f_n) if (t + f_n) else 0.0
        per_genre[g] = {"precision": precision, "recall": recall}

    return {
        "top1": top1_hits / n if n else 0.0,
        "top3": top3_hits / n if n else 0.0,
        "n": n,
        "per_genre": per_genre,
    }


def _ranked_names(suggestions, threshold: float, top_k: int) -> list[str]:
    """Extract ordered, thresholded genre names from a suggestion list."""
    out: list[str] = []
    for s in suggestions or []:
        name = getattr(s, "genre_name", None)
        conf = float(getattr(s, "confidence", 0.0) or 0.0)
        if name is None or conf < threshold:
            continue
        out.append(name)
        if len(out) >= top_k:
            break
    return out
