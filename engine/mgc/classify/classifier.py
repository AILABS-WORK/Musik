"""Centroid + zero-shot genre suggestion.

Given a track's embedding, score it against every genre centroid (cosine
similarity) and, optionally, blend in zero-shot scores keyed by genre name.
Below the confidence threshold a single "needs review" suggestion (with
``genre_id=None``) is returned instead of a guess.

Light deps only (numpy + stdlib). No heavy ML imports here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mgc.types import METHOD_CENTROID, METHOD_ZEROSHOT, GenreNode, Suggestion

# Provenance label for k-NN-over-exemplars scoring.
METHOD_KNN = "knn"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (0.0 when either is degenerate)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _knn_candidates(store, emb, model: str) -> dict[str, dict]:
    """Score every genre by the mean cosine of ``emb`` to its exemplar embeddings.

    For each genre node, gather the embeddings of its exemplar tracks
    (``store.get_exemplars(gid)`` -> ``store.get_embedding(tid, model)``), drop
    any that are missing, and use the mean cosine similarity as that genre's
    score. Genres with no usable exemplars are skipped. Returns a candidate dict
    keyed by genre name in the same shape used by the centroid path, but with the
    score carried under the ``"knn"`` key.
    """
    cands: dict[str, dict] = {}
    if emb is None:
        return cands
    for g in store.iter_genres():
        exemplar_ids = store.get_exemplars(g.id)
        if not exemplar_ids:
            continue
        sims = []
        for tid in exemplar_ids:
            vec = store.get_embedding(tid, model)
            if vec is None:
                continue
            sims.append(_cosine(emb, vec))
        if not sims:
            continue
        entry = cands.setdefault(
            g.name, {"genre_id": g.id, "centroid": None, "zeroshot": None, "knn": None}
        )
        entry["genre_id"] = g.id
        entry["knn"] = float(np.mean(sims))
    return cands


def suggest(
    store,
    track_id: int,
    model: str,
    top_k: int = 3,
    threshold: float = 0.35,
    zero_shot: Optional[dict] = None,
    method: str = "centroid",
) -> list[Suggestion]:
    """Suggest genres for a single track.

    Two scoring backends are available via ``method``:

    * ``"centroid"`` (default): score the track's embedding against every genre
      centroid (cosine). When ``zero_shot`` ``{genre_name: score}`` is supplied,
      scores for matching genre names are averaged with the centroid score, and
      zero-shot-only names are added as extra candidates.
    * ``"knn"``: score each genre by the *mean cosine* of the track embedding to
      that genre's exemplar embeddings (``store.get_exemplars`` ->
      ``store.get_embedding``). ``zero_shot`` is blended in the same way as for
      centroids when supplied.

    Candidates are ranked descending by score. If the best score is below
    ``threshold`` a single "unknown / needs review" Suggestion
    (``genre_id=None``) is returned; otherwise up to ``top_k`` Suggestions are
    returned. The fallback/unknown method label matches the requested ``method``.
    """
    emb = store.get_embedding(track_id, model)
    zero_shot = zero_shot or {}
    use_knn = method == "knn"
    # Method label used for the "unknown / needs review" sentinel.
    fallback_method = METHOD_KNN if use_knn else METHOD_CENTROID

    # Candidate accumulator keyed by genre name.
    # Each entry: {"genre_id", "centroid", "zeroshot", "knn"}.
    if use_knn:
        cands = _knn_candidates(store, emb, model)
    else:
        cands = {}
        if emb is not None:
            for gid, name, vec, _is_text in store.iter_centroids():
                cands.setdefault(
                    name, {"genre_id": gid, "centroid": None, "zeroshot": None, "knn": None}
                )
                cands[name]["genre_id"] = gid
                cands[name]["centroid"] = _cosine(emb, vec)

    for name, score in zero_shot.items():
        entry = cands.setdefault(
            name, {"genre_id": None, "centroid": None, "zeroshot": None, "knn": None}
        )
        entry["zeroshot"] = float(score)
        if entry["genre_id"] is None:
            g = store.get_genre_by_name(name)
            if g is not None:
                entry["genre_id"] = g.id

    # The embedding-derived signal is either the centroid cosine or the k-NN
    # mean cosine depending on the requested method.
    primary_key = "knn" if use_knn else "centroid"
    primary_method = METHOD_KNN if use_knn else METHOD_CENTROID

    ranked: list[tuple[str, dict, float, str]] = []
    for name, e in cands.items():
        ps, zs = e[primary_key], e["zeroshot"]
        if ps is not None and zs is not None:
            score = (ps + zs) / 2.0
            method_label = METHOD_ZEROSHOT if zs > ps else primary_method
        elif zs is not None:
            score = zs
            method_label = METHOD_ZEROSHOT
        elif ps is not None:
            score = ps
            method_label = primary_method
        else:
            continue
        ranked.append((name, e, score, method_label))

    if not ranked:
        # Nothing to compare against: treat as unknown / needs review.
        return [Suggestion(track_id=track_id, genre_id=None, genre_name=None,
                           confidence=0.0, method=fallback_method)]

    ranked.sort(key=lambda r: r[2], reverse=True)
    best = ranked[0][2]

    if best < threshold:
        return [Suggestion(track_id=track_id, genre_id=None, genre_name=None,
                           confidence=float(best), method=fallback_method)]

    out: list[Suggestion] = []
    for name, e, score, method in ranked[: max(0, top_k)]:
        out.append(Suggestion(
            track_id=track_id,
            genre_id=e["genre_id"],
            genre_name=name,
            confidence=float(score),
            method=method,
        ))
    return out


def suggest_all(
    store,
    model: str,
    top_k: int = 3,
    threshold: float = 0.35,
    zero_shot: Optional[dict] = None,
    method: str = "centroid",
) -> dict[int, list[Suggestion]]:
    """Run :func:`suggest` for every track that has an embedding for ``model``.

    ``method`` ("centroid" | "knn") and ``zero_shot`` are passed through to
    :func:`suggest` for each track.
    """
    ids, _mat = store.load_matrix(model)
    return {
        tid: suggest(
            store, tid, model, top_k=top_k, threshold=threshold,
            zero_shot=zero_shot, method=method,
        )
        for tid in ids
    }


def ancestors(store, genre_id: int) -> list[GenreNode]:
    """Return the parent chain of ``genre_id`` nearest-first (genre then subset).

    Used to roll a subgenre up to its parent folders. The starting node itself
    is not included; only its ancestors.
    """
    out: list[GenreNode] = []
    node = store.get_genre(genre_id)
    if node is None:
        return out
    seen = {genre_id}
    parent_id = node.parent_id
    while parent_id is not None and parent_id not in seen:
        parent = store.get_genre(parent_id)
        if parent is None:
            break
        out.append(parent)
        seen.add(parent_id)
        parent_id = parent.parent_id
    return out
