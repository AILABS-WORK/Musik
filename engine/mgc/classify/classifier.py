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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (0.0 when either is degenerate)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def suggest(
    store,
    track_id: int,
    model: str,
    top_k: int = 3,
    threshold: float = 0.35,
    zero_shot: Optional[dict] = None,
) -> list[Suggestion]:
    """Suggest genres for a single track.

    Scores the track's embedding against every genre centroid (cosine). When
    ``zero_shot`` ``{genre_name: score}`` is supplied, scores for matching genre
    names are averaged with the centroid score, and zero-shot-only names are
    added as extra candidates. Candidates are ranked descending by score. If the
    best score is below ``threshold`` a single "unknown / needs review"
    Suggestion (``genre_id=None``) is returned; otherwise up to ``top_k``
    Suggestions are returned.
    """
    emb = store.get_embedding(track_id, model)
    zero_shot = zero_shot or {}

    # Candidate accumulator keyed by genre name.
    # Each entry: {"genre_id", "centroid", "zeroshot"}.
    cands: dict[str, dict] = {}

    if emb is not None:
        for gid, name, vec, _is_text in store.iter_centroids():
            cands.setdefault(name, {"genre_id": gid, "centroid": None, "zeroshot": None})
            cands[name]["genre_id"] = gid
            cands[name]["centroid"] = _cosine(emb, vec)

    for name, score in zero_shot.items():
        entry = cands.setdefault(name, {"genre_id": None, "centroid": None, "zeroshot": None})
        entry["zeroshot"] = float(score)
        if entry["genre_id"] is None:
            g = store.get_genre_by_name(name)
            if g is not None:
                entry["genre_id"] = g.id

    ranked: list[tuple[str, dict, float, str]] = []
    for name, e in cands.items():
        cs, zs = e["centroid"], e["zeroshot"]
        if cs is not None and zs is not None:
            score = (cs + zs) / 2.0
            method = METHOD_ZEROSHOT if zs > cs else METHOD_CENTROID
        elif zs is not None:
            score = zs
            method = METHOD_ZEROSHOT
        elif cs is not None:
            score = cs
            method = METHOD_CENTROID
        else:
            continue
        ranked.append((name, e, score, method))

    if not ranked:
        # Nothing to compare against: treat as unknown / needs review.
        return [Suggestion(track_id=track_id, genre_id=None, genre_name=None,
                           confidence=0.0, method=METHOD_CENTROID)]

    ranked.sort(key=lambda r: r[2], reverse=True)
    best = ranked[0][2]

    if best < threshold:
        return [Suggestion(track_id=track_id, genre_id=None, genre_name=None,
                           confidence=float(best), method=METHOD_CENTROID)]

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
) -> dict[int, list[Suggestion]]:
    """Run :func:`suggest` for every track that has an embedding for ``model``."""
    ids, _mat = store.load_matrix(model)
    return {tid: suggest(store, tid, model, top_k=top_k, threshold=threshold) for tid in ids}


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
