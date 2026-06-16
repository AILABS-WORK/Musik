"""Open-vocabulary attribute search ("give me songs with cowbells").

Router:
  - if the query fuzzy-matches a known **AudioSet** class (e.g. "Cowbell",
    "Electric guitar", "Female singing") -> rank the library by that class's
    probability (precise, thresholdable).
  - otherwise -> **CLAP** free-text: embed the prompt and cosine-rank cached CLAP
    audio embeddings, with per-query z-score calibration so results are
    thresholdable ("return ALL matches").

Returns a thresholdable ranked list either way.
"""

from __future__ import annotations

import os

import numpy as np


def _match_label(query: str, labels: list[str]):
    """Best AudioSet label for a free-text query, or None. Tolerant of plurals
    and phrasing ('songs with cowbells' -> 'Cowbell')."""
    q = (query or "").strip().lower()
    if not q or not labels:
        return None
    for i, lab in enumerate(labels):  # exact
        if lab.lower() == q:
            return (i, lab)
    cands = []
    for i, lab in enumerate(labels):
        ll = lab.lower()
        if ll in q or q in ll or ll.rstrip("s") in q or q.rstrip("s") in ll:
            cands.append((len(ll), i, lab))  # prefer the most specific (longest) label
    if cands:
        cands.sort(reverse=True)
        _, i, lab = cands[0]
        return (i, lab)
    return None


def _rank(store, ids, scores, n: int, threshold):
    pairs = sorted(zip(ids, [float(s) for s in scores]), key=lambda p: -p[1])
    if threshold is not None:
        pairs = [p for p in pairs if p[1] >= threshold][:1000]
    else:
        pairs = pairs[: max(1, n)]
    out = []
    for tid, sc in pairs:
        t = store.get_track(tid)
        out.append({"track_id": tid, "name": os.path.basename(t.path) if t else str(tid),
                    "score": round(sc, 3)})
    return out


def _clap_search(store, query: str, n: int, threshold, clap_embedder):
    ids, mat = store.load_matrix("clap")
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return {"results": [], "method": "clap",
                "note": "open-vocab text search needs CLAP audio embeddings — embed with model 'clap'."}
    try:
        if clap_embedder is None:
            from mgc.embed import get_embedder
            clap_embedder = get_embedder("clap")
        prompts = [query, f"this is a sound of {query}", f"a track with prominent {query}"]
        tvecs = [np.asarray(clap_embedder.text_embed(p), dtype=np.float32).ravel() for p in prompts]
        tvec = np.mean(tvecs, axis=0)
        tvec = tvec / (np.linalg.norm(tvec) or 1.0)
    except Exception as e:  # CLAP not installed / failed
        return {"results": [], "method": "clap", "note": f"CLAP unavailable: {str(e)[:120]}"}
    rows = mat.astype(np.float32)
    rows = rows / np.where(np.linalg.norm(rows, axis=1, keepdims=True) == 0, 1.0,
                           np.linalg.norm(rows, axis=1, keepdims=True))
    cos = rows @ tvec
    # per-query z-score calibration -> thresholdable
    z = (cos - cos.mean()) / (cos.std() or 1.0)
    return {"results": _rank(store, ids, z, n, threshold), "method": "clap", "matched_label": None}


def search(store, query: str, n: int = 50, threshold=None, clap_embedder=None) -> dict:
    """Open-vocab attribute search. Returns
    ``{"results":[{track_id,name,score}], "method":"audioset"|"clap", "matched_label":str|None}``."""
    from mgc.tagging import get_audioset_labels

    labels = get_audioset_labels()
    m = _match_label(query, labels) if labels else None
    if m is not None:
        idx, label = m
        ids, mat = store.load_audioset_matrix()
        if mat.shape[0] == 0:
            return {"results": [], "method": "audioset", "matched_label": label,
                    "note": "no AudioSet tags yet — run Tag first."}
        col = mat[:, idx] if idx < mat.shape[1] else np.zeros(len(ids), dtype=np.float32)
        return {"results": _rank(store, ids, col, n, threshold), "method": "audioset",
                "matched_label": label}
    return _clap_search(store, query, n, threshold, clap_embedder)
