"""Identify a track by its sound.

Two flavours:

``identify_in_library`` embeds a query audio file with the same baseline (or
named) embedder used to build the library, then ranks it against every stored
embedding of that model by cosine similarity. A file always matches itself
best, so this doubles as an "is this already in my library?" probe.

``identify_external`` is an OPTIONAL acoustic-fingerprint lookup against the
AcoustID web service via the ``pyacoustid`` library + the Chromaprint ``fpcalc``
binary. It is imported lazily and raises a clear ``RuntimeError`` (with an
install hint) when the dependency or API key is missing.

Cross-module deps (audio decode, embed) are imported lazily inside the
functions so this module stays cheap to import and independent at import time.
"""

from __future__ import annotations

import os

import numpy as np


def identify_in_library(
    store,
    path: str,
    model: str,
    n: int = 5,
    window_seconds: float = 5.0,
) -> list[dict]:
    """Identify ``path`` against the library's stored embeddings.

    The query file is decoded into up to 8 non-overlapping windows of
    ``window_seconds`` seconds (at the embedder's native sample rate), each
    window is embedded, and the per-window vectors are mean-pooled and
    L2-normalized into a single query vector. That vector is scored by cosine
    similarity against every embedding stored under ``model``.

    Returns up to ``n`` ``{"track_id", "name", "score"}`` dicts sorted by score
    descending. ``name`` is the basename of the matched track's path. Returns an
    empty list when the library has no embeddings for ``model`` or the query
    produces a degenerate (zero) vector.
    """
    # Lazy cross-module imports: keep identify independent at import time and
    # only pull in audio/embed deps when actually identifying.
    from mgc.audio.decode import load_windows
    from mgc.embed import get_embedder, pool_and_normalize

    embedder = get_embedder(model)

    windows = load_windows(
        path,
        embedder.sample_rate,
        window_seconds,
        window_seconds,
        8,
    )
    if not windows:
        return []

    vec = pool_and_normalize(
        [embedder.embed(w, embedder.sample_rate) for w in windows]
    )

    ids, mat = store.load_matrix(model)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return []

    q = np.asarray(vec, dtype=np.float32).ravel()
    qnorm = float(np.linalg.norm(q))
    if qnorm == 0.0:
        return []
    q = q / qnorm

    rows = mat.astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    scores = (rows / norms) @ q

    ranked = sorted(
        zip(ids, scores), key=lambda p: float(p[1]), reverse=True
    )[: max(0, n)]

    out: list[dict] = []
    for tid, score in ranked:
        track = store.get_track(tid)
        name = os.path.basename(track.path) if track else str(tid)
        out.append({"track_id": tid, "name": name, "score": float(score)})
    return out


def identify_external(path: str) -> dict:
    """Identify ``path`` via the AcoustID acoustic-fingerprint web service.

    OPTIONAL. Requires the ``pyacoustid`` Python package, the Chromaprint
    ``fpcalc`` binary on PATH, and an ``ACOUSTID_API_KEY`` environment variable.
    Raises ``RuntimeError`` with an install hint when any of these is missing.

    Returns a dict with the best match's ``{"score", "recording_id", "title",
    "artist"}`` (fields may be ``None`` when AcoustID has no metadata), or
    ``{"match": None}`` when the fingerprint matches nothing.
    """
    hint = (
        "AcoustID lookup unavailable: pip install pyacoustid + install "
        "Chromaprint fpcalc + set ACOUSTID_API_KEY"
    )

    try:
        import acoustid  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure -> clear hint
        raise RuntimeError(hint) from exc

    api_key = os.environ.get("ACOUSTID_API_KEY")
    if not api_key:
        raise RuntimeError(hint)

    try:
        results = list(acoustid.match(api_key, path))
    except Exception as exc:  # noqa: BLE001 - fpcalc missing / network / etc.
        raise RuntimeError(f"{hint} (lookup failed: {exc})") from exc

    if not results:
        return {"match": None}

    # acoustid.match yields (score, recording_id, title, artist) tuples,
    # best score first.
    score, recording_id, title, artist = results[0]
    return {
        "score": float(score) if score is not None else None,
        "recording_id": recording_id,
        "title": title,
        "artist": artist,
    }
