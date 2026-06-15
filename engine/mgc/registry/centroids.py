"""Genre registry: centroid computation and few-shot custom genres.

A genre's centroid is the L2-normalized mean of its exemplar embeddings (the
"example-derived" prototype used by centroid classification). Custom genres are
created few-shot from a handful of tracks; text-seeded genres get a CLAP text
embedding instead. Nothing here needs heavy ML deps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mgc.types import LEVEL_SUBGENRE


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (unchanged if norm is ~0)."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return (v / norm).astype(np.float32)


def recompute_centroid(store, genre_id: int, model: str) -> Optional[np.ndarray]:
    """Recompute and persist a genre's example-derived centroid.

    The centroid is the L2-normalized mean of the embeddings (for ``model``) of
    the genre's exemplar tracks. Exemplars without an embedding are skipped.
    Returns the centroid, or ``None`` if the genre has no usable exemplars.
    """
    track_ids = store.get_exemplars(genre_id)
    vectors = []
    for tid in track_ids:
        vec = store.get_embedding(tid, model)
        if vec is not None:
            vectors.append(np.asarray(vec, dtype=np.float32).ravel())
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    centroid = _l2_normalize(mean)
    store.set_centroid(genre_id, centroid, is_text=False)
    return centroid


def add_exemplar(store, genre_id: int, track_id: int, model: str) -> Optional[np.ndarray]:
    """Add ``track_id`` as an exemplar of ``genre_id`` and refresh the centroid.

    Returns the updated centroid (or ``None`` if no exemplar has an embedding).
    """
    store.add_exemplar(genre_id, track_id)
    return recompute_centroid(store, genre_id, model)


def create_genre_by_example(
    store,
    name: str,
    track_ids: list[int],
    model: str,
    parent_id: Optional[int] = None,
    level: str = LEVEL_SUBGENRE,
) -> int:
    """Create a custom genre few-shot from example tracks.

    Upserts a ``source="custom"`` genre at the given ``level``/``parent_id``,
    registers all ``track_ids`` as exemplars, recomputes its centroid, and
    returns the new genre id.
    """
    from mgc.types import GenreNode  # local to keep top-level imports minimal

    genre_id = store.upsert_genre(
        GenreNode(name=name, parent_id=parent_id, level=level, source="custom")
    )
    for tid in track_ids:
        store.add_exemplar(genre_id, tid)
    recompute_centroid(store, genre_id, model)
    return genre_id


def seed_by_name(store, genre_id: int, text: str, clap_embedder) -> np.ndarray:
    """Seed a genre's centroid from a text prompt via CLAP.

    ``clap_embedder`` must expose ``text_embed(text) -> np.ndarray``. The result
    is stored as a text-derived centroid (``is_text=True``) and returned.
    """
    vec = np.asarray(clap_embedder.text_embed(text), dtype=np.float32).ravel()
    store.set_centroid(genre_id, vec, is_text=True)
    return vec
