"""Genre registry: centroids and few-shot custom genres."""

from __future__ import annotations

from mgc.registry.centroids import (
    add_exemplar,
    create_genre_by_example,
    recompute_centroid,
    seed_by_name,
)

__all__ = [
    "recompute_centroid",
    "add_exemplar",
    "create_genre_by_example",
    "seed_by_name",
]
