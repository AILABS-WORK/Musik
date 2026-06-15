"""Core contract types shared across all modules.

This file is part of the FOUNDATION CONTRACT. Module workers import from here
and MUST NOT modify it. If you need an extra field, raise it for integration
rather than editing the dataclasses unilaterally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Genre tiers. `subset` is the broad top bucket (e.g. "Electronic") used for
# classification/browsing only; the materialized folder tree is genre/subgenre.
LEVEL_SUBSET = "subset"
LEVEL_GENRE = "genre"
LEVEL_SUBGENRE = "subgenre"

# Assignment / suggestion provenance.
METHOD_ZEROSHOT = "zeroshot"
METHOD_CENTROID = "centroid"
METHOD_MANUAL = "manual"

# Action log types.
ACTION_TAG = "tag_write"
ACTION_COPY = "copy"
ACTION_MOVE = "move"


@dataclass
class Track:
    path: str
    content_hash: str
    fmt: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    existing_tags: dict = field(default_factory=dict)
    status: str = "new"
    id: Optional[int] = None


@dataclass
class GenreNode:
    name: str
    parent_id: Optional[int] = None
    level: str = LEVEL_GENRE
    source: str = "seed"  # seed | custom
    description: Optional[str] = None
    threshold: Optional[float] = None
    id: Optional[int] = None


@dataclass
class Suggestion:
    track_id: int
    genre_id: Optional[int]
    genre_name: Optional[str]
    confidence: float
    method: str  # METHOD_*


@dataclass
class ClusterResult:
    cluster_id: int
    member_track_ids: list
    suggested_genre_id: Optional[int] = None


@dataclass
class ActionRecord:
    type: str  # ACTION_*
    track_id: Optional[int]
    from_value: Optional[str]
    to_value: Optional[str]
    undo_token: Optional[str] = None
    status: str = "done"  # done | undone
    ts: Optional[str] = None
    id: Optional[int] = None


class Embedder(ABC):
    """A frozen audio embedding backend.

    Implementations set ``name``, ``dims`` and ``sample_rate`` (the input sample
    rate the model expects — the decode layer resamples to this). ``embed``
    receives a 1-D mono float32 array at ``sample_rate`` and returns a 1-D
    float32 vector of length ``dims``, L2-normalized.
    """

    name: str = "base"
    dims: int = 0
    sample_rate: int = 22050

    @abstractmethod
    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class ZeroShotClassifier(ABC):
    """Optional: a backend that can score a track against named genres directly
    (e.g. Discogs-EffNet genre_discogs400, or CLAP text similarity)."""

    @abstractmethod
    def scores(self, samples: np.ndarray, sr: int) -> dict:  # pragma: no cover - interface
        """Return {genre_name: probability/score}."""
        raise NotImplementedError
