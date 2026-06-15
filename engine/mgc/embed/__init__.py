"""Embedding backends for mgc.

Public surface:
    - ``get_embedder(name)`` -> an :class:`Embedder` instance.
    - ``l2_normalize`` / ``pool_and_normalize`` vector helpers.
    - ``embed_track`` for cached, windowed per-track embedding.

Only the baseline backend imports light deps eagerly; the heavy backends
(discogs/mert/clap) are imported lazily inside :func:`get_embedder` and raise a
``RuntimeError`` with an install hint when their ML deps are missing.
"""

from __future__ import annotations

from mgc.embed.base import l2_normalize, pool_and_normalize
from mgc.embed.baseline import BaselineEmbedder
from mgc.embed.cache import embed_track
from mgc.types import Embedder

__all__ = [
    "get_embedder",
    "l2_normalize",
    "pool_and_normalize",
    "embed_track",
    "BaselineEmbedder",
]

_KNOWN = {"baseline", "discogs", "mert", "clap"}


def get_embedder(name: str) -> Embedder:
    """Return an embedder instance by ``name``.

    ``"baseline"`` is always available. ``"discogs"``/``"mert"``/``"clap"`` are
    loaded lazily and raise ``RuntimeError`` if their heavy deps are missing.
    Any unknown name raises ``ValueError``.
    """
    key = (name or "").lower()
    if key == "baseline":
        return BaselineEmbedder()
    if key == "discogs":
        from mgc.embed import discogs

        discogs._check_deps()
        return discogs.DiscogsEmbedder()
    if key == "mert":
        from mgc.embed import mert

        mert._check_deps()
        return mert.MertEmbedder()
    if key == "clap":
        from mgc.embed import clap

        clap._check_deps()
        return clap.ClapEmbedder()
    raise ValueError(
        f"Unknown embedder {name!r}. Known: {', '.join(sorted(_KNOWN))}."
    )
