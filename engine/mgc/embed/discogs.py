"""Discogs-EffNet embedding backend (Essentia).

Wraps Essentia's ``TensorflowPredictEffnetDiscogs`` to produce 1280-dim
embeddings and, optionally, ``genre_discogs400`` zero-shot genre scores. All
heavy imports are LAZY so that ``import mgc.embed.discogs`` stays light.

Model weights are NOT bundled; the backend expects the Essentia graph files to
be available locally (downloaded from the Essentia models repository).
"""

from __future__ import annotations

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder, ZeroShotClassifier

_INSTALL_HINT = (
    "Discogs-EffNet backend requires Essentia + TensorFlow. "
    "Install with: pip install 'mgc-engine[models]'  (and download the "
    "discogs-effnet-bs64 / genre_discogs400 graphs from the Essentia models hub)."
)


def _check_deps() -> None:
    """Raise RuntimeError with an install hint if Essentia is unavailable."""
    try:
        import essentia.standard  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e


class DiscogsEmbedder(Embedder, ZeroShotClassifier):
    """Essentia Discogs-EffNet embeddings (1280-d) + genre_discogs400 scores."""

    name = "discogs"
    sample_rate = 16000
    dims = 1280

    def __init__(
        self,
        embed_graph: str = "discogs-effnet-bs64-1.pb",
        genre_graph: str | None = "genre_discogs400-discogs-effnet-1.pb",
        labels: list[str] | None = None,
    ) -> None:
        self._embed_graph = embed_graph
        self._genre_graph = genre_graph
        self._labels = labels
        self._embed_model = None
        self._genre_model = None

    def _load_embed(self):
        if self._embed_model is None:
            try:
                from essentia.standard import TensorflowPredictEffnetDiscogs  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._embed_model = TensorflowPredictEffnetDiscogs(
                graphFilename=self._embed_graph, output="PartitionedCall:1"
            )
        return self._embed_model

    def _load_genre(self):
        if self._genre_model is None:
            try:
                from essentia.standard import TensorflowPredict2D  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._genre_model = TensorflowPredict2D(
                graphFilename=self._genre_graph,
                input="serving_default_model_Placeholder",
                output="PartitionedCall:0",
            )
        return self._genre_model

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Mean-pool EffNet embeddings over time -> L2-normalized 1280-d vector."""
        model = self._load_embed()
        x = np.asarray(samples, dtype=np.float32).ravel()
        emb = np.asarray(model(x), dtype=np.float32)  # [n_patches, 1280]
        if emb.ndim == 1:
            pooled = emb
        else:
            pooled = emb.mean(axis=0)
        return l2_normalize(pooled)

    def scores(self, samples: np.ndarray, sr: int) -> dict:
        """Return {genre_name: probability} from genre_discogs400."""
        if self._genre_graph is None:
            raise RuntimeError("DiscogsEmbedder configured without a genre graph.")
        embed_model = self._load_embed()
        genre_model = self._load_genre()
        x = np.asarray(samples, dtype=np.float32).ravel()
        emb = np.asarray(embed_model(x), dtype=np.float32)
        preds = np.asarray(genre_model(emb), dtype=np.float32)
        probs = preds.mean(axis=0) if preds.ndim > 1 else preds
        labels = self._labels or [f"genre_{i}" for i in range(probs.shape[0])]
        return {labels[i]: float(probs[i]) for i in range(min(len(labels), probs.shape[0]))}
