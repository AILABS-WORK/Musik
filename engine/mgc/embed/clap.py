"""CLAP embedding backend (laion-clap).

Produces joint audio/text embeddings so audio can be compared directly to text
prompts (zero-shot genre naming). Heavy imports (laion_clap, torch) are LAZY so
that ``import mgc.embed.clap`` stays light.
"""

from __future__ import annotations

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder

_INSTALL_HINT = (
    "CLAP backend requires laion-clap + torch. "
    "Install with: pip install 'mgc-engine[models]'."
)


def _check_deps() -> None:
    """Raise RuntimeError with an install hint if laion-clap is unavailable."""
    try:
        import laion_clap  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e


class ClapEmbedder(Embedder):
    """LAION-CLAP audio (and text) embeddings — a shared 512-d space."""

    name = "clap"
    sample_rate = 48000
    dims = 512

    def __init__(self, ckpt: str | None = None, enable_fusion: bool = False) -> None:
        self._ckpt = ckpt
        self._enable_fusion = enable_fusion
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import laion_clap  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            model = laion_clap.CLAP_Module(enable_fusion=self._enable_fusion)
            model.load_ckpt(self._ckpt) if self._ckpt else model.load_ckpt()
            self._model = model
        return self._model

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed mono float32 audio -> L2-normalized 512-d vector."""
        model = self._load()
        x = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        emb = model.get_audio_embedding_from_data(x=x, use_tensor=False)
        return l2_normalize(np.asarray(emb, dtype=np.float32).ravel())

    def text_embed(self, text: str) -> np.ndarray:
        """Embed a text prompt into the same 512-d space -> L2-normalized vector."""
        model = self._load()
        emb = model.get_text_embedding([text, ""], use_tensor=False)
        return l2_normalize(np.asarray(emb, dtype=np.float32)[0].ravel())
