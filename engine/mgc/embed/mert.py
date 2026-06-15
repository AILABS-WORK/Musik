"""MERT embedding backend (transformers / m-a-p/MERT-v1-330M).

Mean-pools the model's hidden states over time to produce a fixed-size music
representation. Heavy imports (torch, transformers, torchaudio) are LAZY so that
``import mgc.embed.mert`` stays light.
"""

from __future__ import annotations

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder

_MODEL_ID = "m-a-p/MERT-v1-330M"
_INSTALL_HINT = (
    "MERT backend requires torch + transformers. "
    "Install with: pip install 'mgc-engine[models]'."
)


def _check_deps() -> None:
    """Raise RuntimeError with an install hint if torch/transformers are missing."""
    try:
        import torch  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e


class MertEmbedder(Embedder):
    """m-a-p/MERT-v1-330M mean-pooled hidden-state embeddings."""

    name = "mert"
    sample_rate = 24000
    dims = 1024

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self):
        if self._model is None:
            try:
                import torch  # type: ignore
                from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._torch = torch
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True
            ).eval()
        return self._model, self._processor, self._torch

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed mono float32 audio -> L2-normalized 1024-d vector.

        Hidden states from all transformer layers are averaged over both layers
        and time, yielding a single fixed-size descriptor.
        """
        model, processor, torch = self._load()
        x = np.asarray(samples, dtype=np.float32).ravel()
        inputs = processor(x, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # [n_layers, time, hidden] -> mean over layers and time.
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        pooled = hidden.mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
        return l2_normalize(pooled)
