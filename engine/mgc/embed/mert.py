"""MERT embedding backend (transformers / m-a-p/MERT-v1-*).

MERT is a self-supervised *music* representation model — the strongest
general-purpose option for separating fine-grained (e.g. electronic) subgenres.
We average the model's per-layer hidden states over layers and time to get one
fixed-size descriptor per audio window. Heavy imports (torch, transformers) are
LAZY so ``import mgc.embed.mert`` stays light. Uses the GPU automatically when
available (your RTX 5080); falls back to CPU.

Model is configurable via ``$MGC_MERT_MODEL`` (default m-a-p/MERT-v1-330M, the
best; use m-a-p/MERT-v1-95M for a lighter/faster option).
"""

from __future__ import annotations

import os

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder

_DEFAULT_MODEL = os.environ.get("MGC_MERT_MODEL", "m-a-p/MERT-v1-330M")
_INSTALL_HINT = (
    "MERT backend requires torch + transformers (+ nnAudio & einops for the 330M "
    "model's CQT feature). Install with: pip install 'mgc-engine[models]'."
)


def _check_deps() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e


class MertEmbedder(Embedder):
    """m-a-p/MERT mean-pooled hidden-state embeddings (GPU-accelerated)."""

    name = "mert"
    sample_rate = 24000
    dims = 1024  # corrected to the model's hidden_size on load (768 for 95M)

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def _load(self):
        if self._model is None:
            from mgc._net import enable_os_truststore

            enable_os_truststore()  # help HF downloads behind TLS-inspecting proxies
            try:
                import torch  # type: ignore
                from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(self._model_id, trust_remote_code=True)
            model = model.eval().to(self._device)
            self._model = model
            hidden = getattr(model.config, "hidden_size", None)
            if hidden:
                self.dims = int(hidden)
        return self._model, self._processor, self._torch, self._device

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed mono float32 audio (at 24 kHz) -> L2-normalized descriptor.

        Hidden states from all transformer layers are averaged over layers and
        time into one fixed-size vector.
        """
        model, processor, torch, device = self._load()
        x = np.asarray(samples, dtype=np.float32).ravel()
        inputs = processor(x, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # stack -> [n_layers, batch=1, time, hidden]; drop batch; mean layers+time
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        pooled = hidden.mean(dim=(0, 1)).float().cpu().numpy().astype(np.float32)
        return l2_normalize(pooled)
