"""MuQ embedding backend (tencent-ailab/MuQ via the `muq` pip package).

MuQ is a 2025 MARBLE-SOTA self-supervised *music* encoder — a drop-in
alternative to MERT for A/B testing. Like the MERT backend we average the
model's per-layer hidden states over layers and time to get one fixed-size
descriptor per audio window, so the two backends are interchangeable. Heavy
imports (torch, muq) are LAZY so ``import mgc.embed.muq`` stays light. Uses the
GPU automatically when available (your RTX 5080); falls back to CPU.

Model is configurable via ``$MGC_MUQ_MODEL`` (default OpenMuQ/MuQ-large-msd-iter).
"""

from __future__ import annotations

import os

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder

_DEFAULT_MODEL = os.environ.get("MGC_MUQ_MODEL", "OpenMuQ/MuQ-large-msd-iter")
_INSTALL_HINT = "MuQ backend needs torch + muq: pip install muq"


def _check_deps() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
        from muq import MuQ  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e


class MuQEmbedder(Embedder):
    """OpenMuQ/MuQ mean-pooled hidden-state embeddings (GPU-accelerated)."""

    name = "muq"
    sample_rate = 24000
    dims = 1024  # corrected to the model's hidden_size on load

    def __init__(self, model_id: str = _DEFAULT_MODEL, fp16: bool = True) -> None:
        self._model_id = model_id
        self._fp16 = fp16
        self._model = None
        self._torch = None
        self._device = None

    def _load(self):
        if self._model is None:
            from mgc._net import enable_os_truststore

            enable_os_truststore()  # help HF downloads behind TLS-inspecting proxies
            try:
                import torch  # type: ignore
                from muq import MuQ  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MuQ.from_pretrained(self._model_id)
            model = model.eval().to(self._device)
            self._model = model
            hidden = getattr(getattr(model, "config", None), "hidden_size", None)
            if hidden:
                self.dims = int(hidden)
        return self._model, self._torch, self._device

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed mono float32 audio (at 24 kHz) -> L2-normalized descriptor.

        Hidden states from all transformer layers are averaged over layers and
        time into one fixed-size vector (matching the MERT backend's output
        conventions so the two are interchangeable).
        """
        model, torch, device = self._load()
        x = np.asarray(samples, dtype=np.float32).ravel()
        wavs = torch.tensor(x).unsqueeze(0).to(device)  # [1, samples] @ 24 kHz
        with torch.no_grad():
            out = model(wavs, output_hidden_states=True)
        # stack -> [n_layers, batch=1, time, hidden]; drop batch; mean layers+time
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        pooled = hidden.mean(dim=(0, 1)).float().cpu().numpy().astype(np.float32)
        return l2_normalize(pooled)
