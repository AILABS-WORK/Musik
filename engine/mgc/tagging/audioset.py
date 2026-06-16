"""AudioSet-527 tagging backend (AST — Audio Spectrogram Transformer).

Tags each track against the 527-class AudioSet ontology — which includes a
literal *Cowbell* class plus instruments, vocals, percussion, etc. — giving the
precise path for attribute search and the instrument/vocal chips.

Per track we window into ~10 s chunks (the AST input length), tag each, and take
the element-wise MAX over chunks, so "contains X somewhere" is captured (a
cowbell in one section lights up the Cowbell class).

Heavy imports (torch, transformers) are LAZY; downloads honor the OS trust store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_DEFAULT_MODEL = os.environ.get("MGC_AST_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")
_LABELS_PATH = Path(__file__).with_name("audioset_labels.json")
_INSTALL_HINT = (
    "AudioSet tagger needs torch + transformers. Install with: "
    "pip install 'mgc-engine[models]'."
)


def get_audioset_labels() -> list[str] | None:
    """The 527 AudioSet class names, read from the bundled cache (no model load).

    Returns None until a tagger has run once and written the cache.
    """
    if _LABELS_PATH.exists():
        try:
            return json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


class AudioSetTagger:
    """AST AudioSet-527 tagger (GPU-accelerated). ``tag`` -> 527 probabilities."""

    name = "ast"
    sample_rate = 16000

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._model = None
        self._fe = None
        self._torch = None
        self._device = None
        self._labels: list[str] | None = None

    def _load(self):
        if self._model is None:
            from mgc._net import enable_os_truststore

            enable_os_truststore()
            try:
                import torch  # type: ignore
                from transformers import ASTForAudioClassification, AutoFeatureExtractor  # type: ignore
            except Exception as e:  # pragma: no cover - heavy dep missing
                raise RuntimeError(_INSTALL_HINT) from e
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._fe = AutoFeatureExtractor.from_pretrained(self._model_id)
            model = ASTForAudioClassification.from_pretrained(self._model_id).eval().to(self._device)
            self._model = model
            id2label = model.config.id2label
            self._labels = [id2label[i] for i in range(len(id2label))]
            # cache the label list so search/display never needs to load the model
            try:
                if not _LABELS_PATH.exists():
                    _LABELS_PATH.write_text(json.dumps(self._labels), encoding="utf-8")
            except Exception:
                pass
        return self._model, self._fe, self._torch, self._device

    @property
    def labels(self) -> list[str]:
        self._load()
        return self._labels  # type: ignore[return-value]

    def tag(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """One ~10 s chunk of mono audio (at 16 kHz) -> 527 probabilities."""
        model, fe, torch, device = self._load()
        x = np.asarray(samples, dtype=np.float32).ravel()
        inputs = fe(x, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.sigmoid(logits)[0].float().cpu().numpy().astype(np.float32)

    def tag_file(self, path: str, chunk_seconds: float = 10.0, max_chunks: int = 12) -> np.ndarray | None:
        """Whole track -> chunk-MAX 527-d AudioSet vector (None on decode error)."""
        from mgc.audio.decode import load_windows

        windows = load_windows(path, self.sample_rate, chunk_seconds, chunk_seconds, max_chunks)
        if not windows:
            return None
        probs = np.stack([self.tag(w, self.sample_rate) for w in windows])
        return np.max(probs, axis=0).astype(np.float32)


def tag_all(store, tagger: AudioSetTagger | None = None, progress=None) -> int:
    """Tag every track that lacks an AudioSet vector; persist to `understanding`."""
    tagger = tagger or AudioSetTagger()
    tracks = store.iter_tracks()
    n = 0
    for i, t in enumerate(tracks):
        if store.has_audioset(t.id):
            continue
        try:
            v = tagger.tag_file(t.path)
            if v is not None:
                store.save_understanding(t.id, audioset=v, audioset_model=tagger.name)
                n += 1
        except Exception:
            pass
        if progress:
            progress(i + 1, len(tracks))
    return n


def top_tags(vector: np.ndarray, labels: list[str], k: int = 10, threshold: float = 0.08) -> list[dict]:
    """Top AudioSet tags for one track's 527-d vector -> [{label, prob}]."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    order = np.argsort(-v)[: max(k * 3, k)]
    out = []
    for idx in order:
        p = float(v[idx])
        if p < threshold or idx >= len(labels):
            continue
        out.append({"label": labels[idx], "prob": round(p, 3)})
        if len(out) >= k:
            break
    return out
