"""Deterministic numpy-only spectral embedder.

The baseline backend computes a fixed-length timbral descriptor from a mono
signal using a short-time Fourier transform, a small mel filterbank and a few
classic spectral statistics. It needs no heavy ML deps, is fully deterministic,
and separates timbrally distinct signals (rich low tones vs. pure high tones).
"""

from __future__ import annotations

import numpy as np

from mgc.embed.base import l2_normalize
from mgc.types import Embedder

# STFT / framing parameters (fixed -> deterministic, fixed dims).
_FRAME = 2048
_HOP = 512
_N_MELS = 40
_FMIN = 0.0

# Feature vector layout (all per-frame stats are mean+std across frames):
#   mel log-energies:        N_MELS * 2
#   spectral centroid:       2
#   spectral bandwidth:      2
#   spectral rolloff:        2
#   zero-crossing-rate:      2
#   RMS:                     2
_DIMS = _N_MELS * 2 + 5 * 2


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Triangular mel filterbank: ``[n_mels, n_fft//2 + 1]``."""
    n_bins = n_fft // 2 + 1
    fmax = sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(np.array(_FMIN)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


class BaselineEmbedder(Embedder):
    """A deterministic, dependency-free spectral embedder."""

    name = "baseline"
    sample_rate = 22050
    dims = _DIMS

    def __init__(self) -> None:
        self._window = np.hanning(_FRAME).astype(np.float32)
        self._fb = _mel_filterbank(self.sample_rate, _FRAME, _N_MELS)
        # Frequency of each FFT bin (used for centroid/bandwidth/rolloff).
        self._freqs = np.fft.rfftfreq(_FRAME, d=1.0 / self.sample_rate).astype(np.float32)

    def _frames(self, samples: np.ndarray) -> np.ndarray:
        """Return framed signal ``[n_frames, _FRAME]`` (windowed)."""
        n = samples.shape[0]
        if n < _FRAME:
            samples = np.pad(samples, (0, _FRAME - n))
            n = _FRAME
        n_frames = 1 + (n - _FRAME) // _HOP
        idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n_frames)[:, None]
        framed = samples[idx]
        return framed * self._window[None, :]

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Embed a mono float32 signal -> ``(dims,)`` L2-normalized vector."""
        x = np.asarray(samples, dtype=np.float32).ravel()
        if x.size == 0:
            return np.zeros(self.dims, dtype=np.float32)

        frames = self._frames(x)  # [n_frames, _FRAME]
        # Magnitude spectrum per frame.
        spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)  # [n_frames, n_bins]

        # Log-mel band energies (mean+std across frames).
        mel_energy = spec @ self._fb.T  # [n_frames, n_mels]
        log_mel = np.log1p(mel_energy)
        mel_mean = log_mel.mean(axis=0)
        mel_std = log_mel.std(axis=0)

        # Spectral centroid / bandwidth / rolloff per frame.
        power = spec + 1e-10
        total = power.sum(axis=1)  # [n_frames]
        centroid = (power * self._freqs[None, :]).sum(axis=1) / total
        bandwidth = np.sqrt(
            (power * (self._freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total
        )
        cumspec = np.cumsum(power, axis=1)
        thresh = 0.85 * total[:, None]
        rolloff_idx = (cumspec >= thresh).argmax(axis=1)
        rolloff = self._freqs[rolloff_idx]

        # Zero-crossing rate and RMS per frame (computed on raw, unwindowed frames
        # would differ; windowed frames are fine and deterministic).
        n = x.shape[0]
        if n < _FRAME:
            x = np.pad(x, (0, _FRAME - n))
            n = _FRAME
        n_frames = 1 + (n - _FRAME) // _HOP
        idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n_frames)[:, None]
        raw_frames = x[idx]
        signs = np.sign(raw_frames)
        zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
        rms = np.sqrt(np.mean(raw_frames ** 2, axis=1))

        c_m, c_s = _mean_std(centroid)
        b_m, b_s = _mean_std(bandwidth)
        r_m, r_s = _mean_std(rolloff)
        z_m, z_s = _mean_std(zcr)
        e_m, e_s = _mean_std(rms)

        feat = np.concatenate([
            mel_mean, mel_std,
            np.array([c_m, c_s, b_m, b_s, r_m, r_s, z_m, z_s, e_m, e_s], dtype=np.float32),
        ]).astype(np.float32)

        return l2_normalize(feat)
