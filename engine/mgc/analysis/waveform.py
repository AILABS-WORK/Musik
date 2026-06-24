"""Spectral waveform + per-track frequency fingerprint.

``spectral_waveform`` returns a downsampled bass/mid/high energy timeline so the UI can
draw a Rekordbox-style RGB waveform (low=red, mid=green, high=blue), with brightness =
loudness. ``spectral_profile`` returns an L2-normalised log-band energy vector: a compact
timbral fingerprint for "find tracks with a similar bassline / highs" search.

Pure numpy STFT (no extra deps). Computed on demand; cheap (~1-2s per track).
"""

from __future__ import annotations

import numpy as np

from mgc.audio.decode import load_mono

_SR = 22050
_N_FFT = 2048
_HOP = 1024


def _stft_power(y: np.ndarray) -> np.ndarray:
    """Short-time power spectrum -> [frames, freq_bins]."""
    if y.size < _N_FFT:
        y = np.pad(y, (0, _N_FFT - y.size))
    n = 1 + (len(y) - _N_FFT) // _HOP
    win = np.hanning(_N_FFT).astype(np.float32)
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n)[:, None]
    frames = y[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def _downsample(a: np.ndarray, bins: int) -> np.ndarray:
    if a.size <= bins:
        return a
    edges = np.linspace(0, a.size, bins + 1).astype(int)
    return np.array([a[edges[i]:edges[i + 1]].mean() for i in range(bins)])


def spectral_waveform(path: str, bins: int = 480,
                      start: float | None = None, end: float | None = None) -> dict:
    """Bass/mid/high energy over time, each normalised to [0,1] (sqrt for perceptual).

    With ``start``/``end`` (seconds) only that window is analysed, so the UI can fetch a
    high-resolution slice when zoomed in (e.g. a 4-bar view) instead of downsampling the
    whole track. Returns the same shape plus ``start``/``end`` of the window."""
    y, sr = load_mono(path, _SR)
    if y.size == 0:
        return {"bass": [], "mid": [], "high": [], "start": 0.0, "end": 0.0}
    dur = len(y) / sr
    if start is not None or end is not None:
        s = max(0.0, float(start or 0.0))
        e = min(dur, float(end if end is not None else dur))
        if e - s < 0.05:
            e = min(dur, s + 0.05)
        y = y[int(s * sr):int(e * sr)]
        win_start, win_end = s, e
    else:
        win_start, win_end = 0.0, dur
    if y.size == 0:
        return {"bass": [], "mid": [], "high": [], "start": win_start, "end": win_end}
    spec = _stft_power(y)
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    bass = spec[:, freqs < 250].sum(1)
    mid = spec[:, (freqs >= 250) & (freqs < 4000)].sum(1)
    high = spec[:, freqs >= 4000].sum(1)
    b, m, h = (_downsample(x, bins) for x in (bass, mid, high))
    mx = max(float(b.max()), float(m.max()), float(h.max()), 1e-9)
    out = {"bass": np.sqrt(b / mx), "mid": np.sqrt(m / mx), "high": np.sqrt(h / mx)}
    res = {k: v.round(3).tolist() for k, v in out.items()}
    res["start"], res["end"] = round(win_start, 3), round(win_end, 3)
    return res


def spectral_profile(path: str, n_bands: int = 16) -> list:
    """L2-normalised average energy in ``n_bands`` log-spaced frequency bands.

    A timbral fingerprint: two tracks with cosine-similar profiles share a similar
    balance of bass / mids / highs (bassline weight, brightness, etc.).
    """
    y, sr = load_mono(path, _SR)
    if y.size == 0:
        return []
    avg = _stft_power(y).mean(0)
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    edges = np.logspace(np.log10(20), np.log10(sr / 2), n_bands + 1)
    prof = np.array([float(avg[(freqs >= edges[i]) & (freqs < edges[i + 1])].sum())
                     for i in range(n_bands)])
    prof = np.sqrt(prof)                      # perceptual compression
    prof = prof / (np.linalg.norm(prof) + 1e-9)
    return prof.round(4).tolist()
