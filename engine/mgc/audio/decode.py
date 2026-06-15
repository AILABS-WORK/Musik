"""Audio decoding: load any soundfile-supported file as mono float32 and
slice whole tracks into fixed-length windows.

Light deps only (numpy, soundfile, scipy). librosa is optional and imported
lazily; it is never required for the public API here.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioDecodeError(Exception):
    """Raised when a file cannot be decoded into audio samples."""


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    """Downmix to a 1-D mono float32 array."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        # soundfile returns (frames, channels)
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        raise AudioDecodeError(f"unexpected audio shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resample(samples: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono samples from src_sr to target_sr (polyphase, float32)."""
    if src_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    g = np.gcd(int(src_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(src_sr) // g
    out = resample_poly(samples.astype(np.float64), up, down)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_mono(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode ``path`` to a 1-D mono float32 array resampled to ``target_sr``.

    Returns ``(samples, target_sr)``. Raises :class:`AudioDecodeError` on any
    decode failure.
    """
    if target_sr <= 0:
        raise AudioDecodeError(f"invalid target_sr {target_sr}")
    try:
        data, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - normalize all decode failures
        raise AudioDecodeError(f"failed to decode {path!r}: {exc}") from exc
    mono = _to_mono_float32(data)
    out = _resample(mono, int(src_sr), int(target_sr))
    return out, int(target_sr)


def load_windows(
    path: str,
    target_sr: int,
    window_seconds: float = 5.0,
    hop_seconds: float = 5.0,
    max_windows: int = 24,
) -> list[np.ndarray]:
    """Slice a whole track into mono float32 windows.

    Windows have length ``window_seconds * target_sr`` samples. Hopping by
    ``hop_seconds`` (non-overlapping by default). If the track is shorter than
    one window it returns a single zero-padded window. The result is capped to
    ``max_windows``, spread evenly across the track. Never returns just the
    leading few seconds.
    """
    samples, sr = load_mono(path, target_sr)
    win = max(1, int(round(window_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    n = samples.shape[0]

    starts: list[int] = []
    if n <= win:
        starts = [0]
    else:
        last_start = n - win
        s = 0
        while s <= last_start:
            starts.append(s)
            s += hop
        # ensure the tail is covered if hopping skipped past it
        if starts[-1] != last_start:
            starts.append(last_start)

    if max_windows > 0 and len(starts) > max_windows:
        # evenly spread the selected starts across the available windows
        idx = np.linspace(0, len(starts) - 1, num=max_windows)
        idx = np.unique(np.round(idx).astype(int))
        starts = [starts[i] for i in idx]

    windows: list[np.ndarray] = []
    for s in starts:
        chunk = samples[s : s + win]
        if chunk.shape[0] < win:
            chunk = np.pad(chunk, (0, win - chunk.shape[0]))
        windows.append(np.ascontiguousarray(chunk, dtype=np.float32))
    return windows
