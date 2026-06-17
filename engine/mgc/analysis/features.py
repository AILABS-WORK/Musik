"""Audio analysis features: BPM, musical key, energy, danceability.

numpy/scipy ONLY for the core path (always available). ``librosa`` is optional
and imported lazily purely as a refinement; everything here works without it.

The estimators are intentionally classic / lightweight:

* **energy** — RMS of the signal mapped into ``0..1`` with a perceptual curve.
* **bpm** — tempo from an onset-envelope autocorrelation. We build a spectral-flux
  onset envelope, autocorrelate it, restrict the lag range to a musical tempo
  band (40..240 BPM) and pick the dominant periodicity.
* **music_key** — Krumhansl-Schmuckler key finding over a 12-bin chroma derived
  from an rFFT magnitude spectrum mapped to pitch classes. Returns e.g.
  ``"A min"`` / ``"C maj"``, or ``None`` when no key is salient.
* **danceability** — heuristic in ``[0,1]`` from beat strength (how peaked the
  onset autocorrelation is) and beat regularity.
"""

from __future__ import annotations

import numpy as np

# Framing for the onset envelope / spectral analysis.
_FRAME = 2048
_HOP = 512

# Tempo search band (BPM).
_BPM_MIN = 40.0
_BPM_MAX = 240.0
_BPM_FOLD_MIN = 90.0  # fold detected tempo into [90, 180) for DJ-typical BPMs

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major / minor), starting at the tonic.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Return a ``[n_frames, frame]`` view of ``x`` (zero-padded if too short)."""
    n = x.shape[0]
    if n < frame:
        x = np.pad(x, (0, frame - n))
        n = frame
    n_frames = 1 + (n - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def _magnitude_spectrogram(x: np.ndarray) -> np.ndarray:
    """Windowed rFFT magnitude spectrogram ``[n_frames, n_bins]``."""
    frames = _frame_signal(x, _FRAME, _HOP)
    window = np.hanning(_FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    return spec.astype(np.float64)


def _energy(x: np.ndarray) -> float:
    """RMS mapped to ``0..1`` with a gentle perceptual (sqrt-ish) curve."""
    if x.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    # Map RMS (typically ~0..0.5 for normalized audio) into 0..1. The tanh keeps
    # it bounded and monotonic; the scale makes typical music land mid-range.
    val = float(np.tanh(3.0 * rms))
    return float(np.clip(val, 0.0, 1.0))


def _onset_envelope(spec: np.ndarray) -> np.ndarray:
    """Spectral-flux onset strength envelope, one value per frame."""
    if spec.shape[0] < 2:
        return np.zeros(spec.shape[0], dtype=np.float64)
    # Log-compress to emphasize perceptual onsets, then positive first difference.
    log_spec = np.log1p(spec)
    flux = np.diff(log_spec, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    env = np.concatenate([[0.0], flux])
    # Remove DC so autocorrelation reflects periodicity, not overall level.
    env = env - env.mean()
    return env


def _autocorr(env: np.ndarray) -> np.ndarray:
    """Unbiased-ish autocorrelation of the (zero-mean) onset envelope."""
    n = env.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    full = np.correlate(env, env, mode="full")
    ac = full[n - 1:]  # non-negative lags
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


def _tempo_from_env(env: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Estimate (bpm, beat_strength, beat_regularity) from an onset envelope.

    ``beat_strength`` in ``[0,1]`` is the height of the dominant autocorrelation
    peak. ``beat_regularity`` in ``[0,1]`` rewards a consistent secondary peak at
    twice the period (a steady, repeating pulse).
    """
    ac = _autocorr(env)
    n = ac.shape[0]
    if n < 4:
        return 0.0, 0.0, 0.0

    frames_per_sec = sr / float(_HOP)
    # Convert the BPM band into a lag (in frames) band.
    min_lag = max(1, int(np.floor(frames_per_sec * 60.0 / _BPM_MAX)))
    max_lag = int(np.ceil(frames_per_sec * 60.0 / _BPM_MIN))
    max_lag = min(max_lag, n - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0, 0.0

    band = ac[min_lag : max_lag + 1]
    if band.size == 0 or not np.any(band > 0):
        return 0.0, 0.0, 0.0

    peak_rel = int(np.argmax(band))
    lag = min_lag + peak_rel
    if lag <= 0:
        return 0.0, 0.0, 0.0

    bpm = float(60.0 * frames_per_sec / lag)
    # Octave-fold into a DJ-typical range: autocorrelation very often locks onto
    # the half-tempo (e.g. 140 BPM techno read as 70), which is useless for mixing.
    # Fold into [90, 180): 70->140, 66->132, 200->100. Right for electronic/dance
    # libraries; a genuine sub-90 track is reported at its double.
    if bpm > 0:
        while bpm < _BPM_FOLD_MIN:
            bpm *= 2.0
        while bpm >= 2.0 * _BPM_FOLD_MIN:
            bpm /= 2.0
    beat_strength = float(np.clip(band[peak_rel], 0.0, 1.0))

    # Regularity: presence of a harmonic peak at twice the lag (steady pulse).
    regularity = 0.0
    lag2 = 2 * lag
    if lag2 < n:
        regularity = float(np.clip(ac[lag2], 0.0, 1.0))
    return bpm, beat_strength, regularity


def _chroma(spec: np.ndarray, sr: int) -> np.ndarray:
    """Fold the rFFT magnitude spectrogram into a 12-bin chroma vector."""
    n_bins = spec.shape[1]
    freqs = np.fft.rfftfreq(_FRAME, d=1.0 / sr)
    chroma = np.zeros(12, dtype=np.float64)
    # Aggregate magnitude across frames, then map each frequency bin to a pitch class.
    mag = spec.mean(axis=0)
    valid = freqs > 0
    if not np.any(valid):
        return chroma
    f = freqs[valid]
    m = mag[valid]
    # MIDI-style pitch class from frequency (A4 = 440 Hz).
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    for k in range(12):
        chroma[k] = m[pc == k].sum()
    return chroma


def _estimate_key(chroma: np.ndarray) -> tuple[str | None, float]:
    """Krumhansl-Schmuckler key estimate -> (key_str|None, confidence)."""
    total = float(chroma.sum())
    if total <= 0:
        return None, 0.0
    c = chroma / total
    cz = c - c.mean()
    cz_norm = float(np.linalg.norm(cz))
    if cz_norm < 1e-9:
        return None, 0.0

    maj_z = _MAJOR_PROFILE - _MAJOR_PROFILE.mean()
    min_z = _MINOR_PROFILE - _MINOR_PROFILE.mean()
    maj_norm = float(np.linalg.norm(maj_z))
    min_norm = float(np.linalg.norm(min_z))

    best_score = -2.0
    best_key: str | None = None
    second = -2.0
    for tonic in range(12):
        rotation = np.roll(cz, -tonic)
        maj_corr = float(np.dot(rotation, maj_z) / (cz_norm * maj_norm + 1e-12))
        min_corr = float(np.dot(rotation, min_z) / (cz_norm * min_norm + 1e-12))
        for score, suffix in ((maj_corr, "maj"), (min_corr, "min")):
            if score > best_score:
                second = best_score
                best_score = score
                best_key = f"{_PITCH_NAMES[tonic]} {suffix}"
            elif score > second:
                second = score

    if best_key is None or best_score <= 0.0:
        return None, 0.0
    # Confidence: positive correlation, sharpened by the margin over the runner-up.
    margin = max(0.0, best_score - max(second, 0.0))
    confidence = float(np.clip(best_score * (0.5 + 0.5 * np.tanh(5.0 * margin)), 0.0, 1.0))
    return best_key, confidence


def _danceability(beat_strength: float, beat_regularity: float, energy: float) -> float:
    """Heuristic danceability in ``[0,1]`` from beat cues + energy."""
    # A danceable track has a strong, regular pulse and enough energy.
    score = 0.5 * beat_strength + 0.35 * beat_regularity + 0.15 * energy
    return float(np.clip(score, 0.0, 1.0))


def analyze_samples(samples: np.ndarray, sr: int) -> dict:
    """Analyze a mono float32 signal.

    Returns ``{bpm, music_key, energy, danceability}`` where ``bpm`` is a float
    (0.0 when undetermined), ``music_key`` is a string like ``"A min"`` or
    ``None``, and ``energy``/``danceability`` are floats in ``[0,1]``.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0 or sr <= 0:
        return {"bpm": 0.0, "music_key": None, "energy": 0.0, "danceability": 0.0}

    energy = _energy(x)

    spec = _magnitude_spectrogram(x)
    env = _onset_envelope(spec)
    bpm, beat_strength, beat_regularity = _tempo_from_env(env, sr)

    chroma = _chroma(spec, sr)
    key, key_conf = _estimate_key(chroma)
    if key_conf < 0.0:  # defensive; _estimate_key clamps already
        key = None

    dance = _danceability(beat_strength, beat_regularity, energy)

    return {
        "bpm": float(round(bpm, 2)),
        "music_key": key,
        "energy": float(round(energy, 4)),
        "danceability": float(round(dance, 4)),
    }
