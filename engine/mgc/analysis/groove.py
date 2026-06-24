"""Per-band temporal ("groove") features for genre discrimination.

Research on EDM genre classification is consistent: genres differ less by their average
spectrum than by the RHYTHM within each frequency band over time — isochronous vs
syncopated bass, static vs fluctuating mids/highs, and dynamic swings. The plain
16-band spectral profile we already store is a *static* timbre snapshot, so two tracks
with a similar tone but different grooves look identical to it.

``groove_features`` adds, per frequency range: how much energy lives there (timbre
balance), how much it FLUCTUATES frame-to-frame (groove/rhythmic activity), and its
dynamic spread. That separates e.g. a rolling techno bass from a static house pad even
when their average tone matches. Pure-numpy STFT, no extra deps.
"""

from __future__ import annotations

import numpy as np

from mgc.analysis.waveform import _N_FFT, _SR, _stft_power
from mgc.audio.decode import load_mono

# 6 ranges (Hz) — same names as the similarity breakdown.
_RANGES = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 6000),
    ("highs", 6000, _SR / 2),
]

# 3 stats per range -> 18 features.
DIM = len(_RANGES) * 3


def groove_features(path: str) -> list:
    """Return 18 per-band temporal features (mean-share, flux, dynamics x 6 ranges),
    or [] if the track is too short / unreadable."""
    y, sr = load_mono(path, _SR)
    if y.size < _N_FFT:
        return []
    spec = _stft_power(y)  # [frames, freq_bins]
    if spec.shape[0] < 2:
        return []
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)

    band_ts = np.array([spec[:, (freqs >= lo) & (freqs < hi)].sum(1) for _, lo, hi in _RANGES])
    total = band_ts.sum(0) + 1e-9
    share = band_ts / total  # each band's share of the spectrum over time

    feats: list = []
    for i in range(len(_RANGES)):
        e = band_ts[i]
        en = e / (e.max() + 1e-9)                 # per-band, self-normalised loudness curve
        mean_share = float(share[i].mean())        # timbre: how much energy lives here
        flux = float(np.abs(np.diff(en)).mean())   # groove: frame-to-frame fluctuation
        dyn = float(en.std())                      # dynamics: how much it swells/drops
        feats += [round(mean_share, 4), round(flux, 4), round(dyn, 4)]
    return feats
