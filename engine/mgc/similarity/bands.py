"""Per-frequency-range similarity breakdown.

The stored spectral profile is 16 log-spaced bands (20 Hz .. sr/2). For "where do
these two tracks match?" that's too fine to read, so we fold the 16 bands into 6
named ranges (sub / bass / low-mid / mid / high-mid / highs) and score, per range,
how close the two tracks' energy is. The result reads like a DJ would think: "same
low end, brighter highs on track B".
"""

from __future__ import annotations

import numpy as np

_SR = 22050
_N = 16
# Same edges spectral_profile() uses, so band i here lines up with profile entry i.
_EDGES = np.logspace(np.log10(20), np.log10(_SR / 2), _N + 1)
_CENTERS = np.sqrt(_EDGES[:-1] * _EDGES[1:])

# Named ranges (Hz). The last range absorbs everything above 6 kHz.
RANGES = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 6000),
    ("highs", 6000, _SR),
]
# Which named range each of the 16 bands falls into (by its centre frequency).
_BAND_RANGE = [
    next((ri for ri, (_, lo, hi) in enumerate(RANGES) if lo <= c < hi), len(RANGES) - 1)
    for c in _CENTERS
]


def band_breakdown(prof_a, prof_b) -> dict:
    """Compare two 16-band spectral profiles. Returns
    ``{bands:[{name, a, b, match}], overall}`` where each ``match`` is 0..1 (1 =
    the two tracks carry the same share of energy in that range) and ``overall`` is
    the cosine of the full profiles."""
    if prof_a is None or prof_b is None:
        return {"bands": [], "overall": None}
    a = np.asarray(prof_a, dtype=np.float64).ravel()
    b = np.asarray(prof_b, dtype=np.float64).ravel()
    if a.size != _N or b.size != _N:
        return {"bands": [], "overall": None}

    out = []
    for ri, (name, _lo, _hi) in enumerate(RANGES):
        idx = [i for i in range(_N) if _BAND_RANGE[i] == ri]
        ea = float(a[idx].sum())
        eb = float(b[idx].sum())
        denom = ea + eb + 1e-9
        match = 1.0 - abs(ea - eb) / denom          # relative closeness in this range
        out.append({"name": name, "a": round(ea, 3), "b": round(eb, 3),
                    "match": round(max(0.0, min(1.0, match)), 3)})

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    overall = float(a @ b / (na * nb)) if (na and nb) else None
    return {"bands": out, "overall": round(overall, 3) if overall is not None else None}
