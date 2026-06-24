"""Explainable similarity: WHY two tracks group together (or don't).

Cosine in the embedding space says HOW similar two tracks are; this decomposes the
comparison across the things we actually measure - the AudioSet sounds/instruments,
vocal, mood, tempo, key and energy - and returns, in plain words, what they share
and what differs. That makes grouping into subgenres transparent instead of a
black-box distance.
"""

from __future__ import annotations

import os

import numpy as np


def _cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if (na and nb) else None


def _name(track) -> str:
    return os.path.splitext(os.path.basename(track.path))[0] if track else "?"


def explain_similarity(store, a_id: int, b_id: int, model: str) -> dict:
    """Compare two tracks. Returns
    ``{score, a, b, shared:[...], different:[...]}`` where score is the embedding
    cosine and shared/different are human-readable reasons."""
    ta, tb = store.get_track(a_id), store.get_track(b_id)
    if not ta or not tb:
        return {"error": "missing track"}
    na, nb = _name(ta), _name(tb)

    score = _cos(store.get_embedding(a_id, model), store.get_embedding(b_id, model))
    shared: list[str] = []
    different: list[str] = []

    # ---- sound profile (AudioSet -> instruments / vocal / mood) ------------
    from mgc.tagging import get_audioset_labels
    from mgc.understanding import compile_record

    labels = get_audioset_labels() or []
    ua, ub = store.get_understanding(a_id), store.get_understanding(b_id)
    aa, ab = store.get_analysis(a_id) or {}, store.get_analysis(b_id) or {}
    reca = compile_record(ua["audioset"], labels, aa) if (ua and ua.get("audioset") is not None and labels) else None
    recb = compile_record(ub["audioset"], labels, ab) if (ub and ub.get("audioset") is not None and labels) else None

    if reca and recb:
        ia, ib = set(reca["instruments"]), set(recb["instruments"])
        both = sorted(ia & ib)
        if both:
            shared.append("instruments: " + ", ".join(s.lower() for s in both))
        if ia - ib:
            different.append(f"{na} has " + ", ".join(s.lower() for s in sorted(ia - ib)))
        if ib - ia:
            different.append(f"{nb} has " + ", ".join(s.lower() for s in sorted(ib - ia)))

        va, vb = reca["vocal"], recb["vocal"]
        if va["voice_instrumental"] == vb["voice_instrumental"]:
            shared.append(va["voice_instrumental"])
        else:
            different.append(f"{na} {va['voice_instrumental']} vs {nb} {vb['voice_instrumental']}")
        if va["gender"] != "unknown" and va["gender"] == vb["gender"]:
            shared.append(f"{va['gender']} vocal")

        ma, mb = set(reca["mood"]["tags"]), set(recb["mood"]["tags"])
        if ma & mb:
            shared.append("mood: " + ", ".join(sorted(ma & mb)))
        elif ma and mb:
            different.append(f"mood {'/'.join(sorted(ma)[:2])} vs {'/'.join(sorted(mb)[:2])}")

    # ---- tempo / key / energy (from analysis) -----------------------------
    bpm_a, bpm_b = aa.get("bpm"), ab.get("bpm")
    if bpm_a and bpm_b:
        if abs(bpm_a - bpm_b) <= 4:
            shared.append(f"tempo ~{round((bpm_a + bpm_b) / 2)} BPM")
        else:
            different.append(f"tempo {round(bpm_a)} vs {round(bpm_b)} BPM")

    ka, kb = aa.get("music_key"), ab.get("music_key")
    if ka and kb:
        from mgc.setbuilder.builder import _camelot_distance, _camelot_label, _to_camelot
        ca, cb = _to_camelot(ka), _to_camelot(kb)
        d = _camelot_distance(ca, cb)
        if d == 0.0:
            shared.append(f"key {_camelot_label(ca) or ka} (harmonically compatible)")
        else:
            different.append(f"key {ka} vs {kb}")

    ea, eb = aa.get("energy"), ab.get("energy")
    if ea is not None and eb is not None:
        if abs(ea - eb) <= 0.12:
            shared.append(f"energy ~{round((ea + eb) / 2 * 100)}%")
        else:
            different.append(f"energy {round(ea * 100)}% vs {round(eb * 100)}%")

    # ---- frequency-range breakdown (where they match: lows / mids / highs) --
    bands: list = []
    try:
        from mgc.similarity.bands import band_breakdown
        pa = store.get_spectral(a_id) if hasattr(store, "get_spectral") else None
        pb = store.get_spectral(b_id) if hasattr(store, "get_spectral") else None
        bb = band_breakdown(pa, pb)
        bands = bb.get("bands", [])
    except Exception:
        bands = []

    return {"score": round(score, 3) if score is not None else None,
            "a": na, "b": nb, "shared": shared, "different": different,
            "bands": bands}
