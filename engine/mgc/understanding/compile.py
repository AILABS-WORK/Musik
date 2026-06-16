"""Understanding compiler — turn an AudioSet-527 probability vector (+ the
BPM/key/energy analysis) into a structured per-song record: instruments present,
vocal/instrumental + perceived gender, a 2-D mood coordinate, a templated
caption, and canonical tags.

Everything here is derived from data we already compute (the AudioSet vector +
analysis) — no extra models. It is honest about confidence: vocal gender and
mood are *perceived/indicative*, not hard labels.
"""

from __future__ import annotations

import numpy as np

# AudioSet-527 class names that denote instruments (curated subset that exists
# in the ontology). Matched against the actual label list at runtime.
_INSTRUMENTS = {
    "Guitar", "Electric guitar", "Bass guitar", "Acoustic guitar",
    "Steel guitar, slide guitar", "Banjo", "Sitar", "Mandolin", "Ukulele",
    "Piano", "Electric piano", "Organ", "Electronic organ", "Hammond organ",
    "Synthesizer", "Sampler", "Harpsichord", "Percussion", "Drum kit",
    "Drum machine", "Drum", "Snare drum", "Bass drum", "Timpani", "Tabla",
    "Cymbal", "Hi-hat", "Tambourine", "Maraca", "Gong", "Tubular bells",
    "Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone",
    "Steelpan", "Cowbell", "Wood block", "Brass instrument", "French horn",
    "Trumpet", "Trombone", "Bowed string instrument", "String section",
    "Violin, fiddle", "Cello", "Double bass", "Flute", "Saxophone", "Clarinet",
    "Harp", "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", "Theremin",
}
_SUNG = {"Singing", "Male singing", "Female singing", "Child singing", "Choir",
         "Vocal music", "A capella", "Rapping", "Chant", "Synthetic singing"}
_MALE = {"Male singing", "Male speech, man speaking"}
_FEMALE = {"Female singing", "Female speech, woman speaking"}


def compile_record(audioset, labels: list[str], analysis: dict | None = None,
                   instr_threshold: float = 0.12, vocal_threshold: float = 0.2) -> dict:
    """AudioSet vector + analysis -> {instruments, vocal, mood, caption, tags_canonical}."""
    v = np.asarray(audioset, dtype=np.float32).ravel()
    idx = {lab: i for i, lab in enumerate(labels)}

    def p(name: str) -> float:
        i = idx.get(name)
        return float(v[i]) if (i is not None and i < len(v)) else 0.0

    # --- instruments present -------------------------------------------------
    instruments = {n: round(p(n), 3) for n in _INSTRUMENTS if p(n) >= instr_threshold}
    instruments = dict(sorted(instruments.items(), key=lambda kv: -kv[1])[:10])
    top_instr = list(instruments.keys())[:3]

    # --- vocal / gender (perceived) -----------------------------------------
    sung = max((p(n) for n in _SUNG), default=0.0)
    male = max((p(n) for n in _MALE), default=0.0)
    female = max((p(n) for n in _FEMALE), default=0.0)
    if sung >= vocal_threshold:
        vi = "vocal"
    elif sung < vocal_threshold * 0.5:
        vi = "instrumental"
    else:
        vi = "uncertain"
    if vi == "instrumental" or max(male, female) < 0.12:
        gender = "unknown"
    else:
        gender = "female" if female > male else "male"
    vocal = {"voice_instrumental": vi, "sung_score": round(sung, 3),
             "gender": gender, "gender_conf": round(abs(female - male), 3)}

    # --- mood (indicative 2-D coordinate) -----------------------------------
    a = analysis or {}
    energy = a.get("energy")
    key = a.get("music_key")
    bpm = a.get("bpm")
    arousal = float(energy) if energy is not None else 0.5
    is_major = bool(key) and "maj" in str(key).lower()
    valence = 0.5 + (0.15 if is_major else -0.10) + (arousal - 0.5) * 0.3
    valence = max(0.0, min(1.0, valence))
    mood = {"arousal": round(arousal, 3), "valence": round(valence, 3)}

    # --- caption (templated, never invents instruments) ---------------------
    ew = "High-energy" if arousal > 0.66 else ("Mellow" if arousal < 0.4 else "Mid-energy")
    vphrase = "instrumental" if vi == "instrumental" else (
        f"{gender} vocal" if gender != "unknown" else "vocal")
    head = f"{ew} {vphrase} track"
    if bpm:
        head += f" at {round(float(bpm))} BPM"
    if key:
        head += f" in {key}"
    caption = head + "."
    if top_instr:
        caption += " Prominent " + ", ".join(s.lower() for s in top_instr) + "."

    # --- canonical tags ------------------------------------------------------
    tags: list[str] = []
    tags.append("instrumental" if vi == "instrumental"
                else (f"{gender} vocal" if gender != "unknown" else "vocal"))
    tags.extend(top_instr)
    if key:
        tags.append(str(key))
    tags.append(ew.lower())
    seen, canonical = set(), []
    for t in tags:
        if t and t.lower() not in seen:
            seen.add(t.lower())
            canonical.append(t)

    return {"instruments": instruments, "vocal": vocal, "mood": mood,
            "caption": caption, "tags_canonical": canonical}
