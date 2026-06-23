"""LLM-assisted genre guess from a track's filename (which usually names the LABEL),
its sound tags and BPM.

The label is a strong genre signal the heuristics ignore (BCCO/Drumcode -> techno,
SUARA -> tech house, ...), and the LLM knows many labels. But it hallucinates labels
it doesn't know and reasons poorly about tempo, so every guess is validated against a
plausible-BPM table and a confidence floor before we keep it. MuQ still does the
similarity/propagation; this only proposes names.
"""

from __future__ import annotations

import json
import os

from mgc.llm import ollama

VOCAB = (
    "Deep House, Tech House, Minimal House, Garage House, Acid House, Progressive House, "
    "Melodic House, Afro House, Disco House, Techno, Minimal Techno, Dub Techno, "
    "Peak Time Techno, Hard Techno, Hypnotic Techno, Melodic Techno, Acid Techno, "
    "Trance, Hard Trance, Goa Trance, Progressive Trance, Drum and Bass, Dubstep, "
    "Breakbeat, UK Garage, Speed Garage, Disco, Italo-Disco, Nu Disco, Ambient, "
    "Downtempo, Electro, Electronica"
)

# Plausible BPM (octave-tolerant) per genre keyword — a guess outside its range is
# almost certainly wrong (e.g. "Nu Disco @ 166").
GENRE_BPM = {
    "italo": (112, 132), "nu disco": (110, 126), "disco": (108, 130),
    "deep house": (115, 126), "tech house": (120, 130), "acid house": (118, 130),
    "melodic house": (118, 126), "afro house": (118, 128), "minimal house": (120, 128),
    "garage house": (120, 130), "house": (118, 130), "minimal techno": (125, 134),
    "dub techno": (120, 134), "hard techno": (140, 165), "peak time": (138, 150),
    "hypnotic techno": (130, 142), "melodic techno": (120, 128), "acid techno": (130, 150),
    "techno": (125, 145), "hard trance": (140, 150), "goa": (138, 150),
    "progressive trance": (132, 142), "trance": (130, 145), "drum and bass": (160, 180),
    "dubstep": (135, 150), "speed garage": (130, 140), "uk garage": (128, 138),
    "garage": (126, 138), "breakbeat": (125, 140), "electro": (125, 135),
    "trip hop": (80, 110), "downtempo": (60, 112), "ambient": (0, 110),
}

_SYSTEM = (
    "You are a techno/house DJ and crate-digger who knows record labels. Given a track "
    "filename (it usually contains the LABEL and artist), detected sound tags and BPM, "
    "pick the single most fitting subgenre from VOCAB (or a very close real one). The "
    "label is a strong hint (e.g. BCCO/Drumcode=techno, SUARA=tech house). ONLY answer "
    "if you actually recognise the label/artist or the tags+BPM make it clear; otherwise "
    "give low confidence. Output ONLY compact JSON: {\"genre\":\"\",\"confidence\":0-1}."
)


def bpm_plausible(genre: str, bpm) -> bool:
    if bpm is None:
        return True
    g = genre.lower()
    for kw, (lo, hi) in GENRE_BPM.items():
        if kw in g:
            return any(lo - 4 <= b <= hi + 4 for b in (bpm, bpm / 2, bpm * 2))
    return True


_GROUNDED_SYSTEM = (
    "You are a music genre expert. Using ONLY the provided web search snippets, "
    "identify the electronic music subgenre of this track. If the snippets don't "
    "say, output low confidence. Do NOT invent. Output ONLY JSON "
    "{\"genre\":\"\",\"confidence\":0-1}."
)


def llm_genre_grounded(artist: str | None, title: str | None, label: str | None,
                       tags: list, bpm, model: str | None = None,
                       search=None) -> dict | None:
    """Genre guess grounded on real web snippets instead of the LLM's memory.

    Searches the web for the track, feeds the snippets to the LLM, and asks it to
    name the subgenre using ONLY those snippets. Returns ``{genre, confidence,
    plausible, grounded}`` or ``None`` (no snippets / LLM unavailable / parse
    failure) so the caller can fall back. ``search`` is injectable for tests."""
    if search is None:
        from mgc.llm.websearch import search_snippets as search
    if not ollama.available():
        return None
    name = " ".join(p for p in (artist, title) if p).strip()
    query = f"{name} genre".strip() if name else "music genre"
    if label:
        query = f"{query} {label}"
    snippets = search(query)
    if not snippets:
        return None
    user = ("web search snippets:\n- " + "\n- ".join(snippets) +
            f"\n\ntags: {', '.join(tags)}\nbpm: {int(bpm) if bpm else '?'}")
    try:
        out = ollama.chat(
            [{"role": "system", "content": _GROUNDED_SYSTEM},
             {"role": "user", "content": user}],
            model=model, temperature=0.1, timeout=60)
        d = json.loads(out)
    except Exception:
        return None
    genre = str(d.get("genre", "")).strip()
    conf = float(d.get("confidence", 0) or 0)
    if not genre:
        return None
    return {"genre": genre, "confidence": round(conf, 2),
            "plausible": bpm_plausible(genre, bpm), "grounded": True}


def llm_genre(path: str, tags: list, bpm, model: str | None = None,
              min_conf: float = 0.6, validate: bool = True) -> dict | None:
    """LLM genre guess. With ``validate`` (default) returns ``{genre, confidence}`` only
    when confident AND BPM-plausible (for auto use); with ``validate=False`` returns the
    raw guess regardless (for a user-confirmed suggestion)."""
    if not ollama.available():
        return None
    user = (f"VOCAB: {VOCAB}\nfilename: {os.path.basename(path)}\n"
            f"tags: {', '.join(tags)}\nbpm: {int(bpm) if bpm else '?'}")
    try:
        out = ollama.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            model=model, temperature=0.2, timeout=60)
        d = json.loads(out)
    except Exception:
        return None
    genre = str(d.get("genre", "")).strip()
    conf = float(d.get("confidence", 0) or 0)
    if not genre:
        return None
    if validate and (conf < min_conf or not bpm_plausible(genre, bpm)):
        return None
    return {"genre": genre, "confidence": round(conf, 2),
            "plausible": bpm_plausible(genre, bpm)}
