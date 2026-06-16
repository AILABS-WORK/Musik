"""AI set builder: turn a free-text vibe description into an ordered DJ set.

The set follows a coarse energy/BPM arc. We parse the description into a target
energy curve (plus optional genre, BPM and length hints), then greedily pick the
unused candidate track whose analysed energy is closest to each position's
target, tie-breaking on BPM proximity to the previous track for smooth mixing.

Only light deps (numpy/stdlib). Cross-module access is via the Store API, with
analysis (energy/bpm) used for ordering and embeddings only used to scope the
candidate pool. No model downloads.
"""

from __future__ import annotations

import re
from typing import Optional

from mgc.understanding.moods import MOOD_ANCHORS

# Keyword buckets -> a representative energy level in 0..1.
_LOW_WORDS = (
    "slow", "chill", "chilled", "deep", "minimal", "downtempo", "mellow",
    "ambient", "calm", "soft", "relaxed", "laid back", "laid-back", "warm up",
    "warm-up", "warmup", "intro",
)
_MID_WORDS = (
    "groovy", "groove", "rolling", "steady", "mid", "midtempo", "mid-tempo",
    "bouncy", "hypnotic", "driving",
)
_HIGH_WORDS = (
    "punchy", "peak", "peak time", "peak-time", "energetic", "energy", "build",
    "building", "speed up", "speed-up", "harder", "hard", "banging", "banger",
    "intense", "powerful", "uplifting", "climax", "fast",
)

# Phrasing that implies a falling tail at the end of the set.
_FALL_WORDS = (
    "slow down", "slow-down", "wind down", "wind-down", "cool down",
    "cool-down", "end deep", "ending deep", "come down", "comedown",
    "bring it down", "outro", "fade out", "fade-out", "then slow",
)
# Phrasing that implies a rising opening / build.
_RISE_WORDS = (
    "build", "speed up", "speed-up", "harder", "ramp up", "ramp-up",
    "rise", "lift", "go harder", "build up", "build-up", "then build",
)

_LOW_E = 0.2
_MID_E = 0.5
_HIGH_E = 0.85


def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _default_arc(points: int = 6) -> list[float]:
    """A gentle rise then fall (single hump), in 0..1."""
    return _hump_arc(points, low=0.3, high=0.8)


def _hump_arc(points: int, low: float, high: float) -> list[float]:
    """A symmetric rise-then-fall curve peaking in the middle."""
    if points <= 1:
        return [round((low + high) / 2.0, 4)]
    arc = []
    for i in range(points):
        # Triangular envelope: 0 at the ends, 1 in the middle.
        frac = i / (points - 1)
        env = 1.0 - abs(2.0 * frac - 1.0)  # 0 -> 1 -> 0
        arc.append(round(low + (high - low) * env, 4))
    return arc


def _monotone_arc(points: int, lo: float, hi: float) -> list[float]:
    if points <= 1:
        return [round((lo + hi) / 2.0, 4)]
    return [round(lo + (hi - lo) * (i / (points - 1)), 4) for i in range(points)]


def parse_description(text: str) -> dict:
    """Parse a free-text vibe into a structured plan.

    Returns dict with:
      - genres: list[str] candidate genre keywords detected
      - energy_arc: list[float] coarse 0..1 target curve (~5-7 points)
      - bpm_hint: tuple(lo, hi) | None
      - length: int | None desired track count
      - notes: str human-readable summary
    """
    raw = text or ""
    low = raw.lower()

    # --- explicit BPM detection -------------------------------------------
    bpm_hint = _parse_bpm(low)

    # --- explicit length detection ----------------------------------------
    length = _parse_length(low)

    # --- energy phrasing --------------------------------------------------
    has_low = _contains_any(low, _LOW_WORDS)
    has_mid = _contains_any(low, _MID_WORDS)
    has_high = _contains_any(low, _HIGH_WORDS)
    wants_rise = _contains_any(low, _RISE_WORDS) or "start slow" in low
    wants_fall = _contains_any(low, _FALL_WORDS)

    points = 6
    arc, shape = _build_arc(low, points, has_low, has_mid, has_high,
                            wants_rise, wants_fall)

    genres = _parse_genres(raw, low)
    attributes = parse_attributes(low)

    notes = _summarize(shape, genres, bpm_hint, length, attributes)

    return {
        "genres": genres,
        "energy_arc": arc,
        "bpm_hint": bpm_hint,
        "length": length,
        "attributes": attributes,
        "notes": notes,
    }


def _build_arc(low, points, has_low, has_mid, has_high, wants_rise, wants_fall):
    """Pick a coarse energy curve + a shape label from detected phrasing."""
    starts_slow = "start slow" in low or low.strip().startswith(("slow", "chill", "deep"))

    # Rise-then-fall: an explicit arc ("start slow ... build ... slow down").
    if (wants_rise and wants_fall) or (starts_slow and (has_high or wants_rise)):
        return _hump_arc(points, low=_LOW_E, high=_HIGH_E), "rise-then-fall"

    # Pure build / ramp up.
    if wants_rise or (has_high and not has_low):
        lo = _LOW_E if has_low else _MID_E
        return _monotone_arc(points, lo, _HIGH_E), "build"

    # Wind down.
    if wants_fall or (has_low and not has_high and not has_mid and "down" in low):
        hi = _HIGH_E if has_high else _MID_E
        return _monotone_arc(points, hi, _LOW_E), "wind-down"

    # Flat-ish around a dominant energy band.
    if has_high and not has_low:
        return [round(_HIGH_E, 4)] * points, "high"
    if has_low and not has_high and not has_mid:
        return [round(_LOW_E, 4)] * points, "low"
    if has_mid and not has_low and not has_high:
        return [round(_MID_E, 4)] * points, "mid"

    # Default: gentle rise then fall.
    return _default_arc(points), "default"


def _parse_bpm(low: str) -> Optional[tuple]:
    """Detect BPM hints like '124 bpm', '120-128 bpm', 'around 126 bpm'."""
    # Range first: "120-128 bpm" or "120 to 128 bpm".
    m = re.search(r"(\d{2,3})\s*(?:-|to|–)\s*(\d{2,3})\s*(?:bpm|beats)", low)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    # Single value followed by bpm.
    m = re.search(r"(\d{2,3})\s*(?:bpm|beats per minute|beats)", low)
    if m:
        v = int(m.group(1))
        return (v, v)
    return None


def _parse_length(low: str) -> Optional[int]:
    """Detect a desired track count like '10 tracks', 'set of 8', '12 songs'."""
    m = re.search(r"(\d{1,3})\s*(?:tracks?|songs?|tunes?|cuts?)", low)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"(?:set|playlist|mix)\s+of\s+(\d{1,3})", low)
    if m:
        return max(1, int(m.group(1)))
    return None


def _parse_genres(raw: str, low: str) -> list[str]:
    """Pull out plausible genre keywords. Heuristic and conservative.

    Recognises a small built-in vocabulary of common dance-music genres plus
    capitalised words that look like genre names in the original text.
    """
    vocab = (
        "house", "deep house", "tech house", "techno", "minimal", "trance",
        "progressive", "drum and bass", "dnb", "garage", "dubstep", "ambient",
        "downtempo", "disco", "funk", "soul", "hip hop", "hip-hop", "trap",
        "afro", "afro house", "melodic techno", "electro", "breakbeat",
        "jungle", "acid", "lo-fi", "lofi",
    )
    found: list[str] = []
    for g in vocab:
        if re.search(r"\b" + re.escape(g) + r"\b", low):
            if g not in found:
                found.append(g)
    return found


def _summarize(shape, genres, bpm_hint, length, attributes=None) -> str:
    parts = [f"{shape} energy arc"]
    if genres:
        parts.append("genres: " + ", ".join(genres))
    a = attributes or {}
    if a.get("gender"):
        parts.append(f"{a['gender']} vocal")
    elif a.get("vocal"):
        parts.append(a["vocal"])
    if a.get("moods"):
        parts.append("mood: " + ", ".join(a["moods"]))
    if a.get("instruments"):
        parts.append("with " + ", ".join(s.lower() for s in a["instruments"]))
    if bpm_hint:
        lo, hi = bpm_hint
        parts.append(f"{lo} bpm" if lo == hi else f"{lo}-{hi} bpm")
    if length:
        parts.append(f"{length} tracks")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Attribute constraints (vocal / gender / mood / instruments) + Camelot keys
# ---------------------------------------------------------------------------

_MOOD_WORDS = {name for name, _v, _a in MOOD_ANCHORS}
_INSTRUMENT_WORDS = {
    "guitar": "Guitar", "electric guitar": "Electric guitar", "piano": "Piano",
    "synth": "Synthesizer", "synthesizer": "Synthesizer", "saxophone": "Saxophone",
    "sax": "Saxophone", "trumpet": "Trumpet", "violin": "Violin, fiddle",
    "strings": "String section", "drums": "Drum kit", "cowbell": "Cowbell",
    "organ": "Organ", "flute": "Flute", "bass": "Bass guitar", "accordion": "Accordion",
}

# note+mode -> (camelot number, wheel letter). Majors are the B ring, minors the A ring.
_CAMELOT = {
    ("c", "maj"): (8, "B"), ("g", "maj"): (9, "B"), ("d", "maj"): (10, "B"),
    ("a", "maj"): (11, "B"), ("e", "maj"): (12, "B"), ("b", "maj"): (1, "B"),
    ("f#", "maj"): (2, "B"), ("gb", "maj"): (2, "B"), ("c#", "maj"): (3, "B"),
    ("db", "maj"): (3, "B"), ("g#", "maj"): (4, "B"), ("ab", "maj"): (4, "B"),
    ("d#", "maj"): (5, "B"), ("eb", "maj"): (5, "B"), ("a#", "maj"): (6, "B"),
    ("bb", "maj"): (6, "B"), ("f", "maj"): (7, "B"),
    ("a", "min"): (8, "A"), ("e", "min"): (9, "A"), ("b", "min"): (10, "A"),
    ("f#", "min"): (11, "A"), ("gb", "min"): (11, "A"), ("c#", "min"): (12, "A"),
    ("db", "min"): (12, "A"), ("g#", "min"): (1, "A"), ("ab", "min"): (1, "A"),
    ("d#", "min"): (2, "A"), ("eb", "min"): (2, "A"), ("a#", "min"): (3, "A"),
    ("bb", "min"): (3, "A"), ("f", "min"): (4, "A"), ("c", "min"): (5, "A"),
    ("g", "min"): (6, "A"), ("d", "min"): (7, "A"),
}


def _to_camelot(key_str):
    if not key_str:
        return None
    parts = str(key_str).strip().lower().split()
    if len(parts) < 2:
        return None
    note, tail = parts[0], parts[1]
    mode = "maj" if "maj" in tail else ("min" if "min" in tail else None)
    if mode is None:
        return None
    return _CAMELOT.get((note, mode))


def _camelot_label(c) -> Optional[str]:
    return f"{c[0]}{c[1]}" if c else None


def _camelot_distance(c1, c2) -> float:
    """0.0 = harmonically compatible (same key, relative major/minor, or +/-1 on
    the wheel); larger = farther. Unknown keys are neutral (0.5)."""
    if not c1 or not c2:
        return 0.5
    n1, l1 = c1
    n2, l2 = c2
    if c1 == c2 or n1 == n2:           # same key, or relative major/minor
        return 0.0
    dn = min((n1 - n2) % 12, (n2 - n1) % 12)
    if l1 == l2 and dn == 1:           # adjacent on the same ring
        return 0.0
    return min(1.0, dn / 6.0 + (0.0 if l1 == l2 else 0.25))


def parse_attributes(low: str) -> dict:
    """Pull sound/vocal constraints out of a vibe description."""
    attrs = {"vocal": None, "gender": None, "moods": [], "instruments": []}
    if any(w in low for w in ("instrumental", "no vocal", "no vocals", "without vocals")):
        attrs["vocal"] = "instrumental"
    elif any(w in low for w in ("vocal", "vocals", "sung", "acapella", "a capella")):
        attrs["vocal"] = "vocal"
    if re.search(r"\bfemale\b", low):
        attrs["gender"], attrs["vocal"] = "female", attrs["vocal"] or "vocal"
    elif re.search(r"\bmale\b", low):
        attrs["gender"], attrs["vocal"] = "male", attrs["vocal"] or "vocal"
    for m in _MOOD_WORDS:
        if re.search(r"\b" + re.escape(m) + r"\b", low):
            attrs["moods"].append(m)
    for word, label in _INSTRUMENT_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", low) and label not in attrs["instruments"]:
            attrs["instruments"].append(label)
    return attrs


def _track_attributes(store, candidates, analysis) -> dict:
    """tid -> {vocal, gender, moods:set, instruments:set} for candidates that have
    an AudioSet vector. Empty when tagging hasn't run (graceful no-op)."""
    from mgc.tagging import get_audioset_labels
    from mgc.understanding import compile_record

    labels = get_audioset_labels()
    out: dict = {}
    if not labels:
        return out
    for tid in candidates:
        u = store.get_understanding(tid)
        if not u or u.get("audioset") is None:
            continue
        rec = compile_record(u["audioset"], labels, analysis=analysis.get(tid) or {})
        out[tid] = {
            "vocal": rec["vocal"]["voice_instrumental"],
            "gender": rec["vocal"]["gender"],
            "moods": set(rec["mood"].get("tags") or []),
            "instruments": set(rec.get("instruments") or {}),
        }
    return out


# ---------------------------------------------------------------------------
# Set construction
# ---------------------------------------------------------------------------

def _resample_arc(arc: list[float], length: int) -> list[float]:
    """Resample a coarse arc to exactly ``length`` target energies (linear)."""
    if length <= 0:
        return []
    if not arc:
        arc = [_MID_E]
    if length == 1:
        return [round(float(arc[len(arc) // 2]), 4)]
    if len(arc) == 1:
        return [round(float(arc[0]), 4)] * length

    src = len(arc) - 1
    out = []
    for i in range(length):
        pos = (i / (length - 1)) * src
        lo = int(pos)
        hi = min(lo + 1, src)
        frac = pos - lo
        val = arc[lo] * (1.0 - frac) + arc[hi] * frac
        out.append(round(float(val), 4))
    return out


def _genre_track_ids(store, parsed: dict, restrict_genre_ids) -> Optional[set]:
    """Track ids biased toward requested genres, or None if no genre filter.

    Resolves restrict_genre_ids (explicit) first; otherwise matches parsed
    genre keywords against genre node names. Returns the set of track ids
    assigned to any matching genre, or None when nothing applies.
    """
    genre_ids: list[int] = []
    if restrict_genre_ids:
        genre_ids = list(restrict_genre_ids)
    elif parsed.get("genres"):
        wanted = [g.lower() for g in parsed["genres"]]
        try:
            for node in store.iter_genres():
                name = (node.name or "").lower()
                if any(w in name or name in w for w in wanted):
                    genre_ids.append(node.id)
        except Exception:
            return None

    if not genre_ids:
        return None

    track_ids: set = set()
    for gid in genre_ids:
        try:
            for row in store.conn.execute(
                "SELECT track_id FROM assignments WHERE genre_id=?", (gid,)
            ).fetchall():
                track_ids.add(row["track_id"])
        except Exception:
            continue
    return track_ids or None


def build_set(
    store,
    description: str,
    model: str,
    length: Optional[int] = None,
    restrict_genre_ids: Optional[list] = None,
) -> dict:
    """Build an ordered set from a vibe ``description``.

    Returns dict with:
      - track_ids: ordered list[int] (unique)
      - arc: list[float] per-position target energies (len == len(track_ids))
      - reasons: list[str] one short reason per chosen track
      - parsed: the parse_description() dict
    """
    parsed = parse_description(description)

    # --- candidate pool ---------------------------------------------------
    ids, _mat = store.load_matrix(model)
    candidates = list(ids)
    if not candidates:
        # No embeddings for this model: fall back to every track.
        candidates = [t.id for t in store.iter_tracks()]

    # Optional genre bias: keep only matching tracks if any remain.
    genre_ids = _genre_track_ids(store, parsed, restrict_genre_ids)
    if genre_ids is not None:
        biased = [tid for tid in candidates if tid in genre_ids]
        if biased:
            candidates = biased

    analysis = store.load_analysis()

    def energy_of(tid: int) -> float:
        a = analysis.get(tid)
        if a and a.get("energy") is not None:
            return float(a["energy"])
        return 0.5

    def bpm_of(tid: int):
        a = analysis.get(tid)
        if a and a.get("bpm") is not None:
            return float(a["bpm"])
        return None

    # --- attribute constraints (vocal/gender/mood/instruments) ------------
    attrs_req = parsed.get("attributes") or {}
    has_attr_req = bool(attrs_req.get("vocal") or attrs_req.get("gender")
                        or attrs_req.get("moods") or attrs_req.get("instruments"))
    track_attrs = _track_attributes(store, candidates, analysis) if has_attr_req else {}

    # Hard-filter on vocal/gender when we have the data and matches remain.
    if track_attrs and (attrs_req.get("vocal") or attrs_req.get("gender")):
        def _ok(tid: int) -> bool:
            ta = track_attrs.get(tid)
            if ta is None:
                return False
            if attrs_req.get("gender"):
                return ta["gender"] == attrs_req["gender"]
            return ta["vocal"] == attrs_req["vocal"] or ta["vocal"] == "uncertain"
        eligible = [tid for tid in candidates if _ok(tid)]
        if eligible:
            candidates = eligible

    # Camelot key per candidate (always, for harmonic continuity tie-break).
    camelot = {tid: _to_camelot((analysis.get(tid) or {}).get("music_key")) for tid in candidates}

    def attr_score(tid: int) -> float:
        ta = track_attrs.get(tid)
        if ta is None:
            return 0.0
        want = list(attrs_req.get("moods", [])) + list(attrs_req.get("instruments", []))
        if not want:
            return 0.0
        hit = sum(1 for m in attrs_req.get("moods", []) if m in ta["moods"])
        hit += sum(1 for i in attrs_req.get("instruments", []) if i in ta["instruments"])
        return hit / len(want)

    n_cand = len(candidates)
    if n_cand == 0:
        return {"track_ids": [], "arc": [], "reasons": [], "parsed": parsed}

    # --- choose length ----------------------------------------------------
    want = length if length is not None else parsed.get("length")
    if want is None:
        want = 12
    target_len = max(1, min(int(want), n_cand))

    targets = _resample_arc(parsed["energy_arc"], target_len)

    # --- greedy selection: energy arc first, then attribute match, then
    #     harmonic (Camelot) continuity, then smooth BPM -------------------
    chosen: list[int] = []
    reasons: list[str] = []
    used: set = set()
    prev_bpm: Optional[float] = None
    prev_cam = None

    for target in targets:
        best = None
        best_key = None
        for tid in candidates:
            if tid in used:
                continue
            e_dist = abs(energy_of(tid) - target)
            b = bpm_of(tid)
            bpm_dist = abs(b - prev_bpm) if (prev_bpm is not None and b is not None) else 0.0
            cam_dist = _camelot_distance(prev_cam, camelot.get(tid)) if prev_cam is not None else 0.0
            # Energy follows the arc (primary); then prefer attribute matches,
            # then harmonically compatible keys, then a smooth BPM step.
            key = (round(e_dist, 3), round(-attr_score(tid), 3), round(cam_dist, 3), bpm_dist, tid)
            if best_key is None or key < best_key:
                best_key = key
                best = tid

        if best is None:
            break
        used.add(best)
        chosen.append(best)
        e = energy_of(best)
        b = bpm_of(best)
        prev_bpm = b if b is not None else prev_bpm
        prev_cam = camelot.get(best) or prev_cam

        # --- per-track reason, leading with matched attributes ------------
        bits: list[str] = []
        ta = track_attrs.get(best)
        if ta:
            if attrs_req.get("gender") and ta["gender"] == attrs_req["gender"]:
                bits.append(f"{ta['gender']} vocal")
            elif attrs_req.get("vocal"):
                bits.append(ta["vocal"])
            bits.extend(m for m in attrs_req.get("moods", []) if m in ta["moods"])
            bits.extend(i.lower() for i in attrs_req.get("instruments", []) if i in ta["instruments"])
        base = f"energy {e:.2f}"
        if b is not None:
            base += f", {int(round(b))} bpm"
        cam_label = _camelot_label(camelot.get(best))
        if cam_label:
            base += f", {cam_label}"
        reasons.append(" · ".join([*bits, base]))

    return {
        "track_ids": chosen,
        "arc": targets[: len(chosen)],
        "reasons": reasons,
        "parsed": parsed,
    }
