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

    notes = _summarize(shape, genres, bpm_hint, length)

    return {
        "genres": genres,
        "energy_arc": arc,
        "bpm_hint": bpm_hint,
        "length": length,
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


def _summarize(shape, genres, bpm_hint, length) -> str:
    parts = [f"{shape} energy arc"]
    if genres:
        parts.append("genres: " + ", ".join(genres))
    if bpm_hint:
        lo, hi = bpm_hint
        parts.append(f"{lo} bpm" if lo == hi else f"{lo}-{hi} bpm")
    if length:
        parts.append(f"{length} tracks")
    return "; ".join(parts)


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

    n_cand = len(candidates)
    if n_cand == 0:
        return {"track_ids": [], "arc": [], "reasons": [], "parsed": parsed}

    # --- choose length ----------------------------------------------------
    want = length if length is not None else parsed.get("length")
    if want is None:
        want = 12
    target_len = max(1, min(int(want), n_cand))

    targets = _resample_arc(parsed["energy_arc"], target_len)

    # --- greedy selection by energy proximity, BPM tie-break --------------
    chosen: list[int] = []
    reasons: list[str] = []
    used: set = set()
    prev_bpm: Optional[float] = None

    for target in targets:
        best = None
        best_key = None
        for tid in candidates:
            if tid in used:
                continue
            e = energy_of(tid)
            e_dist = abs(e - target)
            b = bpm_of(tid)
            if prev_bpm is not None and b is not None:
                bpm_dist = abs(b - prev_bpm)
            else:
                bpm_dist = 0.0
            # Energy dominates; BPM proximity is the smooth-mixing tie-break.
            key = (round(e_dist, 6), bpm_dist, tid)
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
        if b is not None:
            reasons.append(f"energy {e:.2f}, {int(round(b))} bpm")
        else:
            reasons.append(f"energy {e:.2f}")

    return {
        "track_ids": chosen,
        "arc": targets[: len(chosen)],
        "reasons": reasons,
        "parsed": parsed,
    }
