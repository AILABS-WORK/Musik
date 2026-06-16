"""Named moods on the valence/arousal circumplex (Russell, 1980).

Rather than show only a raw (valence, arousal) coordinate, we place a rich
vocabulary of named moods as anchor points on that 2-D plane and return the
closest ones — so a track reads as "euphoric, driving" or "dark, hypnotic"
instead of two bare numbers. Tuned toward electronic/DJ vocabulary.

Pure geometry over the data we already derive (no extra models). Optionally
nudged by tempo so fast tracks lean "driving/peak-time" and slow ones "deep".
"""

from __future__ import annotations

# (name, valence 0..1, arousal 0..1) — circumplex anchors.
MOOD_ANCHORS: list[tuple[str, float, float]] = [
    ("euphoric", 0.92, 0.88),
    ("uplifting", 0.85, 0.70),
    ("energetic", 0.70, 0.92),
    ("joyful", 0.88, 0.62),
    ("playful", 0.78, 0.58),
    ("groovy", 0.68, 0.60),
    ("driving", 0.52, 0.84),     # peak-time: high energy, neutral valence
    ("anthemic", 0.72, 0.80),
    ("hypnotic", 0.45, 0.54),
    ("deep", 0.42, 0.42),
    ("warm", 0.72, 0.42),
    ("blissful", 0.82, 0.36),
    ("chill", 0.66, 0.26),
    ("dreamy", 0.60, 0.24),
    ("ethereal", 0.54, 0.20),
    ("romantic", 0.70, 0.46),
    ("nostalgic", 0.46, 0.34),
    ("melancholic", 0.28, 0.32),
    ("somber", 0.20, 0.24),
    ("moody", 0.35, 0.46),
    ("brooding", 0.26, 0.52),
    ("dark", 0.20, 0.60),
    ("tense", 0.26, 0.76),
    ("intense", 0.32, 0.86),
    ("aggressive", 0.18, 0.92),
]

_MAX_D = 0.65  # distance at which a mood's similarity reaches ~0


def named_moods(valence: float, arousal: float, bpm: float | None = None,
                k: int = 4, floor: float = 0.34) -> list[dict]:
    """Closest named moods to a (valence, arousal) point -> [{mood, score}].

    Scores are 0..1 (1 = bang on the anchor). ``bpm`` lightly nudges arousal so
    very fast tracks lean energetic and very slow ones lean calm.
    """
    a = float(arousal)
    if bpm:
        # gentle tempo nudge: ~+/-0.08 across 60..150 BPM, clamped
        a = max(0.0, min(1.0, a + (float(bpm) - 105.0) / 560.0))
    v = float(valence)
    scored = []
    for name, mv, ma in MOOD_ANCHORS:
        d = ((v - mv) ** 2 + (a - ma) ** 2) ** 0.5
        s = max(0.0, 1.0 - d / _MAX_D)
        scored.append((s, name))
    scored.sort(reverse=True)
    out = [{"mood": n, "score": round(s, 3)} for s, n in scored[:k] if s >= floor]
    return out or [{"mood": scored[0][1], "score": round(scored[0][0], 3)}]
