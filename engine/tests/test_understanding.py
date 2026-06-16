"""Understanding compiler — instruments/vocal/gender/mood/caption from an
AudioSet probability vector + analysis. No models, pure derivation."""

from __future__ import annotations

import numpy as np

from mgc.understanding import compile_record


def test_compile_female_vocal_guitar():
    labels = ["Music", "Electric guitar", "Drum kit", "Female singing", "Male singing", "Singing"]
    v = np.zeros(len(labels), np.float32)
    v[labels.index("Electric guitar")] = 0.8
    v[labels.index("Drum kit")] = 0.5
    v[labels.index("Female singing")] = 0.7
    v[labels.index("Singing")] = 0.75
    rec = compile_record(v, labels, analysis={"energy": 0.8, "music_key": "C maj", "bpm": 124.0})

    assert "Electric guitar" in rec["instruments"]
    assert rec["vocal"]["voice_instrumental"] == "vocal"
    assert rec["vocal"]["gender"] == "female"
    assert rec["mood"]["arousal"] == 0.8
    assert rec["mood"]["valence"] > 0.5                      # major key -> brighter
    cap = rec["caption"].lower()
    assert "female vocal" in cap and "124 bpm" in cap and "electric guitar" in cap
    assert "instrumental" not in rec["tags_canonical"]


def test_compile_instrumental_minor():
    labels = ["Music", "Piano", "Singing", "Female singing", "Male singing"]
    v = np.zeros(len(labels), np.float32)
    v[labels.index("Piano")] = 0.6                            # no singing at all
    rec = compile_record(v, labels, analysis={"energy": 0.3, "music_key": "A min"})

    assert rec["vocal"]["voice_instrumental"] == "instrumental"
    assert rec["vocal"]["gender"] == "unknown"
    assert "Piano" in rec["instruments"]
    assert rec["mood"]["valence"] < 0.5                       # minor + low energy
    assert "instrumental" in rec["caption"].lower()
    assert "instrumental" in rec["tags_canonical"]


def test_named_moods_are_distinct_per_region():
    from mgc.understanding.moods import named_moods

    bright = named_moods(0.9, 0.88)[0]["mood"]      # happy + hyped
    dark = named_moods(0.18, 0.86)[0]["mood"]       # negative + hyped
    calm = named_moods(0.6, 0.20)[0]["mood"]        # low arousal
    assert bright in {"euphoric", "uplifting", "anthemic", "energetic", "joyful"}
    assert dark in {"aggressive", "intense", "tense", "dark"}
    assert calm in {"dreamy", "ethereal", "chill"}
    assert len({bright, dark, calm}) == 3            # three different moods


def test_compile_surfaces_named_moods():
    labels = ["Music", "Piano"]
    v = np.zeros(len(labels), np.float32)
    v[labels.index("Piano")] = 0.5
    rec = compile_record(v, labels, analysis={"energy": 0.9, "music_key": "C maj", "bpm": 128})
    assert rec["mood"]["tags"]                       # named moods present
    assert rec["mood"]["tags"][0].capitalize() in rec["caption"]  # caption leads with the mood
