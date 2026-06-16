"""Unit tests for the audio analysis module (mgc.analysis).

Exercises only this module + the store + decode layer. Uses numpy-synthesized
audio and the shared ``make_tone`` / ``tmp_library`` / ``tmp_store`` fixtures.
No heavy ML deps, no model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgc.analysis import analyze_all, analyze_samples, analyze_track
from mgc.types import Track

SR = 22050


def _am_tone(carrier_hz=220.0, mod_hz=2.0, seconds=8.0, sr=SR, seed=0):
    """A carrier sine amplitude-modulated at ``mod_hz`` Hz.

    A 2 Hz modulation => 2 amplitude pulses per second => ~120 BPM, giving a
    clear periodic onset structure for the tempo estimator.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    # Modulator in [0,1]; sharpened so each beat is a distinct onset.
    mod = (0.5 * (1.0 + np.sin(2 * np.pi * mod_hz * t - np.pi / 2))) ** 4
    sig = carrier * mod
    sig = 0.4 * sig / (np.max(np.abs(sig)) + 1e-9)
    sig += 0.002 * rng.standard_normal(t.shape)
    return sig.astype(np.float32)


# ---- analyze_samples --------------------------------------------------------

def test_analyze_samples_keys_and_ranges():
    sig = _am_tone()
    out = analyze_samples(sig, SR)

    assert set(out.keys()) == {"bpm", "music_key", "energy", "danceability"}

    assert isinstance(out["bpm"], float)
    assert out["bpm"] > 0.0  # a periodic AM tone has a plausible tempo

    assert isinstance(out["energy"], float)
    assert 0.0 < out["energy"] <= 1.0

    assert isinstance(out["danceability"], float)
    assert 0.0 <= out["danceability"] <= 1.0

    assert out["music_key"] is None or isinstance(out["music_key"], str)


def test_analyze_samples_bpm_plausible_for_2hz_modulation():
    # 2 Hz amplitude modulation ~ 120 BPM. Allow octave errors (60/120/240) since
    # autocorrelation tempo estimation commonly lands on a metrical multiple.
    sig = _am_tone(mod_hz=2.0, seconds=10.0)
    out = analyze_samples(sig, SR)
    bpm = out["bpm"]
    assert 40.0 <= bpm <= 240.0
    # within an octave of 120 BPM
    candidates = [60.0, 120.0, 240.0]
    assert min(abs(bpm - c) for c in candidates) < 25.0


def test_analyze_samples_key_for_pure_tone_is_str_or_none():
    # A pure 440 Hz tone is A4 -> chroma dominated by pitch class A.
    t = np.linspace(0, 4.0, int(4.0 * SR), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    out = analyze_samples(sig, SR)
    assert out["music_key"] is None or isinstance(out["music_key"], str)
    if isinstance(out["music_key"], str):
        # If a key is reported, the tonic should relate to A for a pure A tone.
        assert out["music_key"].startswith("A")


def test_analyze_samples_empty_signal():
    out = analyze_samples(np.zeros(0, dtype=np.float32), SR)
    assert out["bpm"] == 0.0
    assert out["energy"] == 0.0
    assert out["danceability"] == 0.0
    assert out["music_key"] is None


def test_analyze_samples_louder_has_more_energy():
    t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
    quiet = (0.05 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    loud = (0.6 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    assert analyze_samples(loud, SR)["energy"] > analyze_samples(quiet, SR)["energy"]


def test_analyze_samples_energy_bounded_for_clipping_input():
    # Even a hot/over-unity signal must stay within [0,1].
    t = np.linspace(0, 1.0, SR, endpoint=False)
    hot = (5.0 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    e = analyze_samples(hot, SR)["energy"]
    assert 0.0 < e <= 1.0


# ---- analyze_track ----------------------------------------------------------

def test_analyze_track_returns_feature_dict(make_tone, tmp_path):
    path = make_tone(tmp_path / "tone.wav", freq=220.0, seconds=4.0, sr=SR,
                     harmonics=(1, 2, 3))
    out = analyze_track(path, target_sr=SR)
    assert set(out.keys()) == {"bpm", "music_key", "energy", "danceability"}
    assert isinstance(out["bpm"], float)
    assert 0.0 < out["energy"] <= 1.0
    assert 0.0 <= out["danceability"] <= 1.0
    assert out["music_key"] is None or isinstance(out["music_key"], str)


def test_analyze_track_bad_file_returns_empty(tmp_path):
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"not a real wav file" * 8)
    assert analyze_track(str(bad), target_sr=SR) == {}


# ---- analyze_all ------------------------------------------------------------

def _ingest_library(store, paths):
    """Register every wav in the library as a Track and return the ids."""
    ids = []
    for group in paths.values():
        for i, p in enumerate(group):
            tid = store.upsert_track(Track(
                path=str(p),
                content_hash=f"{p}",  # path is unique here
                fmt="wav",
                sample_rate=SR,
            ))
            ids.append(tid)
    return ids


def test_analyze_all_stores_analysis_for_every_track(tmp_store, tmp_library):
    _lib, paths = tmp_library
    ids = _ingest_library(tmp_store, paths)
    total = len(ids)
    assert total > 0

    n = analyze_all(tmp_store)
    assert n == total

    for tid in ids:
        assert tmp_store.has_analysis(tid) is True
        row = tmp_store.get_analysis(tid)
        assert row is not None
        assert 0.0 <= row["energy"] <= 1.0
        assert 0.0 <= row["danceability"] <= 1.0
        assert row["bpm"] is not None
        assert row["music_key"] is None or isinstance(row["music_key"], str)


def test_analyze_all_skips_already_analyzed(tmp_store, tmp_library):
    _lib, paths = tmp_library
    ids = _ingest_library(tmp_store, paths)

    # Pre-mark one track as analyzed; analyze_all must not redo it.
    tmp_store.save_analysis(ids[0], bpm=99.0, music_key="C maj", energy=0.5,
                            danceability=0.5)

    n = analyze_all(tmp_store)
    assert n == len(ids) - 1

    # The pre-existing analysis is untouched.
    row = tmp_store.get_analysis(ids[0])
    assert row["bpm"] == 99.0
    assert row["music_key"] == "C maj"


def test_analyze_all_reports_progress(tmp_store, tmp_library):
    _lib, paths = tmp_library
    ids = _ingest_library(tmp_store, paths)
    seen = []
    analyze_all(tmp_store, progress=lambda done, total: seen.append((done, total)))
    assert seen  # progress was invoked
    assert seen[-1] == (len(ids), len(ids))
