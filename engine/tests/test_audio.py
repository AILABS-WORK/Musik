"""Unit tests for the audio decode layer (mgc.audio.decode).

Exercises only this module. Uses the shared ``make_tone`` fixture to write
deterministic wavs; no heavy ML deps required.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgc.audio import AudioDecodeError, load_mono, load_windows


def test_load_mono_shape_dtype_and_resample(make_tone, tmp_path):
    src_sr = 22050
    seconds = 2.0
    path = make_tone(tmp_path / "tone.wav", freq=440.0, seconds=seconds, sr=src_sr)

    target_sr = 16000
    samples, sr = load_mono(path, target_sr)

    assert sr == target_sr
    assert samples.ndim == 1
    assert samples.dtype == np.float32

    expected = int(round(seconds * src_sr * target_sr / src_sr))
    # length should be ~ original * target/source, within a small tolerance
    assert abs(samples.shape[0] - expected) <= 5
    assert np.isfinite(samples).all()


def test_load_mono_downmixes_stereo(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 660 * t).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr)

    samples, out_sr = load_mono(str(path), sr)
    assert samples.ndim == 1
    assert out_sr == sr
    assert samples.shape[0] == sr
    # mean of the two channels (atol covers 16-bit PCM quantization on write)
    np.testing.assert_allclose(samples, (left + right) / 2.0, atol=2e-4)


def test_load_mono_same_sr_no_resample(make_tone, tmp_path):
    sr = 22050
    path = make_tone(tmp_path / "same.wav", seconds=1.0, sr=sr)
    samples, out_sr = load_mono(path, sr)
    assert out_sr == sr
    assert samples.shape[0] == sr  # 1 second exactly, no resampling drift


def test_load_windows_length_and_dtype(make_tone, tmp_path):
    target_sr = 16000
    window_seconds = 5.0
    path = make_tone(tmp_path / "long.wav", seconds=12.0, sr=22050)

    windows = load_windows(
        path, target_sr, window_seconds=window_seconds, hop_seconds=window_seconds
    )
    assert isinstance(windows, list)
    assert len(windows) >= 1
    win_len = int(round(window_seconds * target_sr))
    for w in windows:
        assert isinstance(w, np.ndarray)
        assert w.ndim == 1
        assert w.dtype == np.float32
        assert w.shape[0] == win_len


def test_load_windows_count_consistent_with_duration(make_tone, tmp_path):
    target_sr = 16000
    window_seconds = 5.0
    # 12 seconds non-overlapping 5s windows => starts at 0,5,(and tail at 7)
    path = make_tone(tmp_path / "dur.wav", seconds=12.0, sr=22050)
    windows = load_windows(
        path, target_sr, window_seconds=window_seconds, hop_seconds=window_seconds
    )
    # whole-track coverage: more than a single leading window
    assert len(windows) >= 2
    # and not absurdly many for 12s of audio
    assert len(windows) <= 4


def test_load_windows_short_track_zero_padded(make_tone, tmp_path):
    target_sr = 16000
    window_seconds = 5.0
    # 2s track, shorter than one 5s window => exactly one padded window
    path = make_tone(tmp_path / "short.wav", seconds=2.0, sr=22050)
    windows = load_windows(
        path, target_sr, window_seconds=window_seconds, hop_seconds=window_seconds
    )
    assert len(windows) == 1
    win_len = int(round(window_seconds * target_sr))
    w = windows[0]
    assert w.shape[0] == win_len
    # tail is zero padding
    assert np.all(w[-target_sr:] == 0.0)
    # leading region has real signal
    assert np.any(w[: target_sr] != 0.0)


def test_load_windows_respects_max_windows(make_tone, tmp_path):
    target_sr = 16000
    window_seconds = 1.0
    path = make_tone(tmp_path / "many.wav", seconds=30.0, sr=22050)
    windows = load_windows(
        path,
        target_sr,
        window_seconds=window_seconds,
        hop_seconds=window_seconds,
        max_windows=5,
    )
    assert len(windows) <= 5
    win_len = int(round(window_seconds * target_sr))
    for w in windows:
        assert w.shape[0] == win_len


def test_decode_garbage_raises(tmp_path):
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"this is not a real wav file" * 10)
    with pytest.raises(AudioDecodeError):
        load_mono(str(bad), 16000)


def test_load_windows_propagates_decode_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"\x00\x01garbage\xff" * 20)
    with pytest.raises(AudioDecodeError):
        load_windows(str(bad), 16000)
