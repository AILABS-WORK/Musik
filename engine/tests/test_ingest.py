"""Unit tests for the ingest module (scanner)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from mgc.ingest import content_hash, read_tags, scan


def test_content_hash_stable_for_same_file(make_tone, tmp_path):
    p = make_tone(tmp_path / "x.wav", freq=440.0, seconds=1.0)
    h1 = content_hash(p)
    h2 = content_hash(p)
    assert isinstance(h1, str) and len(h1) == 40  # sha1 hexdigest
    assert h1 == h2


def test_content_hash_differs_between_files(make_tone, tmp_path):
    a = make_tone(tmp_path / "a.wav", freq=200.0, seconds=1.0, seed=1)
    b = make_tone(tmp_path / "b.wav", freq=900.0, seconds=1.0, seed=2)
    assert content_hash(a) != content_hash(b)


def test_read_tags_returns_dict_without_raising(make_tone, tmp_path):
    p = make_tone(tmp_path / "notags.wav", freq=440.0, seconds=0.5)
    tags = read_tags(p)
    assert isinstance(tags, dict)


def test_read_tags_on_bad_path_returns_empty(tmp_path):
    bogus = tmp_path / "does_not_exist.wav"
    assert read_tags(str(bogus)) == {}


def test_read_tags_on_non_audio_returns_empty(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not a real wav file")
    assert read_tags(str(junk)) == {}


def test_scan_returns_ids_and_counts(tmp_store, tmp_library):
    lib, paths = tmp_library
    n = len(paths["a"]) + len(paths["b"])
    ids = scan(tmp_store, str(lib))
    assert isinstance(ids, list)
    assert len(ids) == n
    assert all(isinstance(i, int) for i in ids)
    assert tmp_store.count_tracks() == n


def test_scan_is_idempotent(tmp_store, tmp_library):
    lib, paths = tmp_library
    n = len(paths["a"]) + len(paths["b"])
    scan(tmp_store, str(lib))
    assert tmp_store.count_tracks() == n
    # Re-running must not create duplicates (dedup by content_hash).
    ids2 = scan(tmp_store, str(lib))
    assert len(ids2) == n
    assert tmp_store.count_tracks() == n


def test_scan_populates_track_fields(tmp_store, tmp_library):
    lib, paths = tmp_library
    ids = scan(tmp_store, str(lib))
    t = tmp_store.get_track(ids[0])
    assert t is not None
    assert t.content_hash and len(t.content_hash) == 40
    assert t.fmt == "wav"
    assert t.duration is not None and t.duration > 0
    assert t.sample_rate == 22050


def test_scan_respects_extensions_filter(tmp_store, tmp_path):
    lib = tmp_path / "mixed"
    lib.mkdir()
    # one real wav
    sr = 22050
    sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
    sf.write(str(lib / "tone.wav"), sig, sr)
    # one non-audio file that should be ignored
    (lib / "readme.txt").write_text("ignore me")
    ids = scan(tmp_store, str(lib), extensions=(".wav",))
    assert len(ids) == 1
    assert tmp_store.count_tracks() == 1


def test_scan_recurses_subdirectories(tmp_store, tmp_path, make_tone):
    lib = tmp_path / "root"
    sub = lib / "nested" / "deep"
    sub.mkdir(parents=True)
    make_tone(lib / "top.wav", freq=300.0, seconds=0.5, seed=1)
    make_tone(sub / "bottom.wav", freq=700.0, seconds=0.5, seed=2)
    ids = scan(tmp_store, str(lib))
    assert len(ids) == 2


def test_scan_skips_unreadable_files_but_continues(tmp_store, tmp_path, make_tone):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_tone(lib / "good.wav", freq=440.0, seconds=0.5)
    # A file with .wav extension but corrupt content: hashing works, soundfile.info fails.
    (lib / "corrupt.wav").write_bytes(b"RIFF....not really audio")
    ids = scan(tmp_store, str(lib))
    # corrupt file still gets a Track (hash + tags ok), duration/sr just None.
    assert len(ids) == 2
    corrupt = tmp_store.get_track_by_hash(content_hash(str(lib / "corrupt.wav")))
    assert corrupt is not None
    assert corrupt.duration is None
    assert corrupt.sample_rate is None


def test_scan_reports_progress(tmp_store, tmp_library):
    lib, paths = tmp_library
    seen = []
    scan(tmp_store, str(lib), progress=lambda path, idx: seen.append((path, idx)))
    assert len(seen) == len(paths["a"]) + len(paths["b"])
