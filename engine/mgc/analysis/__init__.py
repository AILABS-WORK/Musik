"""Audio analysis: BPM / musical key / energy / danceability.

Public surface:
    - ``analyze_samples(samples, sr)`` -> dict of features (numpy/scipy only).
    - ``analyze_track(path, target_sr)`` -> dict; decodes then analyzes; ``{}`` on
      decode error.
    - ``analyze_all(store, progress=None)`` -> int; analyzes every track lacking a
      stored analysis and persists it.

The audio-decode import is LAZY (inside ``analyze_track``) to keep this module
independent of the audio module at import time. ``librosa`` is never required.
"""

from __future__ import annotations

from mgc.analysis.features import analyze_samples

__all__ = ["analyze_samples", "analyze_track", "analyze_all"]


def analyze_track(path: str, target_sr: int = 22050) -> dict:
    """Decode ``path`` to mono at ``target_sr`` and return its feature dict.

    Returns ``{}`` if the file cannot be decoded.
    """
    # Lazy cross-module import: keeps analysis independent of audio at import time.
    from mgc.audio.decode import AudioDecodeError, load_mono

    try:
        samples, sr = load_mono(path, target_sr)
    except AudioDecodeError:
        return {}
    except Exception:  # noqa: BLE001 - any unexpected decode failure -> empty
        return {}
    return analyze_samples(samples, sr)


def analyze_all(store, progress=None) -> int:
    """Analyze every track without stored analysis; persist and count them.

    For each track where ``store.has_analysis`` is False, decode + analyze and
    call ``store.save_analysis(...)``. ``progress``, if given, is called as
    ``progress(done, total)`` after each track. Tracks that fail to decode (empty
    result) are skipped and not counted.
    """
    tracks = store.iter_tracks()
    total = len(tracks)
    analyzed = 0
    for done, track in enumerate(tracks, start=1):
        if not store.has_analysis(track.id):
            feats = analyze_track(track.path, target_sr=track.sample_rate or 22050)
            if feats:
                store.save_analysis(
                    track.id,
                    bpm=feats.get("bpm"),
                    music_key=feats.get("music_key"),
                    energy=feats.get("energy"),
                    danceability=feats.get("danceability"),
                )
                analyzed += 1
        if progress is not None:
            progress(done, total)
    return analyzed
