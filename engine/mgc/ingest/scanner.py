"""Filesystem scanner: hash, tag-read and upsert audio files into the Store.

Light deps only (stdlib + soundfile + mutagen). The Store API is the single
source of truth; this module just feeds it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Optional

import soundfile as sf

from mgc.config import Config
from mgc.types import Track

# Tag keys we care about surfacing into Track.existing_tags.
_INTERESTING_TAGS = ("genre", "artist", "title", "album")

# Bytes per chunk when streaming a file through the hash.
_HASH_CHUNK = 1 << 20  # 1 MiB


def content_hash(path: str) -> str:
    """Return a stable SHA-1 hex digest of the full file bytes (streamed)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tags(path: str) -> dict:
    """Read embedded tags via mutagen (easy mode).

    Returns ``{key: first_value}`` for genre/artist/title/album when present.
    Returns ``{}`` on any failure (unreadable / unsupported / no tags).
    """
    try:
        import mutagen  # lazy: keep import errors local and non-fatal

        meta = mutagen.File(path, easy=True)
        if not meta:
            return {}
        out: dict = {}
        for key in _INTERESTING_TAGS:
            try:
                val = meta.get(key)
            except Exception:
                val = None
            if not val:
                continue
            # easy tags are lists of strings; take the first value.
            out[key] = val[0] if isinstance(val, (list, tuple)) else val
        return out
    except Exception:
        return {}


def _audio_info(path: str) -> tuple[Optional[float], Optional[int]]:
    """Best-effort (duration_seconds, sample_rate) via soundfile; (None, None) on failure."""
    try:
        info = sf.info(path)
        duration = float(info.frames) / info.samplerate if info.samplerate else None
        return duration, int(info.samplerate) if info.samplerate else None
    except Exception:
        return None, None


def scan(
    store,
    root: str,
    extensions: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[str, int], None]] = None,
) -> list[int]:
    """Recursively scan ``root`` and upsert one Track per supported audio file.

    For each file whose suffix is in ``extensions`` (default ``Config().extensions``):
    compute the content hash, read tags, probe duration/sample-rate, and upsert a
    Track. Files that cannot be read at all are skipped; the scan keeps going.

    Returns the list of track ids upserted (idempotent via content_hash, so a
    re-scan of an unchanged library returns the same ids without duplicating rows).

    ``progress``, if given, is called as ``progress(path, index)`` once per
    upserted file (1-based index).
    """
    exts = tuple(extensions) if extensions is not None else tuple(Config().extensions)
    # Normalize to lowercase, dot-prefixed suffixes for matching.
    wanted = {e.lower() if e.startswith(".") else "." + e.lower() for e in exts}

    ids: list[int] = []
    root_path = Path(root)
    count = 0
    for p in sorted(root_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in wanted:
            continue
        path_str = str(p)
        try:
            chash = content_hash(path_str)
        except Exception:
            # Truly unreadable (cannot even read bytes) — skip but keep going.
            continue
        tags = read_tags(path_str)
        duration, sample_rate = _audio_info(path_str)
        track = Track(
            path=path_str,
            content_hash=chash,
            fmt=p.suffix.lower().lstrip("."),
            duration=duration,
            sample_rate=sample_rate,
            existing_tags=tags,
        )
        tid = store.upsert_track(track)
        ids.append(tid)
        count += 1
        if progress is not None:
            progress(path_str, count)
    return ids
