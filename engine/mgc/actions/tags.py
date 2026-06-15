"""Genre tag reading/writing via mutagen, with an undoable action log.

Writes the specific subgenre into the file's single genre field so DJ software
(Rekordbox etc.) sees it directly. Format-specific frames are used: ID3 ``TCON``
for MP3, Vorbis ``GENRE`` for FLAC/OGG, and the ``\xa9gen`` atom for MP4/M4A.
Optionally the parent genre is written to a grouping field.
"""

from __future__ import annotations

import os
from typing import Optional

from mgc.types import ACTION_TAG


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def read_genre(path: str) -> Optional[str]:
    """Return the file's genre string, or None if unset/unreadable.

    Uses mutagen's easy/format-aware readers per extension.
    """
    ext = _ext(path)
    try:
        if ext == ".mp3":
            from mutagen.id3 import ID3, ID3NoHeaderError

            try:
                tags = ID3(path)
            except ID3NoHeaderError:
                return None
            frame = tags.get("TCON")
            if frame is None:
                return None
            vals = list(frame.text)
            return str(vals[0]) if vals else None
        if ext in (".flac", ".ogg", ".oga"):
            if ext == ".flac":
                from mutagen.flac import FLAC

                audio = FLAC(path)
            else:
                from mutagen.oggvorbis import OggVorbis

                audio = OggVorbis(path)
            vals = audio.get("genre") or audio.get("GENRE")
            if not vals:
                return None
            return str(vals[0])
        if ext in (".m4a", ".mp4", ".aac"):
            from mutagen.mp4 import MP4

            audio = MP4(path)
            vals = audio.tags.get("\xa9gen") if audio.tags else None
            if not vals:
                return None
            return str(vals[0])
    except Exception:
        return None
    return None


def _write_genre_to_file(
    path: str,
    subgenre: str,
    parent: Optional[str],
    write_parent_to_grouping: bool,
) -> None:
    """Persist genre (and optional grouping) into the file, preserving other tags."""
    ext = _ext(path)
    if ext == ".mp3":
        from mutagen.id3 import ID3, TCON, TIT1, ID3NoHeaderError

        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        tags.setall("TCON", [TCON(encoding=3, text=[subgenre])])
        if write_parent_to_grouping and parent:
            tags.setall("TIT1", [TIT1(encoding=3, text=[parent])])
        tags.save(path)
    elif ext in (".flac", ".ogg", ".oga"):
        if ext == ".flac":
            from mutagen.flac import FLAC

            audio = FLAC(path)
        else:
            from mutagen.oggvorbis import OggVorbis

            audio = OggVorbis(path)
        audio["genre"] = [subgenre]
        if write_parent_to_grouping and parent:
            audio["grouping"] = [parent]
        audio.save()
    elif ext in (".m4a", ".mp4", ".aac"):
        from mutagen.mp4 import MP4

        audio = MP4(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags["\xa9gen"] = [subgenre]
        if write_parent_to_grouping and parent:
            audio.tags["\xa9grp"] = [parent]
        audio.save()
    else:
        raise RuntimeError(f"Unsupported audio format for tagging: {ext or path}")


def write_genre(
    store,
    track,
    subgenre: str,
    parent: Optional[str] = None,
    write_parent_to_grouping: bool = False,
    dry_run: bool = False,
) -> dict:
    """Write ``subgenre`` into the track's genre field and log the action.

    Returns ``{"track_id", "path", "from", "to"}`` where ``from`` is the genre
    before the write. When ``dry_run`` no file or DB mutation occurs.
    """
    path = track.path
    old = read_genre(path)
    result = {"track_id": track.id, "path": path, "from": old, "to": subgenre}
    if dry_run:
        return result

    _write_genre_to_file(path, subgenre, parent, write_parent_to_grouping)
    store.log_action(
        ACTION_TAG,
        track.id,
        from_value=old or "",
        to_value=subgenre,
        undo_token=path,
    )
    return result


def undo_tags(store) -> int:
    """Restore prior genre for each 'done' tag_write action, newest-first.

    For each restored action the file's genre is rewritten to ``from_value``
    (an empty string clears the field) and the action is marked 'undone'.
    Returns the number of actions reverted.
    """
    actions = store.iter_actions(status="done", type=ACTION_TAG)
    count = 0
    for action in reversed(actions):  # newest-first
        path = action.undo_token
        prior = action.from_value or ""
        try:
            if path:
                if prior:
                    _write_genre_to_file(path, prior, None, False)
                else:
                    _clear_genre(path)
        except Exception:
            pass
        store.set_action_status(action.id, "undone")
        count += 1
    return count


def _clear_genre(path: str) -> None:
    """Remove the genre field entirely, preserving other tags."""
    ext = _ext(path)
    try:
        if ext == ".mp3":
            from mutagen.id3 import ID3, ID3NoHeaderError

            try:
                tags = ID3(path)
            except ID3NoHeaderError:
                return
            tags.delall("TCON")
            tags.save(path)
        elif ext in (".flac", ".ogg", ".oga"):
            if ext == ".flac":
                from mutagen.flac import FLAC

                audio = FLAC(path)
            else:
                from mutagen.oggvorbis import OggVorbis

                audio = OggVorbis(path)
            for key in ("genre", "GENRE"):
                if key in audio:
                    del audio[key]
            audio.save()
        elif ext in (".m4a", ".mp4", ".aac"):
            from mutagen.mp4 import MP4

            audio = MP4(path)
            if audio.tags and "\xa9gen" in audio.tags:
                del audio.tags["\xa9gen"]
                audio.save()
    except Exception:
        pass
