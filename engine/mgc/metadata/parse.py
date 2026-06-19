"""Pull a real (artist, title) out of messy SoundCloud / promo track names.

The ID3 ``artist`` on free-DL / promo files is usually the label or premiere channel
(BCCO, Mixmag premiere, Hot Meal Records), while the actual ``Artist - Title`` sits
in the title after a ``Premiere:`` prefix and before a ``[Label]`` suffix. Cleaning
this up lifts Discogs/MusicBrainz hit rates dramatically. Pure + best-effort.
"""

from __future__ import annotations

import os
import re

_EXT = re.compile(r"\.(mp3|flac|wav|m4a|aac|ogg|aiff?)$", re.I)
_PREMIERE = re.compile(r"^.*?premiere\s*[:|]\s*", re.I)
_FREE = re.compile(r"\(?\s*free\s*download\s*\)?", re.I)
_BRACKET = re.compile(r"\[[^\]]*\]")                       # [label / catalogue no.]
_MIX = re.compile(r"\b(original|extended|radio|club|dub|vocal)\s+(mix|edit|version)\b", re.I)
_LABELISH = re.compile(r"records?|recordings|premiere|mixmag|promo|\bfree\b|\bvinyl\b|music", re.I)
_DASH = re.compile(r"\s[-–—]\s")


def parse_artist_title(title: str | None, artist: str | None = None,
                       path: str | None = None) -> tuple[str | None, str | None]:
    """Return a best-effort ``(artist, title)`` for a messy track name."""
    raw = title or (os.path.basename(path) if path else "") or ""
    s = _EXT.sub("", raw)
    s = _PREMIERE.sub("", s)
    s = _FREE.sub("", s)
    s = _BRACKET.sub("", s)
    s = _MIX.sub("", s)
    s = s.strip(" -–—|:_\t")

    parts = [p.strip() for p in _DASH.split(s) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]

    # one chunk only: fall back to the ID3 artist if it isn't an obvious label/channel
    a = (artist or "").strip()
    if a and not _LABELISH.search(a):
        return a, (parts[0] if parts else None)
    return None, (parts[0] if parts else None)
