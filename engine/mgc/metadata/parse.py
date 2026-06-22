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
# leading "GTG Premiere | ", "BCCO Premiere:", "Premiere:", "PREMIERE -"
_PREMIERE = re.compile(r"^.*?\bpremiere\b\s*[:|\-]?\s*", re.I)
_FREE = re.compile(r"\(?\s*free\s*download\s*\)?", re.I)
_BRACKET = re.compile(r"\[[^\]]*\]")                       # [label / catalogue no.]
# a parenthetical that describes a mix/edit/remix (drop it; keep e.g. "(SA)")
_PAREN_MIX = re.compile(r"\([^)]*\b(?:mix|edit|remix|version|bootleg|rework|dub|vip|remaster|master)\b[^)]*\)", re.I)
# leading vinyl position: "A1.", "B2 ", "01 -", "1. "
_VINYL = re.compile(r"^[A-E]?\d{1,2}\s*[.\-]\s+", re.I)
# trailing country/region tag DJs append: "Enoch (SA)", "Sterac (NL)"
_COUNTRY = re.compile(r"\s*\((?:[A-Z]{2,3}|[A-Z][a-z]+)\)\s*$")
_LABELISH = re.compile(r"records?|recordings|premiere|mixmag|promo|\bfree\b|\bvinyl\b|music|\bvol\b", re.I)
_DASH = re.compile(r"\s[-–—]\s")


def _clean_artist(a: str) -> str:
    return _COUNTRY.sub("", a).strip()


def parse_artist_title(title: str | None, artist: str | None = None,
                       path: str | None = None) -> tuple[str | None, str | None]:
    """Return a best-effort ``(artist, title)`` for a messy track name."""
    raw = title or (os.path.basename(path) if path else "") or ""
    s = _EXT.sub("", raw)
    s = _PREMIERE.sub("", s)
    s = _VINYL.sub("", s)
    s = _FREE.sub("", s)
    s = _BRACKET.sub("", s)
    s = _PAREN_MIX.sub("", s)
    s = s.strip(" -–—|:_\t")

    parts = [p.strip() for p in _DASH.split(s) if p.strip()]
    if len(parts) >= 2:
        return _clean_artist(parts[0]), parts[1]

    # one chunk only: fall back to the ID3 artist if it isn't an obvious label/channel
    a = (artist or "").strip()
    if a and not _LABELISH.search(a):
        return _clean_artist(a), (parts[0] if parts else None)
    return None, (parts[0] if parts else None)
