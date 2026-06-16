"""MusicBrainz metadata lookups (genres, tags, year, region) via the /ws/2 API.

MusicBrainz hosts no audio, so we use it as a *label oracle*: authoritative, CC0
genres / tags / year / region for a track once we can name it (artist + title).
Those labels feed the by-example classifier (see bootstrap.py) and enrich the
understanding record. Network + best-effort, honors the OS trust store, respects
the ~1 request/second rate limit, and sends a real User-Agent (required by MB).
Pure parsers are split out so they can be tested without the network.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

_BASE = "https://musicbrainz.org/ws/2"
_UA = "Musik/0.1 ( miguelito.villax@gmail.com )"
_MIN_INTERVAL = 1.05  # ~1 req/s; MB returns 503 if you go faster
_last_call = [0.0]


def _get(url: str, timeout: float = 12.0) -> dict:
    """Throttled GET against MusicBrainz, returning parsed JSON."""
    try:
        from mgc._net import enable_os_truststore

        enable_os_truststore()
    except Exception:
        pass
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _genre_names(entity: dict, min_count: int = 0) -> list[str]:
    """Curated genre names from an entity's ``genres[]`` (vote-thresholdable)."""
    out = []
    for g in entity.get("genres") or []:
        if g.get("name") and (g.get("count") or 0) >= min_count:
            out.append(g["name"])
    return out


def parse_recording(data: dict) -> dict:
    """Pure parser for a recording lookup -> {genres, tags, year, artist, ...}."""
    ac = data.get("artist-credit") or []
    artist = ac[0]["artist"]["name"] if ac else None
    artist_mbid = ac[0]["artist"]["id"] if ac else None
    releases = data.get("releases") or []
    rg_mbid = (releases[0].get("release-group") or {}).get("id") if releases else None
    return {
        "genres": _genre_names(data),
        "tags": [t["name"] for t in (data.get("tags") or []) if t.get("name")],
        "year": (data.get("first-release-date") or "")[:4] or None,
        "artist": artist,
        "artist_mbid": artist_mbid,
        "release_group_mbid": rg_mbid,
    }


def mb_lookup(artist: str, title: str | None = None, get=None) -> dict:
    """MusicBrainz metadata for a track by ``artist`` [+ ``title``].

    Returns ``{}`` (or ``{"error": ...}``) on miss, else a dict with
    ``recording_mbid, genres, tags, year, artist, area``. Falls back recording ->
    release-group -> artist for genre coverage. ``get`` overrides the HTTP backend.
    """
    get = get or _get
    if not artist:
        return {}
    try:
        if title:
            q = f'recording:"{title}" AND artist:"{artist}"'
            search = get(f"{_BASE}/recording?query={urllib.parse.quote(q)}&fmt=json&limit=1")
            recs = search.get("recordings") or []
            if recs:
                mbid = recs[0]["id"]
                full = get(f"{_BASE}/recording/{mbid}"
                           "?inc=genres+tags+artist-credits+releases+release-groups&fmt=json")
                info = parse_recording(full)
                info["recording_mbid"] = mbid
                if not info["genres"] and info.get("release_group_mbid"):
                    rg = get(f"{_BASE}/release-group/{info['release_group_mbid']}?inc=genres+tags&fmt=json")
                    info["genres"] = _genre_names(rg)
                if info.get("artist_mbid"):
                    art = get(f"{_BASE}/artist/{info['artist_mbid']}?inc=genres+tags&fmt=json")
                    if not info["genres"]:
                        info["genres"] = _genre_names(art)
                    info["area"] = (art.get("area") or {}).get("name")
                return info
        # artist-only fallback (no title, or no recording match)
        q = f'artist:"{artist}"'
        search = get(f"{_BASE}/artist?query={urllib.parse.quote(q)}&fmt=json&limit=1")
        arts = search.get("artists") or []
        if arts:
            mbid = arts[0]["id"]
            art = get(f"{_BASE}/artist/{mbid}?inc=genres+tags&fmt=json")
            return {"artist_mbid": mbid, "artist": art.get("name"),
                    "genres": _genre_names(art),
                    "tags": [t["name"] for t in (art.get("tags") or [])],
                    "area": (art.get("area") or {}).get("name")}
    except Exception as e:  # noqa: BLE001 - network/proxy/etc.
        return {"error": str(e)[:140]}
    return {}


def genre_vocabulary(get=None, max_pages: int = 30) -> list[str]:
    """The MusicBrainz canonical genre list via /ws/2/genre/all (paginated)."""
    get = get or _get
    names: list[str] = []
    for page in range(max_pages):
        try:
            data = get(f"{_BASE}/genre/all?fmt=json&limit=100&offset={page * 100}")
        except Exception:
            break
        chunk = data.get("genres") or []
        names.extend(g["name"] for g in chunk if g.get("name"))
        if len(chunk) < 100:
            break
    return names
