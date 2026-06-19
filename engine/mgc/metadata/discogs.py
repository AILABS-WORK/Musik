"""Discogs metadata lookup (genres + the granular 'styles' DJs actually use).

Discogs is the underground-electronic database: its STYLES (Tech House, Minimal
Techno, Deep House, Dub Techno) are exactly the subgenre granularity we want, with
far better coverage of free-DL / label releases than AcoustID or MusicBrainz. We
authenticate with a consumer key + secret as query params (no OAuth user flow is
needed for database/search). Matching is by text (artist/title), so it is fuzzy and
best-effort. Pure parsing is split out so it can be tested without the network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

_BASE = "https://api.discogs.com"
_UA = "Musik/0.1 +https://github.com/AILABS-WORK/Musik"
_MIN_INTERVAL = 1.1  # authenticated Discogs allows ~60 requests/minute
_last_call = [0.0]


def creds(key: str | None = None, secret: str | None = None) -> tuple[str | None, str | None]:
    if key and secret:
        return key, secret
    try:
        from mgc._env import load_env

        load_env()
    except Exception:
        pass
    return os.environ.get("DISCOGS_KEY"), os.environ.get("DISCOGS_SECRET")


def _get(url: str, timeout: float = 12.0) -> dict:
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


def parse_search(data: dict) -> dict:
    """Pure: take the first search result that carries a genre/style.

    Discogs ``style`` is the fine subgenre (Tech House, Minimal Techno); ``genre`` is
    the broad family (Electronic, House). We return both, styles first.
    """
    for res in data.get("results") or []:
        styles = res.get("style") or []
        genres = res.get("genre") or []
        if styles or genres:
            return {
                "styles": styles,
                "genres": genres,
                "year": str(res.get("year") or "") or None,
                "discogs_id": res.get("id"),
                "title": res.get("title"),
            }
    return {}


def lookup(artist: str | None, title: str | None = None,
           key: str | None = None, secret: str | None = None, get=None) -> dict:
    """Search Discogs for a release by artist [+ title] -> {styles, genres, year}.

    Best-effort: ``{}`` / ``{"error": ...}`` on miss. Falls back from an
    artist+track search to a free-text query when the structured search is empty.
    """
    get = get or _get
    k, s = creds(key, secret)
    if not k or not s:
        return {"error": "no_discogs_creds"}
    if not artist and not title:
        return {}
    auth = {"key": k, "secret": s, "per_page": 5, "type": "release"}
    try:
        params = dict(auth, artist=artist or "")
        if title:
            params["track"] = title
        out = parse_search(get(f"{_BASE}/database/search?" + urllib.parse.urlencode(params)))
        if out:
            return out
        # fuzzy free-text fallback ("artist title")
        q = " ".join(x for x in (artist, title) if x).strip()
        if q:
            params = dict(auth, q=q)
            return parse_search(get(f"{_BASE}/database/search?" + urllib.parse.urlencode(params)))
        return {}
    except Exception as e:
        return {"error": str(e)[:200]}
