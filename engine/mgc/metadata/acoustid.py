"""AcoustID audio-fingerprint identification.

``fpcalc`` (Chromaprint) turns audio into a fingerprint; the AcoustID API maps that
fingerprint to a MusicBrainz recording MBID; ``musicbrainz.mb_lookup_by_mbid`` then
resolves the MBID to authoritative genre / region. This identifies a track by its
SOUND, so messy filenames and tags don't matter (the whole point: our SoundCloud
titles are unusable for plain MB search, but the audio fingerprint is exact).

Requirements:
* ``fpcalc`` binary -- path from ``MGC_FPCALC`` env, else ``fpcalc`` on PATH.
* a free AcoustID application API key -- ``MGC_ACOUSTID_KEY`` env (or passed in).
  Register one in a minute at https://acoustid.org/new-application

Network calls are best-effort and throttled; pure result-picking is split out so it
can be tested without the network or a key.
"""

from __future__ import annotations

import os
import time

_MIN_INTERVAL = 0.34  # AcoustID allows ~3 lookups/second
_last_call = [0.0]


def fpcalc_path() -> str:
    """Locate fpcalc: MGC_FPCALC env, else next to the venv python (where setup.ps1
    drops it), else assume it's on PATH."""
    env = os.environ.get("MGC_FPCALC")
    if env:
        return env
    import sys

    for name in ("fpcalc.exe", "fpcalc"):
        cand = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.exists(cand):
            return cand
    return "fpcalc"


def api_key(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("MGC_ACOUSTID_KEY") or None


def fingerprint(path: str) -> tuple[float, bytes]:
    """Return ``(duration_seconds, fingerprint)`` for an audio file via fpcalc."""
    os.environ.setdefault("FPCALC", fpcalc_path())
    import acoustid

    return acoustid.fingerprint_file(path)


def pick_best(results: list[dict]) -> dict | None:
    """Pure: choose the best AcoustID result that carries a MusicBrainz recording.

    Each AcoustID result has a ``score`` and may carry ``recordings``. We take the
    highest-scoring result that has at least one recording and return
    ``{score, recording_mbid, title, artist}``.
    """
    best = None
    for r in sorted(results or [], key=lambda x: x.get("score", 0.0), reverse=True):
        recs = r.get("recordings") or []
        if not recs:
            continue
        rec = recs[0]
        ac = rec.get("artists") or []
        return {
            "score": float(r.get("score", 0.0)),
            "recording_mbid": rec.get("id"),
            "title": rec.get("title"),
            "artist": ac[0]["name"] if ac else None,
        }
    return best


def lookup(fp: bytes, duration: float, key: str, meta: str = "recordings",
           looker=None) -> list[dict]:
    """Call the AcoustID lookup API; return the raw ``results`` list (or [])."""
    if looker is not None:
        return looker(fp, duration, key, meta)
    try:
        from mgc._net import enable_os_truststore

        enable_os_truststore()
    except Exception:
        pass
    import acoustid

    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    resp = acoustid.lookup(key, fp, duration, meta=meta)
    return resp.get("results") or []


def identify(path: str, key: str | None = None, looker=None,
             fingerprinter=None) -> dict:
    """Identify a file -> ``{recording_mbid, score, title, artist}`` (or ``{}``).

    Best-effort: returns ``{}`` on any failure (no key, fpcalc missing, no match,
    network error) so a batch can keep going.
    """
    k = api_key(key)
    if not k:
        return {"error": "no_acoustid_key"}
    try:
        dur, fp = (fingerprinter or fingerprint)(path)
        best = pick_best(lookup(fp, dur, k, looker=looker))
        return best or {}
    except Exception as e:  # fpcalc/network/parse failure -> skip this track
        return {"error": str(e)[:200]}
