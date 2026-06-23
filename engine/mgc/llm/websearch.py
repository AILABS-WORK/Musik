"""Best-effort web search for grounding genre guesses (stdlib only).

The local LLM hallucinates obscure genre labels from memory. Feeding it real web
snippets about the actual track/artist grounds the guess. We hit DuckDuckGo's
lite HTML endpoint (no API key, no JS) and scrape short text snippets.

EVERYTHING here is best-effort: on any network/proxy/parse failure we return an
empty list and the caller falls back to the ungrounded path. The TLS-inspecting
proxy on this machine means ``enable_os_truststore()`` must run before the
request or cert validation fails.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request

from mgc._net import enable_os_truststore

_URL = "https://lite.duckduckgo.com/lite/?q="
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# DuckDuckGo lite puts result text in <td> cells inside the results table; the
# result-snippet / result-link classes carry the useful prose.
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.I | re.S)


def _clean(fragment: str) -> str:
    text = _TAG.sub(" ", fragment)
    text = html.unescape(text)
    text = _WS.sub(" ", text).strip()
    return text[:200]


def search_snippets(query: str, n: int = 6, timeout: float = 8.0) -> list[str]:
    """Return up to ``n`` short text snippets for ``query`` (<=200 chars each).

    Best-effort: returns ``[]`` on ANY exception (network, proxy, parse)."""
    enable_os_truststore()
    try:
        url = _URL + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        body = raw.decode("utf-8", "replace")
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for cell in _CELL.findall(body):
        text = _clean(cell)
        # skip empties, pure navigation/numbering, and dupes
        if len(text) < 12 or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= n:
            break
    return out
