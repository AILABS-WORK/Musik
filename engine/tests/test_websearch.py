"""Web-grounded genre guess (no real network)."""

from __future__ import annotations

from mgc.llm import genre as genre_mod
from mgc.llm import websearch


def test_strip_html_yields_short_clean_snippets():
    sample = (
        "<html><body><table>"
        "<tr><td><a class='result-link'>Charlotte de Witte &mdash; Formula</a></td></tr>"
        "<tr><td class='result-snippet'>Belgian DJ known for <b>techno</b> &amp; "
        "acid&nbsp;techno releases on KNTXT.</td></tr>"
        "<tr><td>x</td></tr>"  # too short, dropped
        "</table></body></html>"
    )
    snippets = websearch._CELL.findall(sample)
    cleaned = [websearch._clean(s) for s in snippets if len(websearch._clean(s)) >= 12]
    assert any("techno" in s for s in cleaned)
    assert all("<" not in s and ">" not in s for s in cleaned)  # tags stripped
    assert all(len(s) <= 200 for s in cleaned)
    assert "&" in "Charlotte de Witte & Formula" and \
        any("&" in s and "&amp;" not in s for s in cleaned)  # entities unescaped


def test_grounded_returns_none_when_no_snippets(monkeypatch):
    # LLM "available" so we exercise the empty-snippets branch, not the no-ollama one.
    monkeypatch.setattr(genre_mod.ollama, "available", lambda: True)

    def boom(*a, **k):  # must never be called when snippets are empty
        raise AssertionError("LLM should not be called with no snippets")

    monkeypatch.setattr(genre_mod.ollama, "chat", boom)
    res = genre_mod.llm_genre_grounded(
        "Some Artist", "Some Title", "Some Label", ["techno"], 130,
        search=lambda q, *a, **k: [])
    assert res is None


def test_grounded_uses_snippets_and_returns_shape(monkeypatch):
    monkeypatch.setattr(genre_mod.ollama, "available", lambda: True)
    seen = {}

    def fake_chat(messages, **k):
        seen["user"] = messages[-1]["content"]
        return '{"genre": "Acid Techno", "confidence": 0.8}'

    monkeypatch.setattr(genre_mod.ollama, "chat", fake_chat)
    res = genre_mod.llm_genre_grounded(
        "Charlotte de Witte", "Formula", "KNTXT", ["techno"], 135,
        search=lambda q, *a, **k: ["Belgian techno DJ, acid techno on KNTXT."])
    assert res["genre"] == "Acid Techno"
    assert res["confidence"] == 0.8
    assert res["grounded"] is True
    assert res["plausible"] is True  # 135 BPM fits acid techno
    assert "acid techno" in seen["user"].lower()  # snippet reached the prompt


def test_search_snippets_returns_empty_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("proxy blocked")

    monkeypatch.setattr(websearch.urllib.request, "urlopen", boom)
    assert websearch.search_snippets("anything") == []
