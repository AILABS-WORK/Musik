"""Minimal Ollama client (stdlib only). Local LLM at http://localhost:11434."""

from __future__ import annotations

import json
import os
import urllib.request

_DEFAULT = "http://localhost:11434"
# preference order for a general chat model (skip embedding models). gemma3:4b is
# fast and reliable with JSON mode; some big tags (e.g. gemma4:12b) return empty
# content under format=json, so they are deprioritised.
_PREFERRED = ("gemma3:4b", "qwen2.5:7b", "llama3.1:8b", "qwen2.5:14b",
              "gemma3:12b", "llama3.2:3b", "gemma4:12b")


def base_url() -> str:
    return os.environ.get("OLLAMA_HOST", _DEFAULT).rstrip("/")


def available(timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(base_url() + "/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def models(timeout: float = 5.0) -> list[str]:
    try:
        raw = urllib.request.urlopen(base_url() + "/api/tags", timeout=timeout).read()
        return [m["name"] for m in json.loads(raw).get("models", [])]
    except Exception:
        return []


def pick_model(prefer: str | None = None) -> str | None:
    avail = models()
    if not avail:
        return None
    if prefer and prefer in avail:
        return prefer
    for m in _PREFERRED:
        if m in avail:
            return m
    for m in avail:                       # any non-embedding model
        if "embed" not in m.lower():
            return m
    return avail[0]


def chat(messages: list, model: str | None = None, json_mode: bool = True,
         temperature: float = 0.4, timeout: float = 180.0) -> str:
    """Return the assistant message content (a JSON string when ``json_mode``)."""
    model = model or pick_model()
    if not model:
        raise RuntimeError("no Ollama model available")
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature}}
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(
        base_url() + "/api/chat", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())["message"]["content"]
