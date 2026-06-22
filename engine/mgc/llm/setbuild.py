"""LLM-reasoned DJ set ordering.

The model gets the vibe description + a compact table of candidate tracks (bpm, key,
energy, genre) and returns an ordered tracklist that follows the described energy/BPM
arc, mixes harmonically, and keeps tempo steps smooth — the kind of judgement keyword
heuristics can't do (e.g. never opening a "light, builds gradually" set at 167 BPM).
Pure: the caller supplies candidates and validates the result.
"""

from __future__ import annotations

import json

from mgc.llm import ollama

_SYSTEM = (
    "You are an expert DJ building ONE continuous set from a crate. You are given a "
    "vibe description and candidate tracks, each as: id | title | BPM | key(Camelot) | "
    "energy(0-1) | genre. SELECT and ORDER tracks so the set follows the description's "
    "arc precisely: honour where it should START (e.g. light/groovy/low energy + lower "
    "BPM) and how it should DEVELOP (e.g. gradually more dancy => energy and BPM rise "
    "over the set). Mix harmonically (prefer adjacent Camelot keys), keep BPM steps "
    "small and mostly monotonic with the arc, and respect requested genres. "
    "Return ONLY JSON: {\"set\":[{\"id\":<int>,\"reason\":\"<why here>\"}]} with exactly "
    "the requested number of tracks, every id taken from the candidates, no repeats, in "
    "play order."
)


def _row(c: dict) -> str:
    bpm = int(c["bpm"]) if c.get("bpm") else "?"
    return "id=%d | %s | %sbpm | %s | e%.2f | %s" % (
        c["id"], (c.get("name") or "")[:55], bpm, c.get("key") or "?",
        float(c.get("energy") or 0.0), c.get("genre") or "?")


def llm_build_set(description: str, candidates: list, target_len: int,
                  minutes: int | None = None, model: str | None = None) -> dict | None:
    """Return ``{track_ids, reasons, model}`` or None if the LLM path is unavailable."""
    if not candidates or not ollama.available():
        return None
    table = "\n".join(_row(c) for c in candidates)
    span = f"{target_len} tracks" + (f" (~{minutes} min)" if minutes else "")
    user = (f"Description: {description}\n\nBuild a set of exactly {span}.\n"
            f"Candidates ({len(candidates)}):\n{table}")
    try:
        out = ollama.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            model=model)
        data = json.loads(out)
    except Exception:
        return None

    valid = {c["id"] for c in candidates}
    ids: list[int] = []
    reasons: list[str] = []
    seen: set = set()
    for item in data.get("set", []) if isinstance(data, dict) else []:
        tid = item.get("id") if isinstance(item, dict) else None
        if isinstance(tid, int) and tid in valid and tid not in seen:
            ids.append(tid)
            seen.add(tid)
            reasons.append(str(item.get("reason", ""))[:140])
    if not ids:
        return None
    return {"track_ids": ids, "reasons": reasons, "model": model or ollama.pick_model()}
