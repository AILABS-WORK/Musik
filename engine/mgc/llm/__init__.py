"""Local LLM (Ollama) for *reasoning* decisions — set ordering, naming.

Never used for audio grounding (the model can't hear): MuQ embeddings / by-example
stay the source of truth for what a track sounds like. The LLM only reasons over the
metadata we extract. Everything here is best-effort; callers fall back to heuristics
when Ollama isn't running.
"""

from mgc.llm import ollama

__all__ = ["ollama"]
