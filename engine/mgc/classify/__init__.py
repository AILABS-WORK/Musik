"""Genre suggestion (centroid cosine + optional zero-shot blend)."""

from mgc.classify.classifier import ancestors, suggest, suggest_all

__all__ = ["suggest", "suggest_all", "ancestors"]
