"""Taxonomy module: parse and seed the RateYourMusic genre tree."""

from __future__ import annotations

from mgc.taxonomy.rym import parse_rym, seed_taxonomy

__all__ = ["parse_rym", "seed_taxonomy"]
