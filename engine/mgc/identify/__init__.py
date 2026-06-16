"""Identify a track by its sound.

``identify_in_library`` matches a query audio file against the library's stored
embeddings (cosine over ``store.load_matrix``). ``identify_external`` is an
optional AcoustID/Chromaprint fingerprint lookup.
"""

from mgc.identify.identify import identify_external, identify_in_library

__all__ = ["identify_in_library", "identify_external"]
