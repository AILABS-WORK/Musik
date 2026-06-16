"""Segment-level similarity (define a subgenre by a region of a waveform)."""

from mgc.segments.segments import (
    build_segment_index,
    embed_segment,
    find_similar_segments,
    index_track,
)

__all__ = ["embed_segment", "index_track", "build_segment_index", "find_similar_segments"]
