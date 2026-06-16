"""MusicBrainz metadata: lookups (genres/tags/year/region) + by-example seeding."""

from mgc.metadata.bootstrap import seed_genres_from_mb
from mgc.metadata.genre_graph import GenreGraph, get_graph
from mgc.metadata.musicbrainz import genre_vocabulary, mb_lookup, parse_recording

__all__ = ["mb_lookup", "parse_recording", "genre_vocabulary", "seed_genres_from_mb",
           "GenreGraph", "get_graph"]
