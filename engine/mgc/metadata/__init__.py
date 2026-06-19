"""MusicBrainz metadata: lookups (genres/tags/year/region) + by-example seeding."""

from mgc.metadata import acoustid, discogs
from mgc.metadata.bootstrap import seed_genres_from_mb
from mgc.metadata.genre_graph import GenreGraph, get_graph
from mgc.metadata.musicbrainz import (
    genre_vocabulary, mb_lookup, mb_lookup_by_mbid, parse_recording,
)
from mgc.metadata.parse import parse_artist_title

__all__ = ["mb_lookup", "mb_lookup_by_mbid", "parse_recording", "genre_vocabulary",
           "seed_genres_from_mb", "GenreGraph", "get_graph", "acoustid", "discogs",
           "parse_artist_title"]
