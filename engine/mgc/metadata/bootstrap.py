"""Seed by-example genre centroids from MusicBrainz genres on the user's OWN tracks.

MusicBrainz gives authoritative genre LABELS; we already have the AUDIO and its
embeddings. So attach MB genres to the user's identified tracks and group their
embeddings by MB genre to auto-build centroids in our own embedding space (Option
A in the design). This bootstraps the by-example classifier from ground truth.
Each seeded genre is a normal by-example genre afterward, so the user can keep,
extend, or delete it. We never overwrite a genre the user already defined.
"""

from __future__ import annotations


def seed_genres_from_mb(store, model: str, resolve=None, min_examples: int = 3,
                        level: str = "subgenre", progress=None) -> dict:
    """Group the library by MusicBrainz genre and create a by-example centroid for
    each genre that has at least ``min_examples`` embedded tracks.

    ``resolve(track) -> list[str]`` returns MB genre names for a track (default: a
    live MusicBrainz lookup from the track's artist/title tags). Injectable for tests.
    Skips genres that already exist. Returns ``{genre_name: n_examples_used}``.
    """
    if resolve is None:
        from mgc.metadata.musicbrainz import mb_lookup

        def resolve(track):
            tags = track.existing_tags or {}
            artist = tags.get("artist") or tags.get("albumartist")
            if not artist:
                return []
            return mb_lookup(artist, tags.get("title")).get("genres") or []

    ids, _mat = store.load_matrix(model)
    embedded = set(ids)

    tracks = [t for t in store.iter_tracks() if t.id in embedded]
    by_genre: dict[str, list[int]] = {}
    for i, t in enumerate(tracks):
        for g in resolve(t) or []:
            name = (g or "").strip()
            if name:
                by_genre.setdefault(name, []).append(t.id)
        if progress:
            progress(i + 1, len(tracks))

    from mgc.registry.centroids import create_genre_by_example

    existing = {(g.name or "").lower() for g in store.iter_genres()}
    created: dict[str, int] = {}
    for genre, tids in by_genre.items():
        if len(tids) < min_examples or genre.lower() in existing:
            continue
        try:
            create_genre_by_example(store, genre, tids, model, level=level)
            created[genre] = len(tids)
        except Exception:
            continue
    return created
