"""MusicBrainz parsing + by-example genre seeding (no network)."""

from __future__ import annotations

from mgc.embed import embed_track
from mgc.embed.baseline import BaselineEmbedder
from mgc.metadata import parse_recording, seed_genres_from_mb
from mgc.types import Track


def test_parse_recording_extracts_genres_and_year():
    data = {
        "first-release-date": "2019-05-03",
        "genres": [{"name": "deep house", "count": 5}, {"name": "house", "count": 2}],
        "tags": [{"name": "chill", "count": 3}],
        "artist-credit": [{"artist": {"id": "abc", "name": "Some Artist"}}],
        "releases": [{"release-group": {"id": "rg1"}}],
    }
    rec = parse_recording(data)
    assert rec["genres"] == ["deep house", "house"]
    assert rec["year"] == "2019"
    assert rec["artist"] == "Some Artist"
    assert rec["release_group_mbid"] == "rg1"


def test_seed_genres_from_mb_builds_centroids(tmp_store, tmp_path, make_tone):
    """3 tracks labeled 'Deep House' seed a centroid; 'Techno' (only 2) does not."""
    embedder = BaselineEmbedder()
    plan = [("a", "Deep House"), ("b", "Deep House"), ("c", "Deep House"),
            ("d", "Techno"), ("e", "Techno")]
    label_of = {}
    for i, (name, genre) in enumerate(plan):
        p = make_tone(tmp_path / f"{name}.wav", freq=200 + i * 200, seconds=2.0, seed=i)
        tid = tmp_store.upsert_track(Track(path=p, content_hash=f"h{name}"))
        embed_track(tmp_store, embedder, tmp_store.get_track(tid))
        label_of[tid] = [genre]

    created = seed_genres_from_mb(tmp_store, "baseline",
                                  resolve=lambda t: label_of.get(t.id, []),
                                  min_examples=3)
    assert created == {"Deep House": 3}
    names = {g.name for g in tmp_store.iter_genres()}
    assert "Deep House" in names and "Techno" not in names


def test_seed_skips_existing_genres(tmp_store, tmp_path, make_tone):
    embedder = BaselineEmbedder()
    tids = []
    for i in range(3):
        p = make_tone(tmp_path / f"x{i}.wav", freq=300 + i * 100, seconds=2.0, seed=i)
        tid = tmp_store.upsert_track(Track(path=p, content_hash=f"hx{i}"))
        embed_track(tmp_store, embedder, tmp_store.get_track(tid))
        tids.append(tid)

    from mgc.registry.centroids import create_genre_by_example
    create_genre_by_example(tmp_store, "House", tids[:1], "baseline")

    created = seed_genres_from_mb(tmp_store, "baseline",
                                  resolve=lambda _t: ["House"], min_examples=2)
    assert created == {}  # already exists -> skipped
