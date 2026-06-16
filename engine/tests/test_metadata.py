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


def test_genre_graph_related_expansion():
    from mgc.metadata import GenreGraph

    g = GenreGraph(
        genres=["house", "deep house", "tech house", "techno"],
        edges=[
            {"from": "deep house", "to": "house", "rel": "subgenre"},
            {"from": "tech house", "to": "house", "rel": "subgenre"},
            {"from": "tech house", "to": "techno", "rel": "fusion"},
        ],
    )
    assert g.children("house") == ["deep house", "tech house"] or set(g.children("house")) == {"deep house", "tech house"}
    assert g.parents("deep house") == ["house"]
    rel = {r["genre"] for r in g.related("house")}
    assert "deep house" in rel and "tech house" in rel  # children are related
    # 'techno' is reachable from house only via tech house (depth 2, fusion) -> weaker
    rel_techno = {r["genre"] for r in g.related("deep house", max_depth=2)}
    assert "house" in rel_techno and "tech house" in rel_techno


def test_bundled_genre_graph_loads():
    from mgc.metadata import get_graph

    g = get_graph()
    assert len(g.genres) > 1000          # the full MB vocabulary is bundled
    assert g.has("deep house") and g.has("techno")
    rel = {r["genre"] for r in g.related("house")}
    assert "deep house" in rel and "tech house" in rel  # curated edges work


def test_dump_build_graph_extracts_edges():
    from mgc.metadata.dump import build_graph

    genre = [["1", "g1", "house"], ["2", "g2", "deep house"], ["3", "g3", "techno"]]
    # link_type: id, parent, child_order, gid, e_type0, e_type1, name, ...
    link_type = [
        ["10", "", "0", "lt1", "genre", "genre", "subgenre", "d"],
        ["11", "", "0", "lt2", "genre", "genre", "fusion", "d"],
        ["99", "", "0", "lt3", "artist", "artist", "member of", "d"],
    ]
    link = [["100", "10"], ["101", "11"], ["102", "99"]]
    # l_genre_genre: id, link, entity0(child), entity1(parent), ...
    lgg = [["1000", "100", "2", "1"], ["1001", "101", "3", "1"]]

    g = build_graph(genre, link_type, link, lgg)
    assert set(g["genres"]) == {"house", "deep house", "techno"}
    edges = {(e["from"], e["to"], e["rel"]) for e in g["edges"]}
    assert ("deep house", "house", "subgenre") in edges
    assert ("techno", "house", "fusion") in edges
