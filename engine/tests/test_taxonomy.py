"""Tests for the taxonomy module.

These build a tiny synthetic RYM-shaped fixture under tmp_path (mirroring the
real ``_index.json`` + ``main/*.json`` + ``detailed/*.json`` schema) so they do
not depend on the gitignored real reference data.
"""

from __future__ import annotations

import json

from mgc.taxonomy import parse_rym, seed_taxonomy
from mgc.types import LEVEL_GENRE, LEVEL_SUBGENRE


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _make_refs(tmp_path):
    """Create a minimal references dir mimicking the RYM export schema.

    Two top genres (Electronic, Blues). Electronic has 2 sub genres, one of
    which (House) has a deeper sub-2 child (Acid House). Blues has 1 sub genre.
    """
    refs = tmp_path / "references"
    refs.mkdir()

    _write_json(refs / "_index.json", {
        "total_main_genres": 2,
        "genres": [
            {"name": "Electronic", "description": "Electronic instrumentation.",
             "url": "https://example/genre/electronic/"},
            {"name": "Blues", "description": "Originated in the Deep South.",
             "url": "https://example/genre/blues/"},
        ],
    })

    _write_json(refs / "main" / "electronic.json", {
        "name": "Electronic",
        "description": "Electronic instrumentation.",
        "url": "https://example/genre/electronic/",
        "sub_genres_count": 2,
        "sub_genres": [
            {"name": "House", "url": "https://example/genre/house/",
             "description": "Four-on-the-floor dance music.",
             "level": "sub", "parent": "Electronic"},
            {"name": "Ambient", "url": "https://example/genre/ambient/",
             "description": "Texture over structure.",
             "level": "sub", "parent": "Electronic"},
        ],
    })

    _write_json(refs / "main" / "blues.json", {
        "name": "Blues",
        "description": "Originated in the Deep South.",
        "url": "https://example/genre/blues/",
        "sub_genres_count": 1,
        "sub_genres": [
            {"name": "Electric Blues", "url": "https://example/genre/electric-blues/",
             "description": "Amplified small-combo blues.",
             "level": "sub", "parent": "Blues"},
        ],
    })

    _write_json(refs / "detailed" / "house.json", {
        "name": "House",
        "description": "Four-on-the-floor dance music.",
        "url": "https://example/genre/house/",
        "parent": "Electronic",
        "level": "sub",
        "children_count": 1,
        "children": [
            {"name": "Acid House", "url": "https://example/genre/acid-house/",
             "description": "Squelchy TB-303 basslines.",
             "level": "sub-2", "parent": "House"},
        ],
    })

    return refs


# ---- parse_rym ----------------------------------------------------------

def test_parse_rym_levels_and_parent_names(tmp_path):
    refs = _make_refs(tmp_path)
    nodes = parse_rym(str(refs))
    by_name = {n.name: n for n in nodes}

    # All expected names present, no duplicates.
    assert set(by_name) == {"Electronic", "Blues", "House", "Ambient",
                            "Electric Blues", "Acid House"}
    assert len(nodes) == 6

    # Top buckets are LEVEL_GENRE with no parent name.
    assert by_name["Electronic"].level == LEVEL_GENRE
    assert by_name["Blues"].level == LEVEL_GENRE

    # Deeper nodes are LEVEL_SUBGENRE.
    for nm in ("House", "Ambient", "Electric Blues", "Acid House"):
        assert by_name[nm].level == LEVEL_SUBGENRE

    # Descriptions are carried through.
    assert by_name["Acid House"].description == "Squelchy TB-303 basslines."

    # Parents precede children in order (so id resolution works).
    order = [n.name for n in nodes]
    assert order.index("Electronic") < order.index("House")
    assert order.index("House") < order.index("Acid House")
    assert order.index("Blues") < order.index("Electric Blues")


def test_parse_rym_missing_subdirs(tmp_path):
    """Index only, no main/ or detailed/ dirs -> just the top genres."""
    refs = tmp_path / "references"
    refs.mkdir()
    _write_json(refs / "_index.json", {
        "genres": [{"name": "Jazz", "description": "Improvisation.", "url": "x"}],
    })
    nodes = parse_rym(str(refs))
    assert [n.name for n in nodes] == ["Jazz"]
    assert nodes[0].level == LEVEL_GENRE


# ---- seed_taxonomy ------------------------------------------------------

def test_seed_taxonomy_links_and_levels(tmp_store, tmp_path):
    refs = _make_refs(tmp_path)
    count = seed_taxonomy(tmp_store, str(refs))
    assert count == 6

    genres = tmp_store.iter_genres()
    assert len(genres) == 6

    by_name = {g.name: g for g in genres}

    # Top genres have no parent.
    assert by_name["Electronic"].parent_id is None
    assert by_name["Blues"].parent_id is None
    assert by_name["Electronic"].level == LEVEL_GENRE

    # Sub genres link to their top genre.
    assert by_name["House"].parent_id == by_name["Electronic"].id
    assert by_name["Ambient"].parent_id == by_name["Electronic"].id
    assert by_name["Electric Blues"].parent_id == by_name["Blues"].id
    assert by_name["House"].level == LEVEL_SUBGENRE

    # Deeper sub-2 child links to its sub genre.
    assert by_name["Acid House"].parent_id == by_name["House"].id
    assert by_name["Acid House"].level == LEVEL_SUBGENRE

    # children() reflects the linkage.
    elec_children = {g.name for g in tmp_store.children(by_name["Electronic"].id)}
    assert elec_children == {"House", "Ambient"}
    assert [g.name for g in tmp_store.children(by_name["House"].id)] == ["Acid House"]


def test_seed_taxonomy_idempotent(tmp_store, tmp_path):
    refs = _make_refs(tmp_path)
    first = seed_taxonomy(tmp_store, str(refs))
    count_after_first = len(tmp_store.iter_genres())

    second = seed_taxonomy(tmp_store, str(refs))
    count_after_second = len(tmp_store.iter_genres())

    assert first == second == 6
    assert count_after_first == count_after_second == 6

    # Parent links survive a re-seed unchanged.
    by_name = {g.name: g for g in tmp_store.iter_genres()}
    assert by_name["House"].parent_id == by_name["Electronic"].id
    assert by_name["Acid House"].parent_id == by_name["House"].id


def test_seed_taxonomy_limit(tmp_store, tmp_path):
    refs = _make_refs(tmp_path)
    # Limit to just the two top genres (they are emitted first).
    count = seed_taxonomy(tmp_store, str(refs), limit=2)
    assert count == 2
    names = {g.name for g in tmp_store.iter_genres()}
    assert names == {"Electronic", "Blues"}
