"""Tests for the taxonomy module.

These build a tiny synthetic RYM-shaped fixture under tmp_path (mirroring the
real ``_index.json`` + ``main/*.json`` + ``detailed/*.json`` schema) so they do
not depend on the gitignored real reference data.
"""

from __future__ import annotations

import json
import os

import pytest

from mgc.taxonomy import parse_rym, seed_taxonomy
from mgc.types import LEVEL_GENRE, LEVEL_SUBGENRE

# The real (gitignored) RateYourMusic export, if present. The optional tests at
# the bottom of this file only run when this directory exists.
REAL_REFS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "research", "repos", "joeseesun--music-genre-finder",
    "skill-source", "music-genre-finder", "references",
)
_HAVE_REAL_REFS = os.path.isdir(REAL_REFS_DIR)


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


# ---- DAG materialization (a genre under multiple parents) ----------------

def _make_dag_refs(tmp_path):
    """References dir where one sub-genre is listed under two parents.

    Electronic -> {House, Techno}; both House and Techno claim "Acid" as a
    deeper child. RYM's taxonomy is a DAG, so "Acid" legitimately appears under
    both. The full taxonomy is materialised as a tree, so "Acid" becomes two
    distinct rows (one per parent path), not one collapsed row.
    """
    refs = tmp_path / "references"
    refs.mkdir()

    _write_json(refs / "_index.json", {
        "genres": [
            {"name": "Electronic", "description": "Electronic.", "url": "x"},
        ],
    })
    _write_json(refs / "main" / "electronic.json", {
        "name": "Electronic",
        "sub_genres": [
            {"name": "House", "description": "House.", "level": "sub",
             "parent": "Electronic"},
            {"name": "Techno", "description": "Techno.", "level": "sub",
             "parent": "Electronic"},
        ],
    })
    _write_json(refs / "detailed" / "house.json", {
        "name": "House", "parent": "Electronic", "level": "sub",
        "children": [
            {"name": "Acid", "description": "Acid.", "level": "sub-2",
             "parent": "House"},
        ],
    })
    _write_json(refs / "detailed" / "techno.json", {
        "name": "Techno", "parent": "Electronic", "level": "sub",
        "children": [
            {"name": "Acid", "description": "Acid.", "level": "sub-2",
             "parent": "Techno"},
        ],
    })
    return refs


def test_parse_rym_dag_materializes_multi_parent(tmp_path):
    refs = _make_dag_refs(tmp_path)
    nodes = parse_rym(str(refs))
    # Electronic, House, Techno, Acid(under House), Acid(under Techno) = 5.
    assert len(nodes) == 5
    acid = [n for n in nodes if n.name == "Acid"]
    assert len(acid) == 2  # one per parent path -> not collapsed by name
    assert all(n.level == LEVEL_SUBGENRE for n in acid)


def test_seed_taxonomy_dag_two_parents(tmp_store, tmp_path):
    refs = _make_dag_refs(tmp_path)
    count = seed_taxonomy(tmp_store, str(refs))
    assert count == 5

    genres = tmp_store.iter_genres()
    by_name = {}
    for g in genres:
        by_name.setdefault(g.name, []).append(g)

    house = by_name["House"][0]
    techno = by_name["Techno"][0]
    acids = by_name["Acid"]
    assert len(acids) == 2
    acid_parents = {a.parent_id for a in acids}
    assert acid_parents == {house.id, techno.id}

    # Each parent sees exactly its own Acid child.
    assert [g.name for g in tmp_store.children(house.id)] == ["Acid"]
    assert [g.name for g in tmp_store.children(techno.id)] == ["Acid"]


def test_seed_taxonomy_dag_idempotent(tmp_store, tmp_path):
    refs = _make_dag_refs(tmp_path)
    first = seed_taxonomy(tmp_store, str(refs))
    rows1 = len(tmp_store.iter_genres())
    second = seed_taxonomy(tmp_store, str(refs))
    rows2 = len(tmp_store.iter_genres())
    assert first == second == 5
    assert rows1 == rows2 == 5  # re-seed adds nothing


# ---- optional: real RateYourMusic export (gitignored) -------------------

@pytest.mark.skipif(not _HAVE_REAL_REFS,
                    reason="real RYM references dir not present")
def test_parse_rym_real_full_taxonomy():
    """The full export ingests all tiers -> far more than the old ~2,600."""
    nodes = parse_rym(REAL_REFS_DIR)
    # Old (name-only dedup, partial detailed/) reached ~2,626. The full DAG
    # materialization recovers the complete taxonomy (~5,900 per _meta.json).
    assert len(nodes) > 4000

    # Top buckets are LEVEL_GENRE; deeper tiers are LEVEL_SUBGENRE.
    genres = [n for n in nodes if n.level == LEVEL_GENRE]
    subgenres = [n for n in nodes if n.level == LEVEL_SUBGENRE]
    assert 40 <= len(genres) <= 60          # ~49 top buckets
    assert len(subgenres) > 4000


@pytest.mark.skipif(not _HAVE_REAL_REFS,
                    reason="real RYM references dir not present")
def test_seed_taxonomy_real_count_and_idempotent(tmp_store):
    """Seeding the real export yields >4000 rows and is idempotent."""
    first = seed_taxonomy(tmp_store, REAL_REFS_DIR)
    rows1 = len(tmp_store.iter_genres())
    assert first > 4000
    assert rows1 > 4000

    # Re-seed must not add or duplicate any rows.
    second = seed_taxonomy(tmp_store, REAL_REFS_DIR)
    rows2 = len(tmp_store.iter_genres())
    assert second == first
    assert rows2 == rows1

    # Every non-root genre resolves to a real parent row; roots are LEVEL_GENRE.
    by_id = {g.id: g for g in tmp_store.iter_genres()}
    for g in by_id.values():
        if g.parent_id is None:
            assert g.level == LEVEL_GENRE
        else:
            assert g.parent_id in by_id
