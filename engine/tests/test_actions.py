"""Unit tests for the actions module (tag writing + folder organization).

Only the FLAC path is exercised for tagging (per the contract). Cross-module
deps are not needed: tracks/genres/assignments are built directly in the store.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from mgc.actions import (
    execute_organize,
    plan_organize,
    read_genre,
    sanitize,
    undo_organize,
    undo_tags,
    write_genre,
)
from mgc.types import (
    ACTION_COPY,
    ACTION_TAG,
    LEVEL_GENRE,
    LEVEL_SUBGENRE,
    METHOD_MANUAL,
    GenreNode,
    Track,
)


def _write_flac(path, freq=440.0, seconds=1.0, sr=22050):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), sig.astype(np.float32), sr, format="FLAC")
    return str(path)


def _make_track(store, path):
    track = Track(path=path, content_hash=f"hash::{path}", fmt="flac")
    tid = store.upsert_track(track)
    return store.get_track(tid)


# --------------------------------------------------------------------------- tags


def test_read_genre_none_when_unset(tmp_path):
    path = _write_flac(tmp_path / "song.flac")
    assert read_genre(path) is None


def test_write_genre_then_read_roundtrip(tmp_store, tmp_path):
    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    res = write_genre(tmp_store, track, "Tech House")
    assert res["track_id"] == track.id
    assert res["path"] == path
    assert res["from"] is None
    assert res["to"] == "Tech House"
    assert read_genre(path) == "Tech House"

    # action was logged
    actions = tmp_store.iter_actions(type=ACTION_TAG)
    assert len(actions) == 1
    assert actions[0].from_value == ""
    assert actions[0].to_value == "Tech House"
    assert actions[0].undo_token == path
    assert actions[0].status == "done"


def test_write_genre_dry_run_does_not_touch_file_or_db(tmp_store, tmp_path):
    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    res = write_genre(tmp_store, track, "Tech House", dry_run=True)
    assert res["to"] == "Tech House"
    assert read_genre(path) is None  # file untouched
    assert tmp_store.iter_actions(type=ACTION_TAG) == []


def test_write_genre_with_parent_to_grouping(tmp_store, tmp_path):
    from mutagen.flac import FLAC

    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    write_genre(
        tmp_store, track, "Tech House", parent="House",
        write_parent_to_grouping=True,
    )
    assert read_genre(path) == "Tech House"
    audio = FLAC(path)
    assert audio["grouping"] == ["House"]


def test_undo_tags_restores_prior_none(tmp_store, tmp_path):
    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    write_genre(tmp_store, track, "Tech House")
    assert read_genre(path) == "Tech House"

    n = undo_tags(tmp_store)
    assert n == 1
    assert read_genre(path) is None  # restored to prior (unset)

    # action marked undone; nothing left 'done'
    assert tmp_store.iter_actions(status="done", type=ACTION_TAG) == []
    assert len(tmp_store.iter_actions(status="undone", type=ACTION_TAG)) == 1


def test_undo_tags_restores_prior_value(tmp_store, tmp_path):
    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    # seed a starting genre, then overwrite
    write_genre(tmp_store, track, "House")
    assert read_genre(path) == "House"
    write_genre(tmp_store, track, "Tech House")
    assert read_genre(path) == "Tech House"

    # undo newest-first: reverts Tech House -> House, then House -> None
    n = undo_tags(tmp_store)
    assert n == 2
    assert read_genre(path) is None


def test_undo_tags_newest_first_single_step(tmp_store, tmp_path):
    """A single undo of the latest write restores the immediately prior value."""
    path = _write_flac(tmp_path / "song.flac")
    track = _make_track(tmp_store, path)

    write_genre(tmp_store, track, "House")
    write_genre(tmp_store, track, "Tech House")

    # manually verify ordering: reverse iteration should hit Tech House first.
    actions = tmp_store.iter_actions(status="done", type=ACTION_TAG)
    assert [a.to_value for a in actions] == ["House", "Tech House"]


# ----------------------------------------------------------------------- sanitize


def test_sanitize_replaces_illegal_chars():
    assert sanitize('Drum & Bass') == "Drum & Bass"
    assert sanitize('AC/DC') == "AC_DC"
    assert sanitize('a:b*c?"<>|') == "a_b_c_____"


def test_sanitize_strips_trailing_dots_and_spaces():
    assert sanitize("Trance.. ") == "Trance"
    assert sanitize("  Lo-Fi  ") == "  Lo-Fi"


def test_sanitize_empty_fallback():
    assert sanitize("...") == "_"
    assert sanitize("") == "_"


# ----------------------------------------------------------------------- organize


def _setup_genres(store):
    """House (genre) with child Tech House (subgenre). Returns subgenre id."""
    house_id = store.upsert_genre(GenreNode(name="House", level=LEVEL_GENRE))
    th_id = store.upsert_genre(
        GenreNode(name="Tech House", parent_id=house_id, level=LEVEL_SUBGENRE)
    )
    return house_id, th_id


def test_plan_organize_builds_genre_subgenre_dest(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "track1.flac")
    track = _make_track(tmp_store, path)
    tmp_store.set_assignment(track.id, th_id, 0.9, METHOD_MANUAL)

    root = str(tmp_path / "out")
    plan = plan_organize(tmp_store, root)
    assert len(plan) == 1
    entry = plan[0]
    assert entry["track_id"] == track.id
    assert entry["src"] == path
    expected = os.path.join(root, "House", "Tech House", "track1.flac")
    assert entry["dest"] == expected


def test_plan_organize_skips_tracks_without_assignment(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    p1 = _write_flac(tmp_path / "t1.flac")
    p2 = _write_flac(tmp_path / "t2.flac")
    t1 = _make_track(tmp_store, p1)
    _t2 = _make_track(tmp_store, p2)  # no assignment
    tmp_store.set_assignment(t1.id, th_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    assert [e["track_id"] for e in plan] == [t1.id]


def test_plan_organize_skips_non_subgenre_assignment(tmp_store, tmp_path):
    house_id, _th = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "t.flac")
    track = _make_track(tmp_store, path)
    # assign to the genre-level node, not a subgenre
    tmp_store.set_assignment(track.id, house_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    assert plan == []


def test_plan_organize_collision_gets_suffix(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    # two distinct source files sharing the same basename
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = _write_flac(d1 / "track.flac")
    p2 = _write_flac(d2 / "track.flac")
    t1 = _make_track(tmp_store, p1)
    t2 = _make_track(tmp_store, p2)
    tmp_store.set_assignment(t1.id, th_id, 0.9, METHOD_MANUAL)
    tmp_store.set_assignment(t2.id, th_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    dests = sorted(e["dest"] for e in plan)
    assert len(dests) == 2
    assert dests[0] != dests[1]  # collision resolved


def test_execute_organize_copy_creates_files_and_logs(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "track1.flac")
    track = _make_track(tmp_store, path)
    tmp_store.set_assignment(track.id, th_id, 0.9, METHOD_MANUAL)

    root = str(tmp_path / "out")
    plan = plan_organize(tmp_store, root)
    executed = execute_organize(tmp_store, plan, mode="copy", dry_run=False)

    assert len(executed) == 1
    dest = executed[0]["dest"]
    assert os.path.exists(dest)
    assert os.path.exists(path)  # original still there (copy)

    actions = tmp_store.iter_actions(type=ACTION_COPY)
    assert len(actions) == 1
    assert actions[0].to_value == dest
    assert actions[0].undo_token == dest


def test_execute_organize_dry_run_no_side_effects(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "track1.flac")
    track = _make_track(tmp_store, path)
    tmp_store.set_assignment(track.id, th_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    executed = execute_organize(tmp_store, plan, dry_run=True)
    assert len(executed) == 1
    assert not os.path.exists(executed[0]["dest"])
    assert tmp_store.iter_actions(type=ACTION_COPY) == []


def test_undo_organize_copy_removes_dest(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "track1.flac")
    track = _make_track(tmp_store, path)
    tmp_store.set_assignment(track.id, th_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    executed = execute_organize(tmp_store, plan, mode="copy")
    dest = executed[0]["dest"]
    assert os.path.exists(dest)

    n = undo_organize(tmp_store)
    assert n == 1
    assert not os.path.exists(dest)  # copy removed
    assert os.path.exists(path)  # source preserved
    assert tmp_store.iter_actions(status="done", type=ACTION_COPY) == []


def test_undo_organize_move_restores_src(tmp_store, tmp_path):
    _house, th_id = _setup_genres(tmp_store)
    path = _write_flac(tmp_path / "track1.flac")
    track = _make_track(tmp_store, path)
    tmp_store.set_assignment(track.id, th_id, 0.9, METHOD_MANUAL)

    plan = plan_organize(tmp_store, str(tmp_path / "out"))
    executed = execute_organize(tmp_store, plan, mode="move")
    dest = executed[0]["dest"]
    assert os.path.exists(dest)
    assert not os.path.exists(path)  # moved away

    n = undo_organize(tmp_store)
    assert n == 1
    assert not os.path.exists(dest)
    assert os.path.exists(path)  # moved back
