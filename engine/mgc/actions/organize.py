"""Folder organization: plan and execute a genre/subgenre tree, with undo.

The destination layout is ``root/<parent genre>/<subgenre>/<filename>``. Names
are sanitized for Windows. Copy/move operations are logged so they can be
reversed (copies are deleted; moves are moved back).
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from mgc.types import ACTION_COPY, ACTION_MOVE, LEVEL_SUBGENRE

_ILLEGAL = '\\/:*?"<>|'
_ILLEGAL_TABLE = {ord(c): "_" for c in _ILLEGAL}


def sanitize(name: str) -> str:
    """Return ``name`` with Windows-illegal chars replaced and trailing junk stripped.

    Illegal characters ``\\ / : * ? " < > |`` become underscores; trailing dots
    and spaces (which Windows disallows) are removed. Empty results fall back to
    ``"_"`` so a usable path component is always produced.
    """
    cleaned = name.translate(_ILLEGAL_TABLE)
    cleaned = cleaned.rstrip(" .")
    return cleaned or "_"


def _unique_dest(dest: str, taken: set) -> str:
    """Return ``dest`` or a numerically-suffixed variant not in ``taken`` or on disk."""
    if dest not in taken and not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(dest)
    i = 1
    candidate = f"{base} ({i}){ext}"
    while candidate in taken or os.path.exists(candidate):
        i += 1
        candidate = f"{base} ({i}){ext}"
    return candidate


def plan_organize(store, root: str) -> list[dict]:
    """Plan destination paths for every track with a subgenre assignment.

    For each assignment whose ``genre_id`` resolves to a subgenre node, the
    destination is ``root/<parent genre>/<subgenre>/<filename>``. Tracks lacking
    an assignment (or whose genre is missing/not a subgenre) are skipped.
    Collisions within the plan get a numeric suffix.
    """
    plan: list[dict] = []
    taken: set = set()
    for row in store.iter_assignments():
        genre_id = row["genre_id"]
        if genre_id is None:
            continue
        genre = store.get_genre(genre_id)
        if genre is None or genre.level != LEVEL_SUBGENRE:
            continue
        track = store.get_track(row["track_id"])
        if track is None:
            continue

        parent = store.get_genre(genre.parent_id) if genre.parent_id is not None else None
        parent_name = parent.name if parent is not None else "Unknown"
        filename = os.path.basename(track.path)
        dest = os.path.join(
            root,
            sanitize(parent_name),
            sanitize(genre.name),
            filename,
        )
        dest = _unique_dest(dest, taken)
        taken.add(dest)
        plan.append({"track_id": track.id, "src": track.path, "dest": dest})
    return plan


def execute_organize(
    store,
    plan: list[dict],
    mode: str = "copy",
    dry_run: bool = False,
) -> list[dict]:
    """Copy or move each planned file, creating directories and logging actions.

    ``mode`` is ``"copy"`` or ``"move"``. On a destination collision a numeric
    suffix is appended. Each executed entry is logged (ACTION_COPY/ACTION_MOVE)
    with ``undo_token=dest``. When ``dry_run`` nothing touches disk or the DB.
    Returns the executed entries (with any resolved dest).
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"mode must be 'copy' or 'move', got {mode!r}")

    action_type = ACTION_COPY if mode == "copy" else ACTION_MOVE
    executed: list[dict] = []
    used: set = set()
    for entry in plan:
        src = entry["src"]
        dest = _unique_dest(entry["dest"], used)
        used.add(dest)
        out = {"track_id": entry.get("track_id"), "src": src, "dest": dest}

        if dry_run:
            executed.append(out)
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if mode == "copy":
            shutil.copy2(src, dest)
        else:
            shutil.move(src, dest)
        store.log_action(
            action_type,
            out["track_id"],
            from_value=src,
            to_value=dest,
            undo_token=dest,
        )
        executed.append(out)
    return executed


def undo_organize(store) -> int:
    """Reverse 'done' copy/move actions, newest-first.

    Copies have their destination deleted; moves are moved back to ``from_value``
    (the original source). Each reversed action is marked 'undone'. Returns the
    number of actions reverted.
    """
    actions = [
        a
        for a in store.iter_actions(status="done")
        if a.type in (ACTION_COPY, ACTION_MOVE)
    ]
    count = 0
    for action in reversed(actions):  # newest-first
        dest = action.undo_token
        try:
            if action.type == ACTION_COPY:
                if dest and os.path.exists(dest):
                    os.remove(dest)
            else:  # move: put it back
                src = action.from_value
                if dest and src and os.path.exists(dest):
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dest, src)
        except Exception:
            pass
        store.set_action_status(action.id, "undone")
        count += 1
    return count
