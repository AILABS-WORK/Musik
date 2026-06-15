"""Actions module: tag writing and folder organization with undo support.

All side-effecting operations are logged to the store's actions_log so they can
be reversed. ``tags`` writes the chosen subgenre into the file's genre field
(Rekordbox-readable); ``organize`` materializes a genre/subgenre folder tree.
"""

from __future__ import annotations

from mgc.actions.organize import (
    execute_organize,
    plan_organize,
    sanitize,
    undo_organize,
)
from mgc.actions.tags import read_genre, undo_tags, write_genre

__all__ = [
    "read_genre",
    "write_genre",
    "undo_tags",
    "sanitize",
    "plan_organize",
    "execute_organize",
    "undo_organize",
]
