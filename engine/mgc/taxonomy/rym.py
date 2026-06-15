"""Parse the RateYourMusic JSON export into a flat list of GenreNode.

The export (under ``references/``) is shaped as:

* ``_index.json``         -> ``{"genres": [{name, description, url}, ...]}``
                             the ~49 broad top-level buckets.
* ``main/<slug>.json``    -> ``{name, ..., "sub_genres": [{name, description,
                             level:"sub", parent}, ...]}`` direct children of a
                             top genre.
* ``detailed/<slug>.json``-> ``{name, parent, "children": [{name, description,
                             level:"sub-2", parent}, ...]}`` deeper subgenres.

We map the top index entries to ``LEVEL_GENRE`` and everything deeper
(``sub`` / ``sub-2``) to ``LEVEL_SUBGENRE``. Parent linkage is carried by
name only here (``GenreNode.parent_id`` is resolved later, during seeding,
once parents have row ids).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mgc.types import GenreNode, LEVEL_GENRE, LEVEL_SUBGENRE

# Parent name is stashed on the node via this attribute so ``seed_taxonomy``
# can resolve it to a row id without re-reading files. GenreNode itself only
# carries ``parent_id`` (a resolved int), so we attach a transient string.
_PARENT_NAME_ATTR = "_parent_name"


def _node(name: str, level: str, description: Optional[str],
          parent_name: Optional[str]) -> GenreNode:
    """Build a GenreNode, stashing the (unresolved) parent name on it."""
    desc = description.strip() if isinstance(description, str) and description.strip() else None
    n = GenreNode(name=name.strip(), level=level, source="seed", description=desc)
    setattr(n, _PARENT_NAME_ATTR, parent_name)
    return n


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_rym(refs_dir: str) -> list[GenreNode]:
    """Parse the RYM export at ``refs_dir`` into ordered GenreNode list.

    Top genres first (from ``_index.json``), then their ``sub`` children
    (from ``main/``), then ``sub-2`` children (from ``detailed/``). Duplicate
    names are de-duplicated, keeping the shallowest (first-seen) occurrence so
    a name that appears both as a top genre and as someone's child stays a top
    genre. Order guarantees parents precede children for resolution.
    """
    root = Path(refs_dir)
    nodes: list[GenreNode] = []
    seen: set[str] = set()

    def add(node: GenreNode) -> None:
        key = node.name.lower()
        if key in seen:
            return
        seen.add(key)
        nodes.append(node)

    # 1) Top-level buckets from _index.json -> LEVEL_GENRE (no parent).
    index = _read_json(root / "_index.json") or {}
    for entry in index.get("genres", []):
        name = entry.get("name")
        if not name:
            continue
        add(_node(name, LEVEL_GENRE, entry.get("description"), None))

    # 2) Direct children from main/*.json -> LEVEL_SUBGENRE (parent = file genre).
    main_dir = root / "main"
    if main_dir.is_dir():
        for fp in sorted(main_dir.glob("*.json")):
            data = _read_json(fp)
            if not data:
                continue
            parent_name = data.get("name")
            for sub in data.get("sub_genres", []):
                name = sub.get("name")
                if not name:
                    continue
                add(_node(name, LEVEL_SUBGENRE, sub.get("description"),
                          sub.get("parent") or parent_name))

    # 3) Deeper children from detailed/*.json -> LEVEL_SUBGENRE (parent = its node).
    detailed_dir = root / "detailed"
    if detailed_dir.is_dir():
        for fp in sorted(detailed_dir.glob("*.json")):
            data = _read_json(fp)
            if not data:
                continue
            parent_name = data.get("name")
            for child in data.get("children", []):
                name = child.get("name")
                if not name:
                    continue
                add(_node(name, LEVEL_SUBGENRE, child.get("description"),
                          child.get("parent") or parent_name))

    return nodes


def seed_taxonomy(store, refs_dir: str, limit: Optional[int] = None) -> int:
    """Upsert the parsed RYM taxonomy into ``store``; return count seeded.

    Parent ids are resolved by name as we go (nodes are ordered parents-first).
    Idempotent: before inserting we look up an existing ``(name, parent_id)``
    row and reuse it. This guard is required because SQLite ``UNIQUE`` treats
    ``NULL`` parent_ids as distinct, so the store's ``ON CONFLICT(name,
    parent_id)`` upsert does not dedupe top-level genres on its own. ``limit``
    caps how many nodes are seeded (useful for tests / smoke runs).
    """
    nodes = parse_rym(refs_dir)
    if limit is not None:
        nodes = nodes[:limit]

    # name (lower) -> resolved row id, for parent resolution.
    name_to_id: dict[str, int] = {}
    seeded = 0
    for node in nodes:
        parent_name = getattr(node, _PARENT_NAME_ATTR, None)
        parent_id: Optional[int] = None
        if parent_name:
            parent_id = name_to_id.get(parent_name.strip().lower())
        node.parent_id = parent_id

        existing = store.get_genre_by_name(node.name, parent_id)
        if existing is not None and existing.id is not None:
            gid = existing.id
        else:
            gid = store.upsert_genre(node)
        name_to_id.setdefault(node.name.strip().lower(), gid)
        seeded += 1
    return seeded
