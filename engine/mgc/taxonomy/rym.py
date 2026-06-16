"""Parse the RateYourMusic JSON export into a flat list of GenreNode.

The export (under ``references/``) is shaped as:

* ``_index.json``         -> ``{"genres": [{name, description, url}, ...]}``
                             the ~49 broad top-level buckets.
* ``main/<slug>.json``    -> ``{name, ..., "sub_genres": [{name, description,
                             level:"sub", parent}, ...]}`` direct children of a
                             top genre.
* ``detailed/<slug>.json``-> ``{name, parent, level, "children": [{name,
                             description, level:"sub-2".."sub-6", parent}, ...]}``
                             the deeper sub-genre tiers. Every detailed file
                             describes one node (``name`` + its own ``parent`` /
                             ``level``) *and* its direct ``children``.

RateYourMusic's taxonomy is a DAG, not a clean tree: the same genre name shows
up under several parents (e.g. a fusion style listed beneath each ancestor).
Earlier versions of this module de-duplicated purely by name, which collapsed
all those re-parentings into a single row and only reached ~2,600 nodes.

We now ingest *every* tier (``main`` + all of ``detailed``), build the full
parent->child adjacency by name, and materialise the DAG as a tree: one
``GenreNode`` per distinct root-to-node *path*. A style reachable through three
different ancestors becomes three rows (each with its own ``parent_id``), which
is exactly what the store's ``UNIQUE(name, parent_id)`` schema expresses and
recovers the full ~5,900-node taxonomy.

Level is assigned by depth: the top buckets are ``LEVEL_GENRE`` and everything
below them is ``LEVEL_SUBGENRE``. Parent linkage is carried by *path* here
(``GenreNode.parent_id`` is resolved later, during seeding, once each path has a
row id).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from mgc.types import GenreNode, LEVEL_GENRE, LEVEL_SUBGENRE

# The (unresolved) parent path is stashed on each node via this attribute so
# ``seed_taxonomy`` can resolve it to a row id without re-reading files.
# GenreNode itself only carries ``parent_id`` (a resolved int), so we attach a
# transient tuple-of-names path. ``_PATH_ATTR`` is the node's own full path
# (used as the dict key when resolving children) and ``_PARENT_PATH_ATTR`` is
# its parent's path (``None`` for top genres).
_PATH_ATTR = "_path"
_PARENT_PATH_ATTR = "_parent_path"

# Guard against pathological depth (cycles are already broken by the on-path
# visited set, but this bounds fan-out of any unexpectedly deep chain).
_MAX_DEPTH = 16


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clean_desc(description) -> Optional[str]:
    if isinstance(description, str) and description.strip():
        return description.strip()
    return None


def _build_index(refs_dir: str) -> tuple[
    list[str],                       # ordered root genre names
    dict[str, set[str]],             # parent_lower -> {child_lower}
    dict[str, str],                  # name_lower -> display name
    dict[str, Optional[str]],        # name_lower -> description
]:
    """Read all tiers and return the adjacency + display/description lookups.

    Adjacency is keyed by lowercased names; ordering of roots and the per-parent
    child sets is resolved deterministically (insertion order for roots, sorted
    for children) at walk time.
    """
    root = Path(refs_dir)

    roots: list[str] = []
    roots_seen: set[str] = set()
    children_of: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    descriptions: dict[str, Optional[str]] = {}

    def note(name: str, description=None) -> str:
        key = name.strip().lower()
        display.setdefault(key, name.strip())
        desc = _clean_desc(description)
        # First non-empty description wins (shallower tiers are read first).
        if desc and not descriptions.get(key):
            descriptions[key] = desc
        descriptions.setdefault(key, None)
        return key

    def link(parent: Optional[str], child: str) -> None:
        if not parent:
            return
        pk = parent.strip().lower()
        ck = child.strip().lower()
        if pk != ck:  # never self-parent
            children_of[pk].add(ck)

    # 1) Top-level buckets from _index.json -> roots (LEVEL_GENRE, no parent).
    index = _read_json(root / "_index.json") or {}
    for entry in index.get("genres", []):
        name = entry.get("name")
        if not name:
            continue
        key = note(name, entry.get("description"))
        if key not in roots_seen:
            roots_seen.add(key)
            roots.append(key)

    # 2) Direct children from main/*.json.
    main_dir = root / "main"
    if main_dir.is_dir():
        for fp in sorted(main_dir.glob("*.json")):
            data = _read_json(fp)
            if not data:
                continue
            file_name = data.get("name")
            if file_name:
                note(file_name, data.get("description"))
            for sub in data.get("sub_genres", []):
                name = sub.get("name")
                if not name:
                    continue
                note(name, sub.get("description"))
                link(sub.get("parent") or file_name, name)

    # 3) Deeper children from detailed/*.json. Each file describes one node and
    #    its direct children; we register both the node's own parent edge and
    #    every child edge so the full DAG is captured.
    detailed_dir = root / "detailed"
    if detailed_dir.is_dir():
        for fp in sorted(detailed_dir.glob("*.json")):
            data = _read_json(fp)
            if not data:
                continue
            file_name = data.get("name")
            if file_name:
                note(file_name, data.get("description"))
                link(data.get("parent"), file_name)
            for child in data.get("children", []):
                name = child.get("name")
                if not name:
                    continue
                note(name, child.get("description"))
                # The child's parent is the file node, but honour an explicit
                # ``parent`` too (they normally agree).
                link(child.get("parent") or file_name, name)

    return roots, children_of, display, descriptions


def parse_rym(refs_dir: str) -> list[GenreNode]:
    """Parse the RYM export at ``refs_dir`` into an ordered GenreNode list.

    The DAG of genre names is materialised as a tree: one node per distinct
    root-to-node path. Output order is breadth-first by depth -- every top genre
    first, then every depth-1 sub-genre, and so on. This guarantees a node's
    parent precedes it (required for id resolution during seeding) *and* keeps
    all top buckets at the front (so ``limit`` caps cleanly at a tier boundary).
    Each node stashes its own path and its parent's path so seeding can wire
    ``parent_id`` without name ambiguity.

    Cycles (a name reachable from itself) are broken by skipping any child
    already present on the current path; depth is also bounded by ``_MAX_DEPTH``.
    """
    roots, children_of, display, descriptions = _build_index(refs_dir)

    nodes: list[GenreNode] = []

    def make_node(key: str, path: tuple[str, ...], depth: int,
                  parent_path: Optional[tuple[str, ...]]) -> GenreNode:
        node = GenreNode(
            name=display.get(key, key),
            level=LEVEL_GENRE if depth == 0 else LEVEL_SUBGENRE,
            source="seed",
            description=descriptions.get(key),
        )
        setattr(node, _PATH_ATTR, path)
        setattr(node, _PARENT_PATH_ATTR, parent_path)
        return node

    # BFS frontier of (key, path, depth, parent_path), level by level.
    frontier = [(r, (r,), 0, None) for r in roots]
    while frontier:
        next_frontier = []
        for key, path, depth, parent_path in frontier:
            nodes.append(make_node(key, path, depth, parent_path))
            if depth >= _MAX_DEPTH:
                continue
            on_path = set(path)
            for child in sorted(children_of.get(key, ())):
                if child in on_path:
                    continue  # cycle guard: don't revisit an ancestor
                next_frontier.append((child, path + (child,), depth + 1, path))
        frontier = next_frontier

    return nodes


def seed_taxonomy(store, refs_dir: str, limit: Optional[int] = None) -> int:
    """Upsert the parsed RYM taxonomy into ``store``; return count seeded.

    Nodes are emitted parents-first (pre-order), so each parent path already has
    a resolved row id by the time its children are processed. Parent ids are
    resolved by *path* (not by bare name) so the same genre under different
    ancestors maps to the correct row.

    Idempotent: before inserting we look up the existing ``(name, parent_id)``
    row and reuse it. This guard is required because SQLite ``UNIQUE`` treats
    ``NULL`` parent_ids as distinct, so the store's ``ON CONFLICT(name,
    parent_id)`` upsert would not dedupe top-level genres on its own. ``limit``
    caps how many nodes are seeded (useful for tests / smoke runs).
    """
    nodes = parse_rym(refs_dir)
    if limit is not None:
        nodes = nodes[:limit]

    # Full path (tuple of lowercased names) -> resolved row id.
    path_to_id: dict[tuple[str, ...], int] = {}
    seeded = 0
    for node in nodes:
        parent_path = getattr(node, _PARENT_PATH_ATTR, None)
        parent_id: Optional[int] = path_to_id.get(parent_path) if parent_path else None
        node.parent_id = parent_id

        existing = store.get_genre_by_name(node.name, parent_id)
        if existing is not None and existing.id is not None:
            gid = existing.id
        else:
            gid = store.upsert_genre(node)

        path = getattr(node, _PATH_ATTR, None)
        if path is not None:
            path_to_id[path] = gid
        seeded += 1
    return seeded
