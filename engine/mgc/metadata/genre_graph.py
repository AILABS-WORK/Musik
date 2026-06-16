"""The MusicBrainz genre graph: the full CC0 genre vocabulary plus subgenre /
fusion / influence edges, used for a real taxonomy and a "closely related genres"
expander (so grouping can reach a stylistic cousin that pure audio distance misses).

Loads a bundled ``genres.json`` (generated once from the MusicBrainz API + a
curated electronic-genre edge seed; see generate.py). Pure data + graph traversal,
no network at runtime. Edges are keyed by lowercase genre name so they line up
with whatever genre strings our own tracks carry.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

_GRAPH_PATH = Path(__file__).with_name("genres.json")

# relationship type -> weight when expanding "related"
_REL_WEIGHT = {"subgenre": 1.0, "fusion": 0.7, "influence": 0.5}


class GenreGraph:
    """Vocabulary + edges with parent/child/related queries (case-insensitive)."""

    def __init__(self, genres: list[str] | None = None, edges: list[dict] | None = None):
        self.genres = list(genres or [])
        self._known = {g.lower() for g in self.genres}
        # adjacency: name -> list[(neighbor, rel, direction)] where direction is
        # 'up' (neighbor is the parent/umbrella) or 'down' (neighbor is the child).
        self._adj: dict[str, list[tuple]] = {}
        for e in edges or []:
            frm = (e.get("from") or "").strip().lower()
            to = (e.get("to") or "").strip().lower()
            rel = e.get("rel") or "subgenre"
            if not frm or not to:
                continue
            self._adj.setdefault(frm, []).append((to, rel, "up"))
            self._adj.setdefault(to, []).append((frm, rel, "down"))

    @classmethod
    def load(cls, path: Path | str | None = None) -> "GenreGraph":
        p = Path(path) if path else _GRAPH_PATH
        if not p.exists():
            return cls([], [])
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls([], [])
        return cls(data.get("genres") or [], data.get("edges") or [])

    def has(self, name: str) -> bool:
        return (name or "").strip().lower() in self._known

    def parents(self, name: str) -> list[str]:
        return [n for n, rel, d in self._adj.get((name or "").lower(), []) if rel == "subgenre" and d == "up"]

    def children(self, name: str) -> list[str]:
        return [n for n, rel, d in self._adj.get((name or "").lower(), []) if rel == "subgenre" and d == "down"]

    def related(self, name: str, max_depth: int = 2, limit: int = 25) -> list[dict]:
        """Closely-related genres via weighted BFS over the edges.

        Weight = relationship weight (subgenre 1.0, fusion 0.7, influence 0.5)
        decayed by 0.5 per extra hop. Returns ``[{genre, weight}]`` sorted desc,
        excluding the seed. Empty if the seed isn't in the graph.
        """
        seed = (name or "").strip().lower()
        if seed not in self._adj:
            return []
        best: dict[str, float] = {}
        q: deque = deque([(seed, 0, 1.0)])
        seen = {seed}
        while q:
            node, depth, decay = q.popleft()
            if depth >= max_depth:
                continue
            for neigh, rel, _direction in self._adj.get(node, []):
                w = _REL_WEIGHT.get(rel, 0.4) * decay
                if w > best.get(neigh, 0.0):
                    best[neigh] = w
                if neigh not in seen:
                    seen.add(neigh)
                    q.append((neigh, depth + 1, decay * 0.5))
        best.pop(seed, None)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:limit]
        return [{"genre": g, "weight": round(w, 3)} for g, w in ranked]


_DEFAULT: GenreGraph | None = None


def get_graph() -> GenreGraph:
    """Process-wide cached graph loaded from the bundled genres.json."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GenreGraph.load()
    return _DEFAULT
