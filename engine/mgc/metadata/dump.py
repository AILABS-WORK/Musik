"""Build the COMPLETE genre graph from a MusicBrainz database dump.

The MusicBrainz web API does not expose genre-genre relationships, but the full
dump does (the ``l_genre_genre`` table). Since this is a personal, non-commercial
tool, the cheapest way to get every real subgenre/fusion/influence edge is:

  1. download the core dump once (CC0):
       https://data.metabrainz.org/pub/musicbrainz/data/fullexport/LATEST/mbdump.tar.bz2
     (~6 GB; the genre tables inside it are tiny)
  2. run:  python -m mgc.metadata.dump  /path/to/mbdump.tar.bz2
     -> rewrites engine/mgc/metadata/genres.json with the full vocabulary + edges.

You don't need Postgres; we stream four small TSV members straight out of the tar.
``build_graph`` is pure (works on parsed rows) so it's unit-tested without a dump.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

# Column indices in the MusicBrainz COPY/TSV exports (no header rows).
_GENRE_ID, _GENRE_NAME = 0, 2                       # genre: id, gid, name, ...
_LT_ID, _LT_E0, _LT_E1, _LT_NAME = 0, 4, 5, 6       # link_type: id,parent,child_order,gid,e_type0,e_type1,name,...
_LINK_ID, _LINK_TYPE = 0, 1                         # link: id, link_type, ...
_LGG_LINK, _LGG_E0, _LGG_E1 = 1, 2, 3               # l_genre_genre: id, link, entity0, entity1, ...


def _rel_name(raw: str) -> str:
    n = (raw or "").lower()
    if "subgenre" in n:
        return "subgenre"
    if "fusion" in n:
        return "fusion"
    if "influen" in n:
        return "influence"
    return n or "related"


def build_graph(genre_rows, link_type_rows, link_rows, lgg_rows) -> dict:
    """Pure graph builder from parsed TSV rows -> {genres:[names], edges:[{from,to,rel}]}.

    For a 'subgenre' edge, entity0 is the subgenre and entity1 the umbrella, so the
    edge points from=child -> to=parent (matching genre_graph.py's convention).
    """
    names: dict[str, str] = {}
    for r in genre_rows:
        if len(r) > _GENRE_NAME:
            names[r[_GENRE_ID]] = r[_GENRE_NAME]

    gg_types: dict[str, str] = {}
    for r in link_type_rows:
        if len(r) > _LT_NAME and r[_LT_E0] == "genre" and r[_LT_E1] == "genre":
            gg_types[r[_LT_ID]] = _rel_name(r[_LT_NAME])

    link_rel: dict[str, str] = {}
    for r in link_rows:
        if len(r) > _LINK_TYPE and r[_LINK_TYPE] in gg_types:
            link_rel[r[_LINK_ID]] = gg_types[r[_LINK_TYPE]]

    edges = []
    for r in lgg_rows:
        if len(r) > _LGG_E1 and r[_LGG_LINK] in link_rel:
            child = names.get(r[_LGG_E0])
            parent = names.get(r[_LGG_E1])
            if child and parent:
                edges.append({"from": child, "to": parent, "rel": link_rel[r[_LGG_LINK]]})

    return {"genres": sorted(set(names.values())), "edges": edges}


def _parse_tsv(fileobj):
    for raw in fileobj:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        yield line.rstrip("\n").split("\t")


def build_from_dump(tar_path: str, out_path: str | Path | None = None,
                    merge_curated: bool = True) -> dict:
    """Stream the four genre tables out of mbdump.tar.bz2 and write genres.json."""
    want = ["genre", "link_type", "link", "l_genre_genre"]
    rows: dict[str, list] = {k: [] for k in want}
    with tarfile.open(tar_path, "r:bz2") as tf:
        for k in want:
            try:
                member = tf.getmember(f"mbdump/{k}")
            except KeyError:
                continue
            f = tf.extractfile(member)
            if f is not None:
                rows[k] = list(_parse_tsv(f))

    graph = build_graph(rows["genre"], rows["link_type"], rows["link"], rows["l_genre_genre"])

    if merge_curated:
        from mgc.metadata.generate import curated_edges
        have = {(e["from"].lower(), e["to"].lower()) for e in graph["edges"]}
        vocab = {g.lower() for g in graph["genres"]}
        for e in curated_edges():
            if (e["from"].lower(), e["to"].lower()) not in have and e["from"].lower() in vocab:
                graph["edges"].append(e)

    payload = {"version": "dump", "genres": graph["genres"], "edges": graph["edges"]}
    out = Path(out_path) if out_path else Path(__file__).with_name("genres.json")
    out.write_text(json.dumps(payload), encoding="utf-8")
    return {"genres": len(payload["genres"]), "edges": len(payload["edges"]), "path": str(out)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m mgc.metadata.dump /path/to/mbdump.tar.bz2")
        raise SystemExit(2)
    print(build_from_dump(sys.argv[1]))
