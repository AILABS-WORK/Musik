"""Generate the bundled ``genres.json`` (MusicBrainz vocabulary + a curated
electronic-genre edge seed).

Run once (dev side) to refresh the data:  python -m mgc.metadata.generate

The full ~2,150-genre vocabulary comes from the MusicBrainz API (genre/all, CC0).
The genre-genre relationship edges are not reliably exposed by /ws/2, so we ship a
hand-curated subgenre/fusion seed for the electronic genres this app is built for;
the graph still carries the complete vocabulary for naming/matching.
"""

from __future__ import annotations

import json
from pathlib import Path

# parent genre -> its subgenres (electronic focus). Encoded as 'subgenre' edges
# (from=child, to=parent). Hand-curated; the heart of the "related genres" expander.
_SUBGENRES = {
    "house": ["deep house", "tech house", "progressive house", "electro house",
              "acid house", "future house", "bass house", "tribal house",
              "afro house", "soulful house", "funky house", "jackin house",
              "microhouse", "minimal house", "garage house", "disco house",
              "tropical house", "melodic house", "deep tech", "organic house"],
    "techno": ["minimal techno", "detroit techno", "dub techno", "melodic techno",
               "acid techno", "hard techno", "industrial techno", "ambient techno",
               "peak time techno", "hypnotic techno", "raw techno"],
    "trance": ["progressive trance", "psytrance", "uplifting trance", "vocal trance",
               "tech trance", "goa trance", "hard trance", "acid trance"],
    "drum and bass": ["liquid funk", "neurofunk", "jump up", "techstep", "jungle",
                      "drumfunk", "halftime"],
    "dubstep": ["brostep", "riddim", "future garage", "deep dubstep"],
    "electro": ["electroclash", "electropop", "electro house"],
    "ambient": ["dark ambient", "drone", "downtempo", "ambient techno", "space ambient"],
    "breakbeat": ["big beat", "nu skool breaks", "breaks"],
    "garage": ["uk garage", "2-step", "future garage", "speed garage", "bassline"],
    "hardcore": ["gabber", "happy hardcore", "uk hardcore", "frenchcore"],
    "idm": ["glitch", "braindance"],
    "disco": ["nu disco", "italo disco", "disco house"],
}
# a few cross-genre fusion/influence edges
_FUSION = [
    {"from": "tech house", "to": "techno", "rel": "fusion"},
    {"from": "melodic house", "to": "melodic techno", "rel": "fusion"},
    {"from": "afro house", "to": "house", "rel": "subgenre"},
    {"from": "dub techno", "to": "dub", "rel": "influence"},
    {"from": "liquid funk", "to": "jazz", "rel": "influence"},
    {"from": "nu disco", "to": "house", "rel": "influence"},
]


def curated_edges() -> list[dict]:
    edges: list[dict] = []
    for parent, kids in _SUBGENRES.items():
        for kid in kids:
            edges.append({"from": kid, "to": parent, "rel": "subgenre"})
    edges.extend(_FUSION)
    return edges


def generate_genre_graph(out_path: Path | str | None = None, get=None,
                         version: str = "seed") -> dict:
    """Fetch the MB genre vocabulary, merge the curated edges, write genres.json."""
    if get is None:
        from mgc.metadata.musicbrainz import _get as get

    names: list[str] = []
    for page in range(40):
        data = get(f"https://musicbrainz.org/ws/2/genre/all?fmt=json&limit=100&offset={page * 100}")
        chunk = data.get("genres") or []
        names.extend(g["name"] for g in chunk if g.get("name"))
        if len(chunk) < 100:
            break

    edges = curated_edges()
    # make sure every genre named in an edge is in the vocabulary
    vocab = set(n.lower() for n in names)
    for e in edges:
        for k in ("from", "to"):
            if e[k].lower() not in vocab:
                names.append(e[k])
                vocab.add(e[k].lower())

    payload = {"version": version, "genres": sorted(set(names)), "edges": edges}
    out = Path(out_path) if out_path else Path(__file__).with_name("genres.json")
    out.write_text(json.dumps(payload), encoding="utf-8")
    return {"genres": len(payload["genres"]), "edges": len(edges), "path": str(out)}


if __name__ == "__main__":
    print(generate_genre_graph())
