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
    "electronic": ["house", "techno", "trance", "drum and bass", "dubstep",
                   "breakbeat", "garage", "ambient", "idm", "electro", "hardcore",
                   "hardstyle", "downtempo", "synthwave", "bass music", "uk garage"],
    "house": ["deep house", "tech house", "progressive house", "electro house",
              "acid house", "future house", "bass house", "tribal house",
              "afro house", "soulful house", "funky house", "jackin house",
              "microhouse", "minimal house", "garage house", "disco house",
              "tropical house", "melodic house", "deep tech", "organic house",
              "french house", "latin house", "ghetto house", "hard house",
              "big room", "slap house", "amapiano", "gqom", "hip house",
              "italo house", "lo-fi house", "outsider house", "fidget house"],
    "techno": ["minimal techno", "detroit techno", "dub techno", "melodic techno",
               "acid techno", "hard techno", "industrial techno", "ambient techno",
               "peak time techno", "hypnotic techno", "raw techno", "schranz",
               "hardgroove", "dark techno", "bleep techno", "birmingham techno"],
    "trance": ["progressive trance", "psytrance", "uplifting trance", "vocal trance",
               "tech trance", "goa trance", "hard trance", "acid trance",
               "balearic trance", "dream trance", "full on", "dark psytrance",
               "hi-tech", "forest psy"],
    "drum and bass": ["liquid funk", "neurofunk", "jump up", "techstep", "jungle",
                      "drumfunk", "halftime", "jazzstep", "darkstep", "ragga jungle",
                      "drill and bass", "minimal drum and bass", "deep drum and bass"],
    "dubstep": ["brostep", "riddim", "future garage", "deep dubstep",
                "melodic dubstep", "chillstep", "drumstep"],
    "garage": ["uk garage", "2-step", "future garage", "speed garage", "bassline"],
    "uk garage": ["2-step", "speed garage", "bassline", "grime", "breakstep"],
    "electro": ["electroclash", "electropop", "electro house", "freestyle",
                "miami bass", "electro-funk"],
    "ambient": ["dark ambient", "drone", "downtempo", "ambient techno",
                "space ambient", "ambient dub", "lowercase"],
    "breakbeat": ["big beat", "nu skool breaks", "breaks", "florida breaks",
                  "progressive breaks", "acid breaks"],
    "hardcore": ["gabber", "happy hardcore", "uk hardcore", "frenchcore",
                 "terrorcore", "speedcore", "breakcore", "makina"],
    "hardstyle": ["rawstyle", "euphoric hardstyle", "dubstyle"],
    "idm": ["glitch", "braindance", "drill and bass"],
    "disco": ["nu disco", "italo disco", "disco house", "cosmic disco",
              "space disco", "hi-nrg"],
    "synthwave": ["outrun", "darksynth", "retrowave", "dreamwave"],
    "downtempo": ["trip hop", "chillout", "lounge", "nu jazz", "balearic"],
    "bass music": ["future bass", "wave", "hybrid trap", "trap", "color bass"],
}
# cross-genre fusion / influence edges (not strict parent/child)
_FUSION = [
    {"from": "tech house", "to": "techno", "rel": "fusion"},
    {"from": "melodic house", "to": "melodic techno", "rel": "fusion"},
    {"from": "dub techno", "to": "dub", "rel": "influence"},
    {"from": "liquid funk", "to": "jazz", "rel": "influence"},
    {"from": "nu disco", "to": "house", "rel": "influence"},
    {"from": "trip hop", "to": "hip hop", "rel": "influence"},
    {"from": "future bass", "to": "trap", "rel": "influence"},
    {"from": "garage house", "to": "uk garage", "rel": "fusion"},
    {"from": "grime", "to": "dubstep", "rel": "influence"},
    {"from": "amapiano", "to": "kwaito", "rel": "influence"},
    {"from": "jungle", "to": "ragga", "rel": "influence"},
    {"from": "acid house", "to": "acid techno", "rel": "fusion"},
    {"from": "psytrance", "to": "goa trance", "rel": "fusion"},
    {"from": "synthwave", "to": "italo disco", "rel": "influence"},
    {"from": "big beat", "to": "breakbeat", "rel": "subgenre"},
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


def rebuild_edges_offline(out_path: Path | str | None = None) -> dict:
    """Re-apply the curated edges to the already-bundled vocabulary (no network).

    Keeps the genres.json vocabulary as-is and swaps in the current curated edge
    set, so expanding the seed takes effect instantly without fetching anything.
    """
    path = Path(out_path) if out_path else Path(__file__).with_name("genres.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"genres": []}
    names = list(data.get("genres") or [])
    vocab = {n.lower() for n in names}
    edges = curated_edges()
    for e in edges:
        for k in ("from", "to"):
            if e[k].lower() not in vocab:
                names.append(e[k])
                vocab.add(e[k].lower())
    payload = {"version": "seed+", "genres": sorted(set(names)), "edges": edges}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"genres": len(payload["genres"]), "edges": len(edges), "path": str(path)}


if __name__ == "__main__":
    print(generate_genre_graph())
