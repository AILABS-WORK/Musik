"""Segment-level similarity: a selected region finds the track that contains a
part that sounds like it (and where in that track the part is)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from mgc.segments import build_segment_index, embed_segment, find_similar_segments
from mgc.types import Track


def _tone(sr, secs, freq):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_segment_search_finds_the_matching_part(tmp_store, tmp_path):
    sr = 22050
    sigs = {
        "A": _tone(sr, 4, 440),                                   # all 440 Hz
        "B": _tone(sr, 4, 3000),                                  # all 3000 Hz
        "C": np.concatenate([_tone(sr, 2, 3000), _tone(sr, 2, 440)]),  # 440 in 2nd half
    }
    paths, ids = {}, {}
    for name, sig in sigs.items():
        p = str(tmp_path / f"{name}.wav")
        sf.write(p, sig, sr)
        ids[name] = tmp_store.upsert_track(Track(path=p, content_hash=f"h{name}"))
        paths[name] = p

    build_segment_index(tmp_store, "baseline", window_s=1.0, hop_s=1.0)

    q = embed_segment(paths["A"], 0.0, 2.0, "baseline")           # a 440 Hz region
    res = find_similar_segments(tmp_store, q, "baseline", n=5, exclude_track_id=ids["A"])

    assert res, "expected matches"
    assert res[0]["track_id"] == ids["C"]                         # C has a 440 part; B doesn't
    assert res[0]["start"] >= 1.5                                 # the 440 half of C (~2-4 s)


def test_save_and_list_segment_exemplar(tmp_store, tmp_path):
    sr = 22050
    p = str(tmp_path / "x.wav")
    sf.write(p, _tone(sr, 3, 440), sr)
    tid = tmp_store.upsert_track(Track(path=p, content_hash="hx"))
    vec = embed_segment(p, 0.5, 1.5, "baseline")
    sid = tmp_store.save_segment_exemplar(tid, "baseline", 0.5, 1.5, vec,
                                          label="electroclash cowbell", note="the defining sound")
    rows = tmp_store.get_segment_exemplars()
    assert any(r["id"] == sid and r["label"] == "electroclash cowbell" for r in rows)


def test_suggestions_blend_roundtrip(tmp_store):
    from mgc.types import GenreNode, Track

    tid = tmp_store.upsert_track(Track(path="/a.wav", content_hash="hblend"))
    g1 = tmp_store.upsert_genre(GenreNode(name="Deep House", level="subgenre"))
    g2 = tmp_store.upsert_genre(GenreNode(name="Tech House", level="subgenre"))
    tmp_store.save_suggestions(tid, [(g1, 0.82, "centroid"), (g2, 0.61, "centroid")])

    s = tmp_store.get_suggestions(tid)
    assert [r["name"] for r in s] == ["Deep House", "Tech House"]  # blend, best first
    assert s[0]["confidence"] == 0.82 and s[0]["rank"] == 0


def test_create_genre_from_segment(tmp_store, tmp_path):
    """Labeling a region seeds a by-example genre from the tracks with that sound."""
    import numpy as np
    import soundfile as sf

    from mgc.api.service import Engine
    from mgc.config import Config
    from mgc.segments import build_segment_index
    from mgc.types import Track

    sr = 22050
    sigs = {
        "A": _tone(sr, 4, 440),
        "B": _tone(sr, 4, 3000),
        "C": np.concatenate([_tone(sr, 2, 3000), _tone(sr, 2, 440)]),  # has a 440 part
    }
    ids = {}
    for name, sig in sigs.items():
        p = str(tmp_path / f"{name}.wav")
        sf.write(p, sig, sr)
        ids[name] = tmp_store.upsert_track(Track(path=p, content_hash=f"sg{name}"))
    build_segment_index(tmp_store, "baseline", window_s=1.0, hop_s=1.0)

    eng = Engine(Config(db_path=":memory:", active_model="baseline"), store=tmp_store)
    res = eng.create_genre_from_segment(ids["A"], 0.0, 2.0, "440 sound", n=5)
    assert res["ok"]
    assert ids["A"] in res["examples"] and ids["C"] in res["examples"]  # A + the 440-part track
    assert "440 sound" in {g.name for g in tmp_store.iter_genres()}
