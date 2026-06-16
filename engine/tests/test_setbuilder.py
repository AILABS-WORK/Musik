"""Unit tests for mgc.setbuilder.

Light deps only: tracks + analysis are written directly into the store. No
embeddings or models are required (the builder falls back to all tracks when a
model has no embeddings). The arc/ordering assertions are deliberately lenient
but real.
"""

from __future__ import annotations

from mgc.setbuilder import build_set, parse_description
from mgc.types import Track


# ---------------------------------------------------------------------------
# parse_description
# ---------------------------------------------------------------------------

def test_parse_returns_expected_shape():
    p = parse_description("a chill deep house set")
    assert set(p.keys()) == {"genres", "energy_arc", "bpm_hint", "length", "notes"}
    assert isinstance(p["energy_arc"], list)
    assert 5 <= len(p["energy_arc"]) <= 7
    assert all(0.0 <= x <= 1.0 for x in p["energy_arc"])
    assert isinstance(p["notes"], str) and p["notes"]


def test_parse_low_energy_keywords():
    p = parse_description("slow chill deep minimal downtempo")
    arc = p["energy_arc"]
    # A low-energy description should average low.
    assert sum(arc) / len(arc) < 0.5


def test_parse_high_energy_keywords():
    p = parse_description("punchy peak energetic harder banging")
    arc = p["energy_arc"]
    assert sum(arc) / len(arc) > 0.5


def test_parse_rise_then_fall_phrasing():
    p = parse_description("start slow, build to a peak, then slow down and end deep")
    arc = p["energy_arc"]
    peak = max(arc)
    peak_idx = arc.index(peak)
    # Peak lands somewhere in the middle, ends below the peak (rise then fall).
    assert 0 < peak_idx < len(arc) - 1
    assert arc[-1] < peak
    assert arc[0] < peak


def test_parse_detects_explicit_bpm_single():
    p = parse_description("deep house at 122 bpm")
    assert p["bpm_hint"] == (122, 122)


def test_parse_detects_bpm_range():
    p = parse_description("techno 124-128 bpm peak time")
    assert p["bpm_hint"] == (124, 128)


def test_parse_detects_length():
    p = parse_description("give me 10 tracks of groovy house")
    assert p["length"] == 10


def test_parse_detects_genres():
    p = parse_description("a deep house into techno journey")
    assert "deep house" in p["genres"]
    assert "techno" in p["genres"]


def test_parse_default_arc_when_no_keywords():
    p = parse_description("make me a set")
    arc = p["energy_arc"]
    # Default = gentle rise then fall: peak in the interior.
    peak_idx = arc.index(max(arc))
    assert 0 < peak_idx < len(arc) - 1


# ---------------------------------------------------------------------------
# build_set
# ---------------------------------------------------------------------------

def _make_tracks_with_energy(store, n=8):
    """Create n tracks spanning energies 0.1..0.9 with ascending bpms."""
    energies = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9][:n]
    ids = []
    for i, e in enumerate(energies):
        tid = store.upsert_track(Track(path=f"/lib/t{i}.wav", content_hash=f"h{i}"))
        store.save_analysis(tid, bpm=118.0 + i * 2.0, energy=e)
        ids.append(tid)
    return ids, energies


def test_build_set_returns_requested_length_and_unique(tmp_store):
    _make_tracks_with_energy(tmp_store, n=8)
    res = build_set(
        tmp_store,
        "start slow and groovy then build punchier then slow down deep and minimal",
        model="baseline",
        length=6,
    )
    assert len(res["track_ids"]) == 6
    assert len(set(res["track_ids"])) == 6  # unique
    assert len(res["arc"]) == 6
    assert len(res["reasons"]) == 6


def test_build_set_energies_rise_then_fall(tmp_store):
    ids, _ = _make_tracks_with_energy(tmp_store, n=8)
    res = build_set(
        tmp_store,
        "start slow and groovy then build punchier then slow down deep and minimal",
        model="baseline",
        length=6,
    )

    analysis = tmp_store.load_analysis()
    chosen_e = [analysis[t]["energy"] for t in res["track_ids"]]

    peak = max(chosen_e)
    peak_idx = chosen_e.index(peak)
    n = len(chosen_e)
    # Peak sits in the middle third.
    assert n // 3 <= peak_idx <= (2 * n) // 3 + 1
    # Ends lower than the peak (the set comes back down).
    assert chosen_e[-1] < peak
    # And the early part rises into the peak.
    assert chosen_e[0] <= peak


def test_build_set_reasons_contain_energy_and_bpm(tmp_store):
    _make_tracks_with_energy(tmp_store, n=8)
    res = build_set(tmp_store, "groovy build then slow", model="baseline", length=4)
    for r in res["reasons"]:
        assert "energy" in r
        assert "bpm" in r


def test_build_set_no_embeddings_uses_all_tracks(tmp_store):
    # No embeddings saved for the model -> falls back to all tracks.
    ids, _ = _make_tracks_with_energy(tmp_store, n=5)
    res = build_set(tmp_store, "build it up", model="no-such-model")
    # Default length is min(12, n_candidates) == 5.
    assert len(res["track_ids"]) == 5
    assert set(res["track_ids"]) == set(ids)


def test_build_set_length_capped_to_candidates(tmp_store):
    ids, _ = _make_tracks_with_energy(tmp_store, n=4)
    res = build_set(tmp_store, "deep set", model="baseline", length=20)
    assert len(res["track_ids"]) == 4


def test_build_set_missing_energy_defaults(tmp_store):
    # Track with no analysis -> energy defaults to 0.5, still selectable.
    t = tmp_store.upsert_track(Track(path="/lib/x.wav", content_hash="x"))
    res = build_set(tmp_store, "anything", model="baseline", length=1)
    assert res["track_ids"] == [t]
    assert len(res["arc"]) == 1
    # Reason has energy but no bpm (no analysis).
    assert "energy" in res["reasons"][0]


def test_build_set_empty_store(tmp_store):
    res = build_set(tmp_store, "anything", model="baseline", length=5)
    assert res["track_ids"] == []
    assert res["arc"] == []
    assert res["reasons"] == []
    assert "parsed" in res


def test_build_set_parsed_included(tmp_store):
    _make_tracks_with_energy(tmp_store, n=3)
    res = build_set(tmp_store, "deep house 122 bpm", model="baseline", length=2)
    assert res["parsed"]["bpm_hint"] == (122, 122)
    assert "deep house" in res["parsed"]["genres"]
