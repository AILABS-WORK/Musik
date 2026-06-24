"""Per-frequency-range similarity breakdown."""

from mgc.similarity.bands import RANGES, band_breakdown


def test_identical_profiles_match_fully():
    prof = [0.25] * 16
    out = band_breakdown(prof, prof)
    assert len(out["bands"]) == len(RANGES)
    assert all(b["match"] == 1.0 for b in out["bands"])
    assert out["overall"] == 1.0


def test_band_names_are_the_named_ranges():
    out = band_breakdown([0.25] * 16, [0.25] * 16)
    assert [b["name"] for b in out["bands"]] == [r[0] for r in RANGES]


def test_bass_heavy_vs_bright_differs_low_and_high():
    # a: energy in the lowest bands; b: energy in the highest bands.
    a = [1.0, 1.0, 1.0] + [0.0] * 13
    b = [0.0] * 13 + [1.0, 1.0, 1.0]
    out = band_breakdown(a, b)
    by = {bd["name"]: bd["match"] for bd in out["bands"]}
    assert by["sub"] < 0.5      # a has sub, b doesn't -> low match
    assert by["highs"] < 0.5    # b has highs, a doesn't -> low match


def test_bad_input_is_safe():
    assert band_breakdown(None, [0.1] * 16)["bands"] == []
    assert band_breakdown([0.1] * 4, [0.1] * 16)["bands"] == []
