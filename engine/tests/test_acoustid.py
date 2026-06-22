"""AcoustID fingerprint identification + MusicBrainz by-MBID lookup (no network/key)."""

from mgc.metadata import acoustid as aid
from mgc.metadata.musicbrainz import mb_lookup_by_mbid


def test_pick_best_takes_highest_scoring_with_recording():
    results = [
        {"score": 0.9, "recordings": []},  # no recording -> skip despite high score
        {"score": 0.7, "recordings": [
            {"id": "mbid-1", "title": "Around the World",
             "artists": [{"name": "Daft Punk"}]}]},
        {"score": 0.5, "recordings": [{"id": "mbid-2", "title": "Other"}]},
    ]
    best = aid.pick_best(results)
    assert best["recording_mbid"] == "mbid-1"
    assert best["artist"] == "Daft Punk" and best["title"] == "Around the World"
    assert best["score"] == 0.7


def test_pick_best_none_when_no_recordings():
    assert aid.pick_best([{"score": 1.0, "recordings": []}]) is None
    assert aid.pick_best([]) is None


def test_identify_uses_injected_fingerprint_and_lookup():
    # no fpcalc / no network: inject both
    fp = lambda path: (300.0, b"FAKEFP")
    look = lambda fingerprint, duration, key, meta: [
        {"score": 0.95, "recordings": [{"id": "rec-42", "title": "T",
                                        "artists": [{"name": "A"}]}]}]
    out = aid.identify("x.mp3", key="DUMMY", looker=look, fingerprinter=fp)
    assert out["recording_mbid"] == "rec-42" and out["score"] == 0.95


def test_identify_without_key_reports_missing(monkeypatch):
    # force "no key" regardless of any .env / env var in the test environment
    monkeypatch.setattr(aid, "api_key", lambda explicit=None: None)
    out = aid.identify("x.mp3", looker=lambda *a: [], fingerprinter=lambda p: (1.0, b""))
    assert out.get("error") == "no_acoustid_key"


def test_mb_lookup_by_mbid_parses_genre_and_region():
    fake = {
        "title": "Around the World",
        "artist-credit": [{"artist": {"name": "Daft Punk", "id": "artist-mbid"}}],
        "genres": [{"name": "french house", "count": 5}, {"name": "house", "count": 3}],
        "tags": [{"name": "electronic"}],
        "first-release-date": "1997-01-01",
        "releases": [{"country": "FR"}],
    }
    info = mb_lookup_by_mbid("rec-1", get=lambda url: fake)
    assert info["recording_mbid"] == "rec-1"
    assert "french house" in info["genres"] and "house" in info["genres"]
    assert info["title"] == "Around the World"
    assert info["area"] == "FR"
    assert info["year"] == "1997"


def test_parse_artist_title_strips_premiere_and_label():
    from mgc.metadata import parse_artist_title

    a, t = parse_artist_title("BCCO Premiere: Alexander Johansson & Tim Hök - Sci-Fi Dub [LERICHE03]")
    assert a == "Alexander Johansson & Tim Hök" and t == "Sci-Fi Dub"
    # country/region tag stripped from the artist (helps matching)
    a, t = parse_artist_title("PREMIERE: Enoch (SA) - Kaya [Label]")
    assert a == "Enoch" and t == "Kaya"
    # leading vinyl position stripped
    a, t = parse_artist_title("A1. Livin' Large - The Second Coming [Rise 'n' Shine]")
    assert a == "Livin' Large" and t == "The Second Coming"
    # label as ID3 artist is rejected in favour of the parsed name
    a, t = parse_artist_title("Reality Check", artist="Mixmag premiere")
    assert a is None and t == "Reality Check"


def test_discogs_parse_search_prefers_styles():
    from mgc.metadata import discogs

    data = {"results": [
        {"id": 1, "genre": ["Electronic"], "style": []},          # no style -> still usable
        {"id": 2, "genre": ["Electronic"], "style": ["Tech House"], "year": 2021},
    ]}
    out = discogs.parse_search(data)
    # first result already has a genre, so it's taken
    assert out["genres"] == ["Electronic"]


def test_discogs_lookup_without_creds(monkeypatch):
    from mgc.metadata import discogs

    # force "no creds" regardless of any .env in the test environment
    monkeypatch.setattr(discogs, "creds", lambda key=None, secret=None: (None, None))
    out = discogs.lookup("A", "B", get=lambda url: {"results": []})
    assert out.get("error") == "no_discogs_creds"


def test_save_get_identity_roundtrip(tmp_path):
    from mgc.store import Store
    from mgc.types import Track

    s = Store.open(str(tmp_path / "t.sqlite"))
    tid = s.upsert_track(Track(path="a.mp3", content_hash="h1"))
    assert s.has_identity(tid) is False
    s.save_identity(tid, recording_mbid="rec-1", artist="Daft Punk",
                    title="Around the World", genres=["french house", "house"],
                    area="FR", year="1997", score=0.91)
    got = s.get_identity(tid)
    assert got["recording_mbid"] == "rec-1" and got["area"] == "FR"
    assert got["genres"] == ["french house", "house"]
    assert s.has_identity(tid) is True
