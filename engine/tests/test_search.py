"""Open-vocab search router (AudioSet path) — no model download needed."""

from __future__ import annotations

import numpy as np

from mgc.types import Track


def test_audioset_search_ranks_by_class(tmp_store, monkeypatch):
    import mgc.tagging

    labels = ["Music", "Drum kit", "Cowbell", "Guitar"]  # toy 4-class ontology
    monkeypatch.setattr(mgc.tagging, "get_audioset_labels", lambda: labels)
    from mgc.search import search

    vecs = {
        "a": np.array([0.9, 0.1, 0.05, 0.2], np.float32),
        "b": np.array([0.8, 0.3, 0.85, 0.1], np.float32),  # the cowbell track
        "c": np.array([0.7, 0.2, 0.10, 0.6], np.float32),
    }
    ids = {}
    for name, v in vecs.items():
        tid = tmp_store.upsert_track(Track(path=f"{name}.wav", content_hash=f"h{name}"))
        tmp_store.save_understanding(tid, audioset=v, audioset_model="ast")
        ids[name] = tid

    r = search(tmp_store, "songs with cowbells")
    assert r["method"] == "audioset"
    assert r["matched_label"] == "Cowbell"
    assert r["results"][0]["track_id"] == ids["b"]            # cowbell track ranked first

    r2 = search(tmp_store, "cowbell", threshold=0.5)           # "return ALL above 0.5"
    assert [x["track_id"] for x in r2["results"]] == [ids["b"]]


def test_search_falls_through_to_clap(tmp_store, monkeypatch):
    import mgc.tagging

    monkeypatch.setattr(mgc.tagging, "get_audioset_labels", lambda: ["Music", "Guitar"])
    from mgc.search import search

    r = search(tmp_store, "dreamy nostalgic vaporwave")        # no AudioSet class -> CLAP path
    assert r["method"] == "clap"
    assert r["results"] == []                                  # no CLAP embeddings stored


def test_search_audioset_no_tags_yet(tmp_store, monkeypatch):
    import mgc.tagging

    monkeypatch.setattr(mgc.tagging, "get_audioset_labels", lambda: ["Cowbell"])
    from mgc.search import search

    r = search(tmp_store, "cowbell")
    assert r["method"] == "audioset" and r["results"] == []
    assert "note" in r
