"""Discogs label resolution (no Essentia needed — pure label logic)."""

from __future__ import annotations

import json

import pytest

from mgc.embed.discogs import DiscogsEmbedder


def test_labels_from_metadata_json(tmp_path):
    meta = tmp_path / "genre_discogs400-discogs-effnet-1.json"
    meta.write_text(json.dumps({"classes": ["Deep House", "Tech House", "Trance"]}), encoding="utf-8")
    emb = DiscogsEmbedder(genre_graph=str(tmp_path / "g.pb"), genre_metadata=str(meta))
    assert emb._load_labels() == ["Deep House", "Tech House", "Trance"]


def test_labels_explicit_override():
    emb = DiscogsEmbedder(labels=["A", "B"])
    assert emb._load_labels() == ["A", "B"]


def test_labels_missing_raises_not_placeholder(tmp_path):
    emb = DiscogsEmbedder(genre_graph=str(tmp_path / "nope.pb"),
                          genre_metadata=str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError):
        emb._load_labels()
