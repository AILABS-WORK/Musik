"""Unit tests for mgc.registry (centroids + few-shot custom genres).

Exercises only the registry module against a real Store. Embeddings are written
directly as numpy arrays; cross-module deps are stubbed (a tiny fake CLAP).
"""

import numpy as np

from mgc.registry import (
    add_exemplar,
    create_genre_by_example,
    recompute_centroid,
    seed_by_name,
)
from mgc.types import GenreNode, Track, LEVEL_SUBGENRE


def _l2(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / np.linalg.norm(v)


def _add_track(store, name, vec, model="baseline"):
    tid = store.upsert_track(Track(path=f"{name}.wav", content_hash=name))
    store.save_embedding(tid, model, np.asarray(vec, dtype=np.float32))
    return tid


def test_create_genre_by_example_centroid_is_normalized_mean(tmp_store):
    s = tmp_store
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 3.0, 1.0], dtype=np.float32)
    t1 = _add_track(s, "t1", v1)
    t2 = _add_track(s, "t2", v2)
    t3 = _add_track(s, "t3", v3)

    gid = create_genre_by_example(s, "My Genre", [t1, t2, t3], "baseline")

    # Genre persisted as custom, default subgenre level, no parent.
    g = s.get_genre(gid)
    assert g.name == "My Genre"
    assert g.source == "custom"
    assert g.level == LEVEL_SUBGENRE
    assert g.parent_id is None
    assert s.get_exemplars(gid) == sorted([t1, t2, t3])

    expected = _l2(np.mean(np.stack([v1, v2, v3]), axis=0))
    stored = s.get_centroid(gid)
    assert stored is not None
    assert stored.shape == (4,)
    assert np.allclose(stored, expected, atol=1e-6)
    # Centroid is unit length.
    assert np.isclose(np.linalg.norm(stored), 1.0, atol=1e-6)
    # is_text flag is False for example-derived centroids.
    assert s.iter_centroids()[0][3] is False


def test_add_exemplar_updates_centroid(tmp_store):
    s = tmp_store
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 3.0, 1.0], dtype=np.float32)
    v4 = np.array([4.0, 1.0, 0.0, 2.0], dtype=np.float32)
    t1 = _add_track(s, "t1", v1)
    t2 = _add_track(s, "t2", v2)
    t3 = _add_track(s, "t3", v3)
    t4 = _add_track(s, "t4", v4)

    gid = create_genre_by_example(s, "G", [t1, t2, t3], "baseline")
    before = s.get_centroid(gid)

    returned = add_exemplar(s, gid, t4, "baseline")
    after = s.get_centroid(gid)

    expected = _l2(np.mean(np.stack([v1, v2, v3, v4]), axis=0))
    assert np.allclose(returned, expected, atol=1e-6)
    assert np.allclose(after, expected, atol=1e-6)
    assert not np.allclose(before, after, atol=1e-6)
    assert s.get_exemplars(gid) == sorted([t1, t2, t3, t4])


def test_recompute_centroid_none_without_exemplars(tmp_store):
    s = tmp_store
    gid = s.upsert_genre(GenreNode(name="Empty", source="custom"))
    assert recompute_centroid(s, gid, "baseline") is None
    assert s.get_centroid(gid) is None


def test_recompute_centroid_skips_missing_embeddings(tmp_store):
    s = tmp_store
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    t1 = _add_track(s, "t1", v1)
    # Exemplar track without any embedding for this model.
    t2 = s.upsert_track(Track(path="t2.wav", content_hash="t2"))
    gid = s.upsert_genre(GenreNode(name="G", source="custom"))
    s.add_exemplar(gid, t1)
    s.add_exemplar(gid, t2)

    cen = recompute_centroid(s, gid, "baseline")
    assert np.allclose(cen, _l2(v1), atol=1e-6)


def test_recompute_centroid_none_when_no_embeddings_at_all(tmp_store):
    s = tmp_store
    t1 = s.upsert_track(Track(path="t1.wav", content_hash="t1"))
    gid = s.upsert_genre(GenreNode(name="G", source="custom"))
    s.add_exemplar(gid, t1)
    assert recompute_centroid(s, gid, "baseline") is None


def test_create_genre_by_example_with_parent_and_level(tmp_store):
    s = tmp_store
    parent = s.upsert_genre(GenreNode(name="House", level="genre"))
    v1 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    t1 = _add_track(s, "t1", v1)
    gid = create_genre_by_example(
        s, "Deep House", [t1], "baseline", parent_id=parent, level=LEVEL_SUBGENRE
    )
    g = s.get_genre(gid)
    assert g.parent_id == parent
    assert g.level == LEVEL_SUBGENRE
    assert [c.id for c in s.children(parent)] == [gid]


def test_seed_by_name_stores_text_centroid(tmp_store):
    s = tmp_store
    gid = s.upsert_genre(GenreNode(name="Lofi", source="custom"))

    class FakeClap:
        def __init__(self, vec):
            self._vec = np.asarray(vec, dtype=np.float32)
            self.calls = []

        def text_embed(self, text):
            self.calls.append(text)
            return self._vec

    fake = FakeClap([0.1, 0.2, 0.3, 0.4])
    out = seed_by_name(s, gid, "chill lofi hip hop beats", fake)

    assert fake.calls == ["chill lofi hip hop beats"]
    assert np.allclose(out, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    stored = s.get_centroid(gid)
    assert np.allclose(stored, out, atol=1e-6)
    # Marked as a text-derived centroid.
    cents = {c[0]: c[3] for c in s.iter_centroids()}
    assert cents[gid] is True
