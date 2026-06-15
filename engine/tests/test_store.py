import numpy as np

from mgc.types import Track, GenreNode, LEVEL_SUBGENRE


def test_migrate_and_track_roundtrip(tmp_store):
    s = tmp_store
    tid = s.upsert_track(Track(path="x.wav", content_hash="h1", fmt="wav",
                               duration=2.0, sample_rate=22050,
                               existing_tags={"artist": "a"}))
    t = s.get_track(tid)
    assert t.path == "x.wav"
    assert t.existing_tags["artist"] == "a"
    assert s.count_tracks() == 1

    # upsert with same content hash updates the existing row (no duplicate)
    tid2 = s.upsert_track(Track(path="y.wav", content_hash="h1", fmt="wav"))
    assert tid2 == tid
    assert s.get_track(tid).path == "y.wav"
    assert s.count_tracks() == 1
    assert s.get_track_by_hash("h1").id == tid


def test_embedding_roundtrip(tmp_store):
    s = tmp_store
    tid = s.upsert_track(Track(path="x.wav", content_hash="h1"))
    v = np.arange(8, dtype=np.float32)
    s.save_embedding(tid, "baseline", v)
    assert s.has_embedding(tid, "baseline")
    out = s.get_embedding(tid, "baseline")
    assert np.allclose(out, v)
    ids, mat = s.load_matrix("baseline")
    assert ids == [tid]
    assert mat.shape == (1, 8)


def test_genre_tree_and_centroid(tmp_store):
    s = tmp_store
    parent = s.upsert_genre(GenreNode(name="House", level="genre"))
    child = s.upsert_genre(GenreNode(name="Tech House", parent_id=parent, level=LEVEL_SUBGENRE))
    assert s.get_genre(child).parent_id == parent
    assert [g.id for g in s.children(parent)] == [child]

    s.set_centroid(child, np.ones(4, dtype=np.float32), is_text=False)
    cen = s.get_centroid(child)
    assert cen.shape == (4,)
    cents = s.iter_centroids()
    assert len(cents) == 1 and cents[0][0] == child


def test_exemplars_and_assignment(tmp_store):
    s = tmp_store
    g = s.upsert_genre(GenreNode(name="Trance"))
    t1 = s.upsert_track(Track(path="a.wav", content_hash="ha"))
    s.add_exemplar(g, t1)
    s.add_exemplar(g, t1)  # idempotent
    assert s.get_exemplars(g) == [t1]
    s.set_assignment(t1, g, 0.9, "centroid", status="confirmed")
    row = s.get_assignment(t1)
    assert row["genre_id"] == g and row["status"] == "confirmed"


def test_action_log_and_undo(tmp_store):
    s = tmp_store
    tid = s.upsert_track(Track(path="x.wav", content_hash="h1"))
    aid = s.log_action("tag_write", tid, from_value="", to_value="Tech House", undo_token="x.wav")
    done = s.iter_actions(status="done")
    assert len(done) == 1 and done[0].to_value == "Tech House"
    s.set_action_status(aid, "undone")
    assert s.iter_actions(status="done") == []
    assert len(s.iter_actions(status="undone")) == 1


def test_clusters(tmp_store):
    s = tmp_store
    t1 = s.upsert_track(Track(path="a.wav", content_hash="ha"))
    t2 = s.upsert_track(Track(path="b.wav", content_hash="hb"))
    cid = s.add_cluster("run1")
    s.add_cluster_member(cid, t1)
    s.add_cluster_member(cid, t2)
    clusters = s.get_clusters("run1")
    assert len(clusters) == 1
    assert set(clusters[0].member_track_ids) == {t1, t2}
