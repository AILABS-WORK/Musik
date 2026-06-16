"""Explainable similarity: shared vs differing reasons between two tracks."""

from __future__ import annotations

import numpy as np

from mgc.types import Track


def test_explain_similarity_shared_and_different(tmp_store, monkeypatch):
    import mgc.tagging

    labels = ["Music", "Guitar", "Piano", "Female singing", "Singing", "Synthesizer"]
    monkeypatch.setattr(mgc.tagging, "get_audioset_labels", lambda: labels)
    from mgc.similarity.explain import explain_similarity

    def vec(pairs):
        v = np.zeros(len(labels), np.float32)
        for i, val in pairs.items():
            v[i] = val
        return v

    a = tmp_store.upsert_track(Track(path="/a.wav", content_hash="ea"))
    b = tmp_store.upsert_track(Track(path="/b.wav", content_hash="eb"))
    tmp_store.save_embedding(a, "baseline", np.array([1, 0, 0], np.float32))
    tmp_store.save_embedding(b, "baseline", np.array([0.9, 0.1, 0], np.float32))
    # A: guitar + female vocal ; B: guitar + synth, instrumental
    tmp_store.save_understanding(a, audioset=vec({1: 0.8, 3: 0.7}), audioset_model="ast")
    tmp_store.save_understanding(b, audioset=vec({1: 0.75, 5: 0.6}), audioset_model="ast")
    tmp_store.save_analysis(a, bpm=124.0, music_key="C maj", energy=0.70)
    tmp_store.save_analysis(b, bpm=126.0, music_key="C maj", energy=0.72)

    r = explain_similarity(tmp_store, a, b, "baseline")
    assert r["score"] is not None
    shared = " | ".join(r["shared"]).lower()
    diff = " | ".join(r["different"]).lower()
    assert "guitar" in shared          # both have guitar
    assert "tempo" in shared           # 124 vs 126 -> within 4 BPM
    assert "key" in shared             # both C maj -> harmonically compatible
    assert "vocal" in diff or "instrumental" in diff  # A vocal, B instrumental
