"""Deep-pass orchestration — fakes for the heavy backends (no model downloads).

Verifies the logic: a re-tagged drum stem MAX-boosts the AudioSet vector, the
vocal stem yields a language, deep_done flips, and the facade surfaces the
language. The real Demucs/Whisper backends are exercised separately (smoke).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from mgc.deep import deep_analyze
from mgc.types import Track


def _tone(path, sr=22050, secs=1.0, freq=220.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)
    return path


def test_deep_analyze_boosts_tags_and_detects_language(tmp_store, tmp_path):
    track_p = _tone(str(tmp_path / "t.wav"))
    tid = tmp_store.upsert_track(Track(path=track_p, content_hash="h"))
    base = np.zeros(527, np.float32)
    base[396] = 0.05                                   # weak on the full mix
    tmp_store.save_understanding(tid, audioset=base, audioset_model="ast")

    drum = _tone(str(tmp_path / "drums.wav"))
    voc = _tone(str(tmp_path / "vocals.wav"))

    def fake_sep(_path, _out):
        return {"drums": drum, "other": drum, "vocals": voc, "bass": drum}

    class FakeTagger:
        name = "ast"
        def tag_file(self, _sp):
            v = np.zeros(527, np.float32)
            v[396] = 0.9                                # strong on the isolated stem
            return v

    res = deep_analyze(tmp_store, tid, tagger=FakeTagger(), separator=fake_sep,
                       detector=lambda _p: {"language": "en", "confidence": 0.88})
    assert res["ok"] and res["stem_verified"]
    assert res["language"]["language"] == "en"

    u = tmp_store.get_understanding(tid)
    assert u["deep_done"] == 1
    assert float(u["audioset"][396]) >= 0.85           # MAX-boosted from the stem (float32)
    assert u["vocal"]["language"] == "en"


def test_deep_analyze_missing_track_is_clean():
    class Dummy:
        def get_track(self, _):
            return None

    res = deep_analyze(Dummy(), 999, separator=lambda _p, _o: {})
    assert res["ok"] is False


def test_facade_understanding_surfaces_language(tmp_store, tmp_path):
    from mgc.api.service import Engine
    from mgc.config import Config

    track_p = _tone(str(tmp_path / "t.wav"))
    tid = tmp_store.upsert_track(Track(path=track_p, content_hash="h"))
    tmp_store.save_understanding(tid, audioset=np.zeros(527, np.float32), audioset_model="ast")
    drum = _tone(str(tmp_path / "drums.wav"))
    voc = _tone(str(tmp_path / "vocals.wav"))

    class FakeTagger:
        name = "ast"
        def tag_file(self, _sp):
            return np.zeros(527, np.float32)

    deep_analyze(tmp_store, tid, tagger=FakeTagger(),
                 separator=lambda _p, _o: {"drums": drum, "vocals": voc},
                 detector=lambda _p: {"language": "es", "confidence": 0.9})

    eng = Engine(Config(db_path=":memory:", active_model="baseline"), store=tmp_store)
    u = eng.understanding(tid)
    assert u["vocal"]["language"] == "es"
    assert u["deep_done"] == 1
