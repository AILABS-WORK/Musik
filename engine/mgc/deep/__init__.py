"""Deep analysis pass (opt-in, GPU): separate stems, re-tag the drum/'other'
stems for fine percussion, and detect the sung language on the vocal stem.

Composes the lazy backends (Demucs + AudioSet tagger + Whisper). All backends are
injectable so the orchestration is unit-tested without any model downloads.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mgc.deep.language import detect_language
from mgc.deep.separate import separate

__all__ = ["deep_analyze", "deep_analyze_all", "separate", "detect_language"]


def deep_analyze(store, track_id: int, tagger=None, separator=None, detector=None,
                 work_dir=None) -> dict:
    """Run the deep pass for one track; update its understanding record.

    Steps: separate -> re-tag drums+other stems and MAX-merge into the AudioSet
    vector (lifts quiet percussion) -> detect language on the vocal stem -> persist
    (audioset boosted, vocal.language, deep_done=1). Returns a summary dict.
    """
    track = store.get_track(track_id)
    if not track:
        return {"ok": False, "error": "no such track"}

    work = work_dir or os.path.join(tempfile.gettempdir(), "mgc_stems", str(track_id))
    stems = separate(track.path, work, separator=separator)
    if not stems:
        return {"ok": False, "error": "separation produced no stems"}

    if tagger is None:
        from mgc.tagging import AudioSetTagger
        tagger = AudioSetTagger()

    u = store.get_understanding(track_id)
    boosted = None
    if u and u.get("audioset") is not None:
        boosted = np.asarray(u["audioset"], dtype=np.float32)

    # Re-tag the percussion-bearing stems and keep the element-wise max.
    for stem_name in ("drums", "other"):
        sp = stems.get(stem_name)
        if not sp or not os.path.exists(sp):
            continue
        try:
            v = tagger.tag_file(sp)
        except Exception:
            v = None
        if v is not None:
            v = np.asarray(v, dtype=np.float32)
            boosted = v if boosted is None else np.maximum(boosted, v)

    # Sung-language on the isolated vocal stem.
    lang = None
    vp = stems.get("vocals")
    if vp and os.path.exists(vp):
        try:
            lang = detect_language(vp, detector=detector)
        except Exception:
            lang = None

    stored_vocal = (u or {}).get("vocal")
    new_vocal = dict(stored_vocal) if isinstance(stored_vocal, dict) else {}
    if lang and lang.get("language"):
        new_vocal["language"] = lang["language"]
        new_vocal["language_conf"] = lang.get("confidence")

    store.save_understanding(track_id, audioset=boosted,
                             audioset_model="ast+stems" if boosted is not None else None,
                             vocal=new_vocal or None, deep_done=1)
    return {"ok": True, "stems": sorted(stems.keys()), "language": lang,
            "stem_verified": True}


def deep_analyze_all(store, tagger=None, progress=None) -> int:
    """Deep-analyze every track that has been tagged but not deep-analyzed yet."""
    if tagger is None:
        from mgc.tagging import AudioSetTagger
        tagger = AudioSetTagger()
    tracks = store.iter_tracks()
    done = 0
    for i, t in enumerate(tracks):
        u = store.get_understanding(t.id)
        if not u or u.get("audioset") is None or u.get("deep_done"):
            continue
        try:
            deep_analyze(store, t.id, tagger=tagger)
            done += 1
        except Exception:
            pass
        if progress:
            progress(i + 1, len(tracks))
    return done
