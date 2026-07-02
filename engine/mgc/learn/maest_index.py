"""Index the library with MAEST Discogs-400 style predictions (GPU).

MAEST (MTG, ICASSP 2024) predicts 400 Discogs styles straight from audio — validated on
this library: the user's confirmed Deep House / Hard Techno tracks come back with the
right styles on top. We store each track's top styles as the "AI style" suggestion
source for initial labelling, then report agreement against every user label.

Usage: python -m mgc.learn.maest_index [--db PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODEL_ID = "mtg-upf/discogs-maest-30s-pw-73e-ts"


def load_model():
    try:
        import truststore  # TLS-inspecting proxy: required for any HTTPS (HF hub)
        truststore.inject_into_ssl()
    except Exception:
        pass
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    return fe, model, dev


def predict(fe, model, dev, path: str, n_seg: int = 3) -> np.ndarray:
    """Mean sigmoid over n_seg 30s segments spread across the track."""
    import torch
    from mgc.audio.decode import load_mono
    y, sr = load_mono(path, 16000)
    need = 30 * sr
    if y.size < need:
        y = np.pad(y, (0, need - y.size))
    preds = []
    for frac in np.linspace(0.15, 0.7, n_seg):
        s = int(min(max(0, y.size - need), frac * y.size))
        seg = y[s:s + need]
        inputs = fe(seg, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            preds.append(model(**inputs).logits.sigmoid()[0].float().cpu().numpy())
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r"C:/temp/musik_real/library.sqlite")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from mgc.store import Store
    store = Store.open(args.db)
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS maest (track_id INTEGER PRIMARY KEY, styles TEXT NOT NULL)")
    store.conn.commit()

    fe, model, dev = load_model()
    print(f"MAEST on {dev}")
    id2label = model.config.id2label

    tracks = store.iter_tracks()
    if args.limit:
        tracks = tracks[:args.limit]
    done = 0
    for i, t in enumerate(tracks):
        if not args.force and store.conn.execute(
                "SELECT 1 FROM maest WHERE track_id=?", (t.id,)).fetchone():
            continue
        if not os.path.exists(t.path):
            continue
        try:
            act = predict(fe, model, dev, t.path)
        except Exception as e:
            print(f"  skip {t.id}: {str(e)[:60]}")
            continue
        order = np.argsort(-act)[:10]
        styles = [{"style": id2label[int(j)], "p": round(float(act[j]), 3)} for j in order]
        store.conn.execute("INSERT OR REPLACE INTO maest(track_id, styles) VALUES(?,?)",
                           (t.id, json.dumps(styles)))
        store.conn.commit()
        done += 1
        print(f"  {i + 1}/{len(tracks)} indexed", end="\r", flush=True)
    print(f"\nindexed {done} tracks")

    # ---- validation against every user label -------------------------------
    rows = store.conn.execute(
        """SELECT e.track_id tid, g.name name FROM exemplars e JOIN genres g ON g.id=e.genre_id
           UNION SELECT a.track_id, g.name FROM assignments a JOIN genres g ON g.id=a.genre_id
           WHERE a.status='confirmed' AND a.genre_id IS NOT NULL""").fetchall()
    # user vocab -> acceptable Discogs styles (family matches count as agreement)
    ACCEPT = {
        "hard techno": ["hard techno", "schranz", "techno"],
        "deep house": ["deep house", "house"],
        "groovy house": ["house", "deep house", "tech house", "disco"],
        "minimal vocal": ["minimal", "tech house", "deep house", "house", "minimal techno"],
        "vocal groove": ["house", "deep house", "garage house", "uk garage"],
        "acid house": ["acid house", "acid", "house"],
        "move house": ["house", "tech house", "deep house"],
        "minimal": ["minimal", "minimal techno", "tech house"],
        "acid minimal": ["acid", "minimal", "minimal techno", "techno"],
        "drum groove techno": ["techno", "tribal", "hard techno", "minimal techno"],
    }
    agree = total = 0
    misses = []
    for r in rows:
        row = store.conn.execute("SELECT styles FROM maest WHERE track_id=?", (r["tid"],)).fetchone()
        if row is None:
            continue
        nm = (r["name"] or "").strip().lower()
        acc = ACCEPT.get(nm, [nm])
        top5 = [s["style"].split("---")[-1].lower() for s in json.loads(row["styles"])[:5]]
        total += 1
        if any(a in t for t in top5 for a in acc):
            agree += 1
        else:
            misses.append((nm, top5[:3]))
    if total:
        print(f"agreement with YOUR labels (top-5, family-aware): {agree}/{total} = {agree / total:.0%}")
        for nm, t5 in misses[:8]:
            print(f"   miss [{nm}] got {t5}")


if __name__ == "__main__":
    main()
