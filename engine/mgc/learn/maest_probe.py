"""Validate the MAEST Discogs-400 style model on the user's confirmed tracks.

MAEST (Alonso-Jimenez et al., ICASSP 2024) predicts 400 Discogs styles straight from
audio. The ONNX takes a 30s mel-spectrogram [1876, 96]; the exact mel recipe is
essentia's TensorflowInputMusiCNN (frame 512 / hop 256 / 96 mels @16kHz, log10(1+10000x))
— we implement it with librosa and validate variants against tracks the user labelled,
picking the variant (if any) whose predictions agree with human ground truth.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

MODEL_DIR = r"C:\temp\effnet"


def mel_musicnn(y: np.ndarray, sr: int = 16000, variant: str = "slaney") -> np.ndarray:
    """96-band mel [frames, 96] in the MusiCNN/MAEST style."""
    import librosa
    kw = dict(y=y, sr=sr, n_fft=512, hop_length=256, win_length=512,
              window="hann", center=True, power=2.0, n_mels=96)
    if variant == "slaney":
        m = librosa.feature.melspectrogram(**kw, htk=False, norm="slaney")
    elif variant == "htk":
        m = librosa.feature.melspectrogram(**kw, htk=True, norm=None)
    else:
        raise ValueError(variant)
    m = np.log10(1.0 + 10000.0 * m)
    return m.T.astype(np.float32)  # [frames, 96]


def segments_30s(path: str, n_seg: int = 3) -> list:
    from mgc.audio.decode import load_mono
    y, sr = load_mono(path, 16000)
    need = 30 * sr
    if y.size < need:
        y = np.pad(y, (0, need - y.size))
    outs = []
    for frac in np.linspace(0.15, 0.7, n_seg):
        s = int(min(max(0, y.size - need), frac * y.size))
        outs.append(y[s:s + need])
    return outs


class Maest:
    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "discogs-maest-30s-pw-1.onnx"),
            providers=["CPUExecutionProvider"])
        self.classes = json.load(open(os.path.join(MODEL_DIR, "labels.json"),
                                      encoding="utf8"))["classes"]
        self.in_name = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_name = next((n for n in outs if "activation" in n.lower()), outs[0])

    def predict(self, path: str, variant: str = "slaney", norm=None) -> np.ndarray:
        """Mean sigmoid activations [400] over three 30s segments."""
        preds = []
        for seg in segments_30s(path):
            m = mel_musicnn(seg, variant=variant)
            m = m[:1876]
            if m.shape[0] < 1876:
                m = np.pad(m, ((0, 1876 - m.shape[0]), (0, 0)))
            if norm is not None:
                mu, sd = norm
                m = (m - mu) / sd
            out = self.sess.run([self.out_name], {self.in_name: m[None]})[0]
            preds.append(np.asarray(out).reshape(-1))
        return np.mean(preds, axis=0)

    def top(self, act: np.ndarray, k: int = 5):
        order = np.argsort(-act)[:k]
        return [(self.classes[i], round(float(act[i]), 3)) for i in order]


def main(db: str):
    import sqlite3
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    picks = c.execute(
        """SELECT t.path, g.name FROM assignments a
           JOIN tracks t ON t.id=a.track_id JOIN genres g ON g.id=a.genre_id
           WHERE a.status='confirmed' AND lower(g.name) IN
             ('hard techno','deep house','acid house','groovy house')
           GROUP BY g.name HAVING t.path LIKE '%.mp3' LIMIT 8""").fetchall()
    # a couple per genre
    picks = c.execute(
        """SELECT t.path, g.name FROM assignments a
           JOIN tracks t ON t.id=a.track_id JOIN genres g ON g.id=a.genre_id
           WHERE a.status='confirmed' AND lower(g.name) IN
             ('hard techno','deep house') AND t.path LIKE '%.mp3'
           ORDER BY g.name, a.track_id LIMIT 6""").fetchall()
    mm = Maest()
    for variant in ("slaney", "htk"):
        print(f"=== mel variant: {variant} ===")
        for r in picks:
            act = mm.predict(r["path"], variant=variant)
            top = mm.top(act, 4)
            name = os.path.basename(r["path"])[:34]
            print(f"  [{r['name']:<11}] {name}")
            print(f"      -> " + " | ".join(f"{c.split('---')[-1]} {p}" for c, p in top))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:/temp/musik_real/library.sqlite")
