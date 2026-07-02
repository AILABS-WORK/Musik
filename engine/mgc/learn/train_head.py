"""Train OUR OWN model on the user's labelled tracks (GPU).

Key insight: we store one mean-pooled embedding per track, but each track was embedded
as ~24 five-second windows. Re-extracting per-window embeddings turns ~111 labelled
tracks into ~2,600 training samples — enough to train a small neural projection head
that reshapes MuQ space around the USER's subgenre boundaries (few-shot metric
learning, per the literature).

Honest protocol: cross-validation is GROUPED BY TRACK (windows of one track never span
the train/test split), and test tracks are scored the way deployment scores them — one
mean-pooled vector per track, classified by cosine to class prototypes. Baseline for
comparison is the current shrinkage-LDA on mean embeddings.

Usage (engine venv):  python -m mgc.learn.train_head --db C:/temp/musik_real/library.sqlite
"""

from __future__ import annotations

import argparse
import collections
import os

import numpy as np

CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "musik_window_emb.npz")
MIN_PER_CLASS = 5


def load_labels(store):
    rows = store.conn.execute(
        "SELECT e.track_id tid, g.name name FROM exemplars e JOIN genres g ON g.id=e.genre_id").fetchall()
    rows2 = store.conn.execute(
        "SELECT a.track_id tid, g.name name FROM assignments a JOIN genres g ON g.id=a.genre_id "
        "WHERE a.status='confirmed' AND a.genre_id IS NOT NULL").fetchall()
    lab = {}
    for r in list(rows) + list(rows2):
        nm = (r["name"] or "").strip().lower()
        if nm:
            lab[r["tid"]] = nm
    cnt = collections.Counter(lab.values())
    keep = {n for n, c in cnt.items() if c >= MIN_PER_CLASS}
    return {t: n for t, n in lab.items() if n in keep}


def extract_windows(store, tids, force=False):
    """Per-window MuQ embeddings for the given tracks, cached to an npz."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        have = set(z["tids"].tolist())
        if set(tids) <= have:
            return z["tids"], z["X"], z["widx"]
    from mgc.audio.decode import load_windows
    from mgc.embed import get_embedder
    emb = get_embedder("muq")
    X, out_t, widx = [], [], []
    for i, tid in enumerate(tids):
        t = store.get_track(tid)
        if t is None or not os.path.exists(t.path):
            continue
        try:
            wins = load_windows(t.path, emb.sample_rate, 5.0, 5.0, 24)
            for j, w in enumerate(wins):
                v = emb.embed(w, emb.sample_rate)
                X.append(np.asarray(v, np.float32).ravel())
                out_t.append(tid)
                widx.append(j)
        except Exception as e:
            print(f"  skip {tid}: {e}")
        print(f"  windows {i + 1}/{len(tids)}", end="\r", flush=True)
    tids_a = np.array(out_t)
    X_a = np.array(X, np.float32)
    widx_a = np.array(widx)
    np.savez_compressed(CACHE, tids=tids_a, X=X_a, widx=widx_a)
    print(f"\n  cached {X_a.shape} to {CACHE}")
    return tids_a, X_a, widx_a


class Head:
    """Small MLP projection head trained with class prototypes (cosine classifier).

    Architecture kept deliberately small (~600k params) for ~100 labelled tracks;
    dropout + label smoothing + early stopping keep it from memorising.
    """

    def __init__(self, in_dim, n_cls, emb_dim=128, seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, emb_dim),
        ).to(self.dev)
        self.cls = nn.Linear(emb_dim, n_cls, bias=False).to(self.dev)  # cosine head
        self.scale = 12.0

    def _logits(self, z):
        import torch.nn.functional as F
        z = F.normalize(z, dim=1)
        w = F.normalize(self.cls.weight, dim=1)
        return self.scale * (z @ w.T)

    def fit(self, X, y, Xva=None, yva=None, epochs=120, lr=1e-3):
        import torch
        import torch.nn.functional as F
        opt = torch.optim.AdamW(list(self.net.parameters()) + list(self.cls.parameters()),
                                lr=lr, weight_decay=1e-4)
        Xt = torch.tensor(X, device=self.dev)
        yt = torch.tensor(y, device=self.dev)
        best, best_acc, patience = None, -1.0, 0
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), device=self.dev)
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                loss = F.cross_entropy(self._logits(self.net(Xt[idx])), yt[idx],
                                       label_smoothing=0.1)
                loss.backward()
                opt.step()
            if Xva is not None and ep % 5 == 4:
                acc = self.eval_acc(Xva, yva)
                if acc > best_acc:
                    best_acc, patience = acc, 0
                    best = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 6:
                        break
        if best is not None:
            self.net.load_state_dict(best)
        return self

    def transform(self, X):
        import torch
        import torch.nn.functional as F
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(np.asarray(X, np.float32), device=self.dev))
            return F.normalize(z, dim=1).cpu().numpy()

    def eval_acc(self, X, y):
        import torch
        self.net.eval()
        with torch.no_grad():
            logits = self._logits(self.net(torch.tensor(X, device=self.dev)))
            return float((logits.argmax(1) == torch.tensor(y, device=self.dev)).float().mean())


def proto_predict(Ztr, ytr, Zte):
    """Nearest class prototype (mean, L2-normalised) by cosine."""
    cls = sorted(set(ytr))
    C = np.array([Ztr[np.array(ytr) == c].mean(0) for c in cls])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    Zn = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
    return [cls[int(np.argmax(C @ z))] for z in Zn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    from mgc.store import Store
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedGroupKFold

    store = Store.open(args.db)
    lab = load_labels(store)
    print(f"labelled tracks (classes with >={MIN_PER_CLASS}): {len(lab)} "
          f"across {len(set(lab.values()))} subgenres")

    tids = sorted(lab)
    wt, WX, _ = extract_windows(store, tids, force=args.force_extract)
    mask = np.isin(wt, tids)
    wt, WX = wt[mask], WX[mask]
    print(f"window samples: {WX.shape}")

    ids, mat = store.load_matrix("muq")
    idx = {t: i for i, t in enumerate(ids)}
    Xmean = np.array([mat[idx[t]] for t in tids], np.float64)
    Xmean = Xmean / (np.linalg.norm(Xmean, axis=1, keepdims=True) + 1e-9)

    classes = sorted(set(lab.values()))
    c2i = {c: i for i, c in enumerate(classes)}
    y_tr_track = np.array([c2i[lab[t]] for t in tids])
    yw = np.array([c2i[lab[t]] for t in wt])
    WXn = WX / (np.linalg.norm(WX, axis=1, keepdims=True) + 1e-9)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    lda_acc, net_acc = [], []
    for fold, (tr_i, te_i) in enumerate(skf.split(Xmean, y_tr_track, groups=np.arange(len(tids)))):
        tr_t = set(np.array(tids)[tr_i])
        te_t = np.array(tids)[te_i]
        # --- baseline: LDA on mean embeddings ---
        lda = LinearDiscriminantAnalysis(
            n_components=min(len(classes) - 1, 40), solver="eigen", shrinkage="auto")
        lda.fit(Xmean[tr_i], y_tr_track[tr_i])
        pred = proto_predict(lda.transform(Xmean[tr_i]), y_tr_track[tr_i].tolist(),
                             lda.transform(Xmean[te_i]))
        lda_acc.append(float(np.mean(np.array(pred) == y_tr_track[te_i])))
        # --- ours: neural head on windows, val = 15% of train tracks ---
        rng = np.random.RandomState(fold)
        tr_list = list(tr_t)
        rng.shuffle(tr_list)
        va_t = set(tr_list[:max(3, len(tr_list) // 7)])
        fit_t = tr_t - va_t
        wm_fit = np.isin(wt, list(fit_t))
        wm_va = np.isin(wt, list(va_t))
        head = Head(WX.shape[1], len(classes), seed=fold)
        head.fit(WXn[wm_fit], yw[wm_fit], WXn[wm_va], yw[wm_va])
        Ztr = head.transform(Xmean[tr_i])
        Zte = head.transform(Xmean[te_i])
        pred = proto_predict(Ztr, y_tr_track[tr_i].tolist(), Zte)
        net_acc.append(float(np.mean(np.array(pred) == y_tr_track[te_i])))
        print(f"fold {fold + 1}: LDA={lda_acc[-1]:.3f}  NET={net_acc[-1]:.3f}")

    print(f"\nBASELINE  LDA on means : {np.mean(lda_acc):.3f}")
    print(f"OURS      window-trained head: {np.mean(net_acc):.3f}")

    if np.mean(net_acc) > np.mean(lda_acc) + 0.01:
        print("\nnet wins -> training final head on ALL labels + saving 'learned' space")
        head = Head(WX.shape[1], len(classes), seed=42)
        rng = np.random.RandomState(42)
        all_t = list(set(tids))
        rng.shuffle(all_t)
        va_t = set(all_t[:max(3, len(all_t) // 7)])
        wm_fit = np.isin(wt, [t for t in all_t if t not in va_t])
        wm_va = np.isin(wt, list(va_t))
        head.fit(WXn[wm_fit], yw[wm_fit], WXn[wm_va], yw[wm_va])
        M = mat.astype(np.float64)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        Z = head.transform(M)
        for tid, z in zip(ids, Z):
            store.save_embedding(tid, "learned", z.astype(np.float32))
        import torch
        torch.save({"state": head.net.state_dict(), "classes": classes},
                   os.path.join(os.path.dirname(CACHE), "musik_head.pt"))
        print(f"saved 'learned' space for {len(ids)} tracks + weights")
    else:
        print("\nnet does NOT beat LDA -> keeping LDA (honest call)")


if __name__ == "__main__":
    main()
