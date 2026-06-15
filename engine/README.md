# mgc — Music Genre Classifier Engine (Phase 1)

Embedding-based music genre/subgenre classifier for a DJ library. You define
genres **by example** (point at a few reference tracks — no model retraining),
it bulk-classifies a folder of audio, finds **sound-alike** tracks, and writes
**Rekordbox-readable genre tags** plus a `Genre/Subgenre` folder tree — with
dry-run previews and one-click undo.

> **Status:** Phase 1 engine — **working and fully tested** (103 tests incl. an
> end-to-end pipeline test). Phase 2 (the Tauri desktop app + visual similarity
> browser) wraps this same engine and is specced separately. The engine runs
> today on a dependency-free **baseline** embedder; the high-accuracy music
> models (Discogs/MERT/CLAP) are implemented behind an optional install.

## How it works

```
folder ─▶ scan (hash + read tags) ─▶ embed (cache by hash) ─┬▶ classify (centroid + zero-shot)
                                                            ├▶ cluster (sound-alike groups)
                                                            └▶ similar (nearest neighbours)
   review / confirm ──(active learning: confirmations become examples)──┐
        └─▶ write genre tags (mutagen) + build Genre/Subgenre folders ───┘  (dry-run + undo)
```

The SQLite DB is the source of truth — files are touched only when you accept,
and every change is reversible.

## Install

Requires **Python 3.13** (3.11+ works; 3.13 has the best wheel coverage).

```bash
cd engine
py -3.13 -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell/cmd);  source .venv/bin/activate on macOS/Linux
pip install -e .                 # light deps: numpy, soundfile, scikit-learn, mutagen, typer
```

That's enough to run the whole pipeline on the **baseline** embedder.

### High-accuracy music models (recommended for real electronic subgenres)

```bash
pip install -e ".[models]"       # transformers (MERT), laion-clap, librosa, torch
```

- **GPU (your RTX 5080):** install the CUDA build of PyTorch that matches your
  driver, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (Blackwell needs a recent CUDA 12.8+ wheel). Everything auto-uses the GPU.
- **Discogs-EffNet** (best for electronic subgenres) runs via **Essentia**.
  Essentia's Python wheels are solid on Linux/macOS but flaky on Windows — if
  `pip install essentia-tensorflow` fails on Windows, use WSL2 for the Discogs
  backend, or use **MERT/CLAP** (which install cleanly on Windows) until then.
  This is the one model-packaging caveat from the design.

## Quickstart — cold start on your library

Nothing labeled yet? That's the designed path.

```bash
# 1. point it at your library
mgc init --library "D:/DJ/Unsorted" --db mylib.sqlite --model baseline
mgc scan                          # register tracks (hash + existing tags)
mgc embed                         # embed once, cached by content hash

# 2. (optional) seed the 2,600+ genre RateYourMusic taxonomy (electronic-rich)
mgc seed-taxonomy "../research/repos/joeseesun--music-genre-finder/skill-source/music-genre-finder/references"

# 3. teach it your genres BY EXAMPLE
mgc tracks                        # list track ids
mgc genres --like house           # find a parent id (if you seeded the taxonomy)
mgc add-genre "Tech House" -e 12,40,77 --parent 215 --level subgenre

# 4. classify, review, confirm (active learning)
mgc suggest                       # classify every track against your genres
mgc review                        # lowest-confidence first — the ones to check
mgc confirm 88 215                # accept track 88 as genre 215 (becomes a new example)

# 5. explore
mgc similar 12                    # tracks that sound like track 12
mgc cluster                       # auto group sound-alikes

# 6. apply (always dry-run first)
mgc write-tags                    # preview tag writes
mgc write-tags --apply            # write the subgenre into each file's genre tag
mgc organize                      # preview the Genre/Subgenre folder plan
mgc organize --apply              # build the folder tree (copy by default)
mgc undo                          # reverse the last tag writes + file moves
```

Set `organize_mode` to `move` and `organize_root` in `mgc.config.json` (or just
edit the JSON) to relocate instead of copy.

## Choosing a model

| Model | Install | Best for | Notes |
|-------|---------|----------|-------|
| `baseline` | none | smoke-testing the pipeline | pure-numpy spectral features; rough, not for real subgenre accuracy |
| `discogs` | `[models]` + Essentia | **electronic subgenres** | Discogs-EffNet + `genre_discogs400`; zero-shot labels out of the box |
| `mert` | `[models]` | strong general music embeddings | great for clustering / sound-alike |
| `clap` | `[models]` | define a genre **by name/description** | joint audio+text space (zero-shot by text) |

Pick one **active** model at a time (`--model`); embeddings are cached per model.
The validation step (below) is how you decide which wins on *your* library.

## Validate on your own library

```python
from mgc.config import Config
from mgc.api.service import Engine
from mgc.eval.validate import project_embeddings, accuracy_report

e = Engine(Config.load("mgc.config.json"))
ids, coords = e.project(method="pca")        # 2-D projection to eyeball separation (UMAP if installed)
# hand-label a subset {track_id: "Tech House", ...} then:
# report = accuracy_report(e.store, labeled, e.model)  -> top1/top3 + per-genre precision/recall
```

A `notebooks/` UMAP/t-SNE walkthrough is the Phase-2-adjacent next step.

## Test

```bash
.venv\Scripts\python -m pytest        # 103 tests, incl. end-to-end pipeline
```

The suite needs **no** heavy models — it runs on the baseline embedder and stubs.

## Layout

```
mgc/
  types.py config.py store/   # foundation contract (data model + SQLite)
  ingest/ audio/ embed/       # scan → decode → embed (+cache)
  taxonomy/ registry/         # RYM seed; by-example centroids
  classify/ cluster/ similarity/   # suggest, group, sound-alike
  actions/                    # mutagen tags + folder organize (dry-run/undo)
  eval/ api/ cli.py           # validation, Engine facade, CLI
```

## Not done yet (by design)

- **Model-quality validation on the real 1.5k-track library + GPU** — needs your
  music and the `[models]` install; the engine is ready, run step 5 above.
- **Phase 2:** the Tauri desktop app, visual similarity map, drag-to-organize
  review UI — wraps this engine as a sidecar (separate spec).
