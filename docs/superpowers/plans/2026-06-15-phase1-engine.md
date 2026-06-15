# Phase 1 Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone Python engine (`mgc`) that ingests a folder of untagged audio, embeds each track, suggests fine-grained subgenres (zero-shot + by-example centroids), clusters sound-alikes, finds similar tracks, and writes Rekordbox-readable genre tags + a `Genre/Subgenre` folder tree — all with dry-run + undo.

**Architecture:** Pluggable `Embedder` backends (a dependency-free numpy baseline that always runs, plus lazy heavy backends: Essentia Discogs-EffNet, MERT, CLAP). SQLite is the source of truth; embeddings cache by content hash. A UI-agnostic service layer is wrapped by a Typer CLI now and reused by the Tauri sidecar in Phase 2.

**Tech Stack:** Python 3.13, numpy, soundfile (decode), scikit-learn (clustering/metrics), mutagen (tags), typer (CLI), pytest. Heavy/optional extras (lazy): torch, transformers (MERT), laion-clap, essentia (Discogs-EffNet), librosa.

---

## File Structure (the `engine/` project)

```
engine/
  pyproject.toml                 # project + deps + extras (heavy models behind [models])
  README.md                      # setup, usage on a real library, validation
  mgc/
    __init__.py
    config.py                    # Config dataclass + load/save (TOML); paths, active_model, thresholds, copy/move
    types.py                     # CONTRACT: dataclasses + Embedder ABC + Suggestion/Cluster/ActionRecord
    store/
      __init__.py
      schema.py                  # SQLite DDL (all tables from the spec)
      db.py                      # Store class: connection, migrate, CRUD, vector blob (de)serialization
    ingest/__init__.py + scanner.py   # walk folder, content-hash, read existing tags, upsert tracks
    audio/__init__.py + decode.py     # decode→mono→resample→whole-track windows (soundfile; librosa if present)
    embed/
      __init__.py                # get_embedder(name) factory
      base.py                    # shared pooling/L2-normalize helpers
      baseline.py                # numpy-only spectral/MFCC embedder (always works)
      discogs.py / mert.py / clap.py   # lazy heavy backends (import inside __init__/embed)
      cache.py                   # embed_track(store, embedder, track) with hash-keyed cache
    taxonomy/__init__.py + rym.py     # ingest RYM JSON → normalized genre tree → seed `genres`
    registry/__init__.py + centroids.py  # by-example + by-name centroids, exemplar add, recompute
    classify/__init__.py + classifier.py # zero-shot + centroid similarity → ranked Suggestions + rollup
    cluster/__init__.py + cluster.py  # sklearn HDBSCAN/KMeans → clusters + suggested genre
    similarity/__init__.py + similar.py   # cosine top-N neighbors in RAM
    actions/
      __init__.py
      tags.py                    # mutagen write/read genre per format; round-trip; undo
      organize.py                # plan + execute copy/move tree; dry-run; collisions; undo
    api/__init__.py + service.py      # Engine facade tying modules together (UI-agnostic)
    cli.py                       # Typer CLI over service.py
    eval/__init__.py + validate.py    # t-SNE/UMAP export + top-1/top-3 accuracy on labeled subset
  tests/
    conftest.py                  # tmp library fixture: synth wav generator, in-memory/temp Store
    test_*.py                    # one per module
```

**Contract rule for parallel workers:** `mgc/types.py`, `mgc/store/` and `mgc/config.py` are the shared contract (Task 1). Module workers MUST import from them and MUST NOT modify them. Each worker creates only its own module dir + its own `tests/test_<module>.py`.

---

## Task 1: Foundation / contract (built first, single-threaded)

**Files:** Create `engine/pyproject.toml`, `engine/mgc/__init__.py`, `engine/mgc/config.py`, `engine/mgc/types.py`, `engine/mgc/store/schema.py`, `engine/mgc/store/db.py`, `engine/tests/conftest.py`, `engine/tests/test_store.py`.

Key contract types in `types.py`:
- `@dataclass Track(id, path, content_hash, fmt, duration, sample_rate, existing_tags: dict, status)`
- `@dataclass GenreNode(id, name, parent_id, level, source, description, threshold)` (level ∈ {subset,genre,subgenre}; source ∈ {seed,custom})
- `@dataclass Suggestion(track_id, genre_id, genre_name, confidence, method)` (method ∈ {zeroshot,centroid,manual})
- `@dataclass ClusterResult(cluster_id, member_track_ids, suggested_genre_id)`
- `@dataclass ActionRecord(id, type, track_id, from_value, to_value, undo_token, status, ts)`
- `class Embedder(ABC)`: `name: str`, `dims: int`, `embed(samples: np.ndarray, sr: int) -> np.ndarray` (returns 1-D float32, L2-normalized).

`store/db.py` `Store`: `open(path)`, `migrate()`, upserts/queries for every table, `save_embedding/load_embeddings(model)`, `log_action/iter_undo`. Vectors stored as float32 `np.tobytes()` blobs + `dims`.

- [ ] Write `tests/test_store.py`: migrate creates tables; upsert+get track round-trips; save/load embedding round-trips a numpy vector; log_action + undo retrieval works.
- [ ] Run `engine/.venv/Scripts/python -m pytest tests/test_store.py -v` → PASS.
- [ ] Commit.

## Task 2: `audio/decode.py`
Decode wav/flac/ogg (soundfile; mp3 via libsndfile≥1.1) → mono float32 → resample to target SR → return whole-track windows (e.g. list of N-sec frames or a generator). Use librosa if importable for resample/extra formats, else numpy/soundfile. Test on a synth wav fixture: shape, mono, SR correct; graceful error on corrupt file.

## Task 3: `embed/` (baseline + factory + cache; heavy backends lazy)
- `baseline.py`: numpy-only embedder — STFT via `np.fft`, log-mel-ish bands + MFCC (DCT), per-band mean+std → fixed vector, L2-normalized. Deterministic. `dims` fixed.
- `base.py`: pooling + normalize helpers.
- `__init__.get_embedder(name)`: `baseline` always; `discogs|mert|clap` import lazily and raise a clear "install extras [models]" error if deps missing.
- `cache.py`: `embed_track(store, embedder, track)` → decode → embed → store (skip if cached for that model+hash).
- `discogs.py/mert.py/clap.py`: real implementations with lazy imports (Essentia Discogs-EffNet `genre_discogs400` + embeddings; MERT via transformers; laion-clap with text encoder for by-name).
Test: baseline embeds a synth wav → right dims, unit norm, deterministic; cache skips second call (stub embedder counts calls).

## Task 4: `taxonomy/rym.py`
Parse the RateYourMusic JSON from `research/repos/joeseesun--music-genre-finder/skill-source/music-genre-finder/references/` (_index.json + main/ + detailed/) → normalized `GenreNode` tree (id,name,parent,level,description), English-normalized names, seed into `genres`. Provide `seed_taxonomy(store, refs_dir)`. Test on a tiny fixture JSON tree → nodes + parent links correct; idempotent re-seed.

## Task 5: `registry/centroids.py`
`add_exemplar(store, genre_id, track_id)`, `recompute_centroid(store, genre_id, model)` (mean of exemplar embeddings, L2-norm), `seed_by_name(store, genre_id, text, clap)` (CLAP text vector → centroid, `is_text_centroid=true`; requires active model = clap). Test with stub embeddings: centroid = normalized mean; updates on new exemplar.

## Task 6: `classify/classifier.py`
`suggest(store, track, model, k)` → combine (a) zero-shot scores (Discogs-400 / CLAP-text vs taxonomy) when available and (b) cosine to genre centroids → ranked `Suggestion`s + confidence; below `threshold` → "unknown". `rollup(genre_id)` → parent genre/subset. Test with synthetic centroids: nearest centroid wins; threshold yields unknown; rollup returns parents.

## Task 7: `cluster/cluster.py`
`cluster(store, model)` using `sklearn.cluster.HDBSCAN` (fallback `KMeans`) over in-RAM vectors → `ClusterResult`s; suggested genre = majority zero-shot/centroid label per cluster. Test: two well-separated synthetic blobs → 2 clusters.

## Task 8: `similarity/similar.py`
`similar(store, track_id, model, n)` → cosine over in-RAM matrix → top-N (excluding self). Test: nearest neighbor of a vector is its near-duplicate.

## Task 9: `actions/tags.py` + `actions/organize.py`
- `tags.py`: `write_genre(path, subgenre)` per format (ID3 TCON / Vorbis GENRE / MP4 ©gen) preserving other tags; `read_genre(path)`; record prior value → `actions_log`; `undo`. Test: write→read round-trip on generated mp3/flac fixtures (or skip format if writer lib unavailable); undo restores.
- `organize.py`: `plan(tracks, assignments, root, mode)` → list of (src,dest) with sanitized `Genre/Subgenre/` paths + collision suffixes; `execute(plan, dry_run)`; `undo`. Test: dry-run produces correct plan, no FS change; execute copy creates tree; undo reverses.

## Task 10: `api/service.py` + `cli.py` (integration; single-threaded)
`Engine` facade: `scan, embed_all, suggest_all, cluster, similar, review, confirm, write_tags, organize, undo` over a `Store` + active embedder. `cli.py`: Typer commands mapping 1:1. Wire everything; resolve any interface drift from parallel modules here. Add end-to-end integration test (Task 12).

## Task 11: `eval/validate.py`
`export_projection(store, model)` (UMAP if installed else PCA/t-SNE via sklearn) and `accuracy(store, labeled_csv)` → top-1/top-3 + per-genre precision/recall. Notebook `engine/notebooks/validate.ipynb` calling these. (No pass/fail test; smoke test that PCA projection runs on synthetic vectors.)

## Task 12: End-to-end integration test + README
`tests/test_e2e.py`: build a temp library of synth wavs in 2 sound-groups → scan → embed (baseline) → seed 2 genres by example → suggest → write tags (dry-run then real) → mutagen read-back asserts genre → organize copy → assert `Genre/Subgenre` tree → undo → assert restored. README: install (light + `[models]` extras), run on a real library, run validation, Essentia-on-Windows caveat.

---

## Notes
- DRY/YAGNI/TDD, frequent commits. No FAISS at 1.5k. Heavy model deps never required for the test suite (baseline + stubs only).
- Active embedding model is single + config-driven; never mix embedding spaces.
