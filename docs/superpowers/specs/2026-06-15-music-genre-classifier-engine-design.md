# Design Spec — Music Genre Classifier: Phase 1 Engine

- **Date:** 2026-06-15
- **Status:** Approved (design), pending implementation plan
- **Author:** brainstormed with user (miguelito.villax)
- **Scope of this spec:** Phase 1 only — the standalone Python engine. Phase 2 (Tauri desktop app + UI) is intentionally deferred to its own spec.

---

## 1. Context & Problem

The user is a DJ with a **~1,500-track, electronic-heavy library** (house, techno, trance + their subgenres) where **none of the tracks currently have genre metadata**. They want a tool that auto-classifies music into fine-grained subgenres, lets them define/refine genres by example, organizes a dump of files, and writes genre metadata that **Rekordbox** can read.

Hardware: **RTX 5080 GPU + strong CPU**, Windows primary, "all OS ready" desired.

A prior deep-dive of 5 reference repos (`research/FINDINGS.md`) established that classic fixed-label supervised classifiers (GTZAN CNN/LSTM/RandomForest) **cannot** satisfy "define a new genre by example without retraining," and that the correct architecture is **frozen pretrained audio embeddings + few-shot similarity**, with a **music-trained** model (Discogs-based) for the electronic focus.

### Requirements (from the user)
1. Auto-classify audio files by genre.
2. Fine-grained **subgenres**, not just broad genres.
3. **Define new custom genres BY EXAMPLE** ("these tracks are genre X, classify the rest like them") — no retraining.
4. **Bulk** process a dumped folder.
5. **Write the accepted genre into file metadata** (Rekordbox-readable) **and** build a genre folder tree.
6. Fast and high quality.
7. (Added during brainstorm) **"Sound-alike" discovery** — see which tracks are closely related sound-wise.
8. (Added) **Auto-organize an unsorted dump** into a `Genre/Subgenre` folder structure.
9. (Added) A **hierarchical taxonomy**: broad "subset" tier → genres → subgenres.

### Key decisions locked during brainstorm
- **Form factor:** Tauri desktop app, cross-OS, fast/responsive. (Built in Phase 2.)
- **Engine path:** **C → A** — build & validate a standalone Python engine first (Phase 1), then wrap it as a Python **sidecar** inside Tauri (Phase 2). Pure-native Rust/ONNX (option B) is a possible later distribution optimization, not now.
- **Output:** write genre **tags** AND build a **folder tree**. Tags are the primary output (Rekordbox imports them).
- **Folder tree:** **2 levels — `Genre/Subgenre`** (e.g. `House/Tech House/`). The "subset" tier exists inside the taxonomy for classification/browsing but is **not** materialized as a folder level.
- **Genre tag value:** the **specific subgenre only**, in the single genre field (Rekordbox has one genre field, no genre/subgenre split).
- **Taxonomy:** seed from the **RateYourMusic ~5,900-genre tree** (already cloned in `research/repos/joeseesun--music-genre-finder`), extend with custom subgenres **by example**.
- **Onboarding:** cold start (nothing labeled) → zero-shot first pass + clustering + review/relabel loop → confirmations become example centroids (active learning).
- **Scale decisions:** at 1.5k tracks, brute-force in-RAM cosine similarity; **no FAISS/vector-DB** (YAGNI).

---

## 2. Scope

### In scope (Phase 1 — this spec)
A GPU-accelerated Python engine, driven by a **CLI** + a **validation notebook**, exposing a **service layer** that Phase 2's Tauri sidecar will call verbatim. It must take the folder of untagged tracks through: ingest → embed → suggest → cluster → review/confirm → write tags + build folders, with full dry-run/undo safety.

### Deferred (Phase 2 — separate spec)
- Tauri (Rust + web) shell and packaging/installer.
- Visual similarity browser / similarity map UI.
- Interactive review/confirm GUI, drag-to-organize.
- Sidecar packaging (PyInstaller) and cross-OS distribution.

### Non-goals (Phase 1)
- Online metadata APIs (MusicBrainz/Discogs/Spotify) — optional future enrichment, not now.
- ANN vector index (FAISS) — unnecessary at 1.5k tracks.
- Rekordbox XML / MyTag native integration — standard genre tag is sufficient for v1.
- Retraining/fine-tuning any model.

---

## 3. Architecture

### Module boundaries
```
engine/
  ingest/      scan folder, content-hash, read existing tags (mutagen), build manifest, dedup
  audio/       decode → mono → resample → window across whole track (torchaudio/soundfile/librosa)
  embed/       Embedder interface + backends (Discogs-EffNet / MERT / CLAP); GPU batching; hash-keyed cache
  taxonomy/    ingest RYM JSON → normalized genre tree (id, name, parent, level, description); custom genres
  registry/    genre centroids + exemplars; seed by-example (ref tracks) and by-name (text via CLAP)
  classify/    zero-shot model output + centroid similarity → ranked suggestions + confidence + roll-up
  cluster/     HDBSCAN/k-means grouping → candidate genre per cluster (bulk labeling)
  similarity/  "tracks like this" nearest-neighbor (brute-force cosine in RAM)
  actions/     tag writer (mutagen) + folder organizer (copy/move); dry-run; undo via actions_log
  store/       SQLite schema + vector store + state/assignments/action log
  api/         service-layer functions + CLI now; reused by Tauri sidecar in Phase 2
  eval/        validation: t-SNE/UMAP + top-1/top-3 accuracy + per-genre precision/recall
```

Each module has one purpose, communicates through explicit function interfaces, and is testable in isolation.

### Data flow
`folder → ingest (manifest) → audio decode → embed (cached by content hash) → [zero-shot classify + HDBSCAN cluster] → ranked suggestions → user review/confirm → registry centroids update → reclassify remaining → accept → actions (tags + folders) with dry-run + undo`

### Cross-cutting principles
- **SQLite DB is the source of truth.** Files are only touched on explicit accept; tags/folders can be re-derived anytime.
- **Embeddings cached by content hash.** First pass is the only slow step; everything after is instant and re-runs are incremental/idempotent.
- **Service layer is UI-agnostic.** The CLI is a thin wrapper; Phase 2 Tauri reuses the same functions over local IPC.

---

## 4. ML Core

### Embedding/model shortlist (A/B tested in the spike on the user's real tracks)
- **Essentia Discogs-EffNet** (`discogs-effnet`) — primary **cold-start** workhorse. Ships a `genre_discogs400` head outputting **400 Discogs styles** (electronic-rich: deep-house, tech-house, acid, psytrance, etc.), plus an embedding vector. Gives real electronic-subgenre guesses on day one with zero labels.
- **MERT** (`m-a-p/MERT-v1-330M`) — strongest self-supervised **music** representation; best for clustering, "sound-alike" similarity, and by-example centroids.
- **laion-CLAP (music checkpoint)** — joint audio+text space; enables defining a custom subgenre **by name/description** (zero-shot) in addition to by example.

**Expected resolution:** Discogs-EffNet for zero-shot first-pass labels + one strong embedding (MERT or CLAP) for similarity/clustering/centroids. The validation notebook decides the winner on the user's library (t-SNE/UMAP separation + top-1/top-3 accuracy on a hand-labeled subset).

### Embedding extraction
Decode the **whole track**, window across all of it (explicitly NOT first-3s-only), run model on GPU in batches, mean-pool → **L2-normalized** vector. Cache keyed by content hash.

### Classification (two cooperating signals)
1. **Zero-shot** model output (Discogs-400 styles and/or CLAP text-similarity to RYM genre names/descriptions), mapped to the taxonomy → ranked candidates + confidence. Powers the cold start.
2. **Centroid similarity** — cosine/kNN to genre centroids built from the user's confirmed exemplars.
Below a per-genre confidence threshold → "unknown / needs review."
Adding a custom subgenre = embed a few examples → store centroid → instant, no retraining.

### Clustering
HDBSCAN over embeddings to find natural sound-alike groups in the dump (no need to choose a cluster count; isolates oddballs as noise). Each cluster gets a suggested label from its majority zero-shot guess, enabling bulk confirmation. (k-means available as a fixed-count fallback.)

### Similarity
Brute-force cosine over the ~1,500 in-RAM vectors → "show tracks like this" returns top-N neighbors instantly. Directly serves requirement #7.

### Active learning & hierarchy
- Confirmations update centroids; remaining tracks are reclassified with improved accuracy.
- The engine resurfaces the **lowest-confidence** tracks first for review ("prompt me some music to edit").
- Tracks are classified at **subgenre** level and rolled up to parent **genre** (and subset) for the folder tree and any roll-up views.

---

## 5. Data Model (SQLite)

```
tracks(id, path, content_hash, format, duration, sample_rate, existing_tags, status, added_at)
embeddings(track_id, model, vector, dims)              # BLOBs; loaded into RAM at 1.5k scale
genres(id, name, parent_id, level, source, description, threshold, centroid)
                                                        # level: subset|genre|subgenre ; source: seed|custom
exemplars(genre_id, track_id)                           # user-confirmed examples per genre
assignments(track_id, genre_id, confidence, method, status, decided_at)
                                                        # method: zeroshot|centroid|manual ; status: suggested|confirmed|rejected
clusters(id, run_id, suggested_genre_id)
cluster_members(cluster_id, track_id)
actions_log(id, type, track_id, from_value, to_value, undo_token, status, ts)
                                                        # type: tag_write|copy|move ; powers undo
```

Vector storage: BLOB per embedding (or a single numpy/parquet store) loaded into memory; no external vector DB.

---

## 6. Output Actions & Safety

### Tag writing (mutagen)
- Writes the **specific subgenre** into the single genre field per format: MP3 `TCON`, FLAC/OGG Vorbis `GENRE`, M4A `©gen`.
- Preserves all other existing tags untouched.
- Optimized for Rekordbox (single genre field).

### Folder organizer
- Mirrors taxonomy as **`LibraryRoot/<Genre>/<Subgenre>/track.ext`** (2 levels).
- **Copy by default** (originals safe); **move** optional.
- Windows-illegal characters (`\ / : * ? " < > |`) sanitized in folder names.
- Name collisions get a suffix; never overwrite.

### Safety model (non-negotiable)
- Files touched only on explicit accept.
- Every action runs a **dry-run preview** first (full source→dest / old→new plan).
- Every change recorded in `actions_log` with the previous value.
- **One-click undo** restores tags / reverses copies/moves.

---

## 7. Phase 1 Interface

### Service layer
UI-agnostic Python functions = the engine API. Phase 2's Tauri sidecar exposes these over local IPC unchanged.

### CLI (thin wrapper)
`scan` · `embed` · `suggest` · `cluster` · `similar <track>` · `review` (surfaces lowest-confidence tracks) · `label` · `write-tags --dry-run` · `organize --copy --dry-run` · `undo`

### Validation notebook
t-SNE/UMAP visualization, embedding-model A/B comparison, and top-1/top-3 accuracy + per-genre precision/recall on a hand-labeled subset of the user's own tracks. (Measures quality; not a pass/fail gate.)

### Config file
Library paths, model choice, confidence thresholds, copy vs move, folder root, tag mapping.

---

## 8. Error Handling
- Corrupt/undecodable files → quarantined + reported, never crash the batch.
- Unsupported formats → skipped with a note.
- GPU OOM → batch-size backoff → CPU fallback.
- Tag-write failure (locked/read-only/exotic) → logged, skipped, marked in `actions_log`.
- Folder collisions → suffixed, never overwritten.
- Whole pipeline idempotent via content-hash cache (safe re-runs).

---

## 9. Testing Strategy
- **Unit tests** per module: content hashing, taxonomy parse, centroid math, threshold logic, **tag round-trip** (write → mutagen read-back), organizer dry-run plan correctness, undo restoration.
- **Integration test:** end-to-end on a tiny fixture folder of CC-licensed/sample audio (**NOT GTZAN**) → suggest → write tags → verify via mutagen → organize (copy) → verify tree → undo → verify restored.
- **Real files/DB at boundaries** — no mocking the tag/file layer.
- **Eval harness** is separate from the test suite — it measures classification quality on real data.

---

## 10. Tech Stack
Python 3.11+ · PyTorch (CUDA, for the RTX 5080) · Essentia + `transformers` (MERT) + `laion-clap` (model A/B) · `soundfile`/`torchaudio`/`librosa` (decode) · `mutagen` (tags) · `scikit-learn` + `hdbscan` (clustering) · stdlib `sqlite3` · `numpy` · `typer` (CLI) · `pytest`.

---

## 11. Risks & Open Questions
- **Electronic subgenre separation** is the central quality risk — telling deep-house from tech-house is subtle. Mitigation: Discogs-trained model + multiple exemplars per subgenre + the spike validates before app investment. The C→A phasing exists specifically to surface this early.
- **Model packaging size** (Phase 2): torch + models is multi-GB; affects the Tauri installer. A Phase 2 concern (possible ONNX/quantization later).
- **Taxonomy normalization:** RYM JSON has some Chinese fields; needs normalization to English canonical names during ingest.
- **Which embedding wins** is resolved empirically in the spike, not assumed here.

---

## 12. Success Criteria (Phase 1)
1. Engine ingests the 1.5k untagged tracks and produces a cached embedding per track (first pass in minutes on the 5080).
2. Cold-start zero-shot pass + clustering produce reviewable subgenre suggestions with confidence.
3. User can confirm/relabel; confirmations create centroids that measurably improve subsequent classification (active learning demonstrated).
4. "Tracks like this" similarity query returns musically sensible neighbors.
5. Accept → genre tags written (Rekordbox-readable, verified by read-back) and `Genre/Subgenre` folder tree built, both with working dry-run and undo.
6. Validation notebook reports the chosen model's top-1/top-3 accuracy on a hand-labeled subset of the user's own library.
