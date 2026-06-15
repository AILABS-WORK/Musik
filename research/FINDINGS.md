# Music Genre Classification — Reference Repo Deep-Dive & Architecture Recommendation

_Analysis date: 2026-06-15. Five repos cloned to `research/repos/`, each deep-dived at source-code level by a dedicated agent, then synthesized against the project goal._

## The goal (recap)
A fast, accurate tool that: (1) auto-classifies audio by genre, (2) supports fine-grained **sub-genres**, (3) lets you **define new custom genres BY EXAMPLE** ("these tracks are genre X, classify the rest like them"), (4) **bulk** processes a dumped folder, (5) optionally **writes the genre into file metadata** after you accept, (6) is fast and high quality.

## TL;DR
**None of the five repos is a usable building block** for this goal — but together they make the right architecture obvious. Three are stale academic GTZAN classifiers with a **fixed softmax head** (can't add a genre without full retraining — structurally fails requirement #3). Two never touch audio at all (one is an LLM text lookup, one matches song titles to YouTube channels). The decisive pivot: **don't train a classifier — use a frozen pretrained audio EMBEDDING model + few-shot nearest-centroid/kNN similarity.** Then "add a genre" becomes "store a centroid from your example tracks" — zero retraining, instant, and naturally fine-grained.

## Comparison matrix

| Repo | What it really is | Approach | Sub-genres | Custom-by-example (#3) | Bulk (#4) | Metadata write (#5) | Score |
|---|---|---|---|---|---|---|---|
| **joeseesun/music-genre-finder** | Claude **skill** (no audio) | LLM matches your *text* to a 5,947-genre RateYourMusic JSON taxonomy | Great **vocabulary** (5,947 genres) but it's a dictionary, not a classifier | No — no audio pathway | No | No | **3/10** |
| **ruijramos/Identify-Music-Genre** | Uni coursework (Py+R) | librosa 26 *mean* features → R kNN / NaiveBayes / RandomForest, **retrained every run** | No — 10 fixed GTZAN genres | No | No | No | **3/10** |
| **ruohoruotsi/LSTM** | Teaching project | 33-dim librosa seq (first ~3s) → 2-layer **LSTM** → Dense(8) softmax | No — 8 fixed genres | No (closed-set softmax) | No (1 file/call) | No | **3/10** |
| **mlachmish/MusicGenreClassification** | 2016 academic, Py2/TF1 | 128-bin mel-spectrogram → **CNN** → 10-class softmax | No — 10 fixed | No (no embeddings, no shipped weights, no inference script) | No | No | **2/10** |
| **lilgallon/music-genre-finder** | Abandoned 2019 alpha (no audio) | Searches YouTube for the **song title**, matches channel ID → genre JSON | Niche **labels** only (~16 EDM/hip-hop) | No (add channel IDs by hand; has a scoring bug) | No (3 hardcoded strings; quota-capped) | No | **2/10** |

### Measured accuracy (all on the easy, leaky GTZAN 8–10 broad genres)
- ruijramos RandomForest: ~60–66% · NaiveBayes ~37% · kNN ~28–31%
- ruohoruotsi LSTM: ~68% top-1
- mlachmish mel-CNN: ~47% · MFCC-CNN: ~17%

These are a **floor**, not a target — and they're on broad genres far easier than the sub-genres you actually want.

## The core insight (why classic classifiers can't do what you want)
The three GTZAN classifiers all end in a **fixed-width output layer** (`Dense(8)`, `Dense(10)`, or 3 R models over a 10-label table). The number of genres and the decision boundaries are baked into the trained weights. To add even one genre you must collect labeled audio, re-extract features for the whole corpus, change the output dimension, and **retrain from scratch**. There is no way for "here are 5 tracks that define genre X, now classify the rest" to work — the model has no output slot for X and can't make one at inference time.

**Embeddings invert this.** A frozen pretrained audio model (CLAP, OpenL3, PANNs/CNN14, MusiCNN) maps any track to a vector where musically similar tracks sit close together. A "genre" becomes a **centroid (or a few exemplars) of your example tracks' embeddings**. Classification = cosine similarity / kNN to the nearest centroid. Adding a custom sub-genre = embed your examples, store the centroid, label it. **No retraining, instant, naturally fine-grained.** This is exactly the capability missing from all five repos.

## Recommended architecture (buildable, 5 stages)

1. **Ingestion** — recursive folder walk (mp3/flac/wav/m4a/ogg); read existing tags with **mutagen**; persist a manifest (path, hash, mtime) for incremental re-runs.
2. **Embedding extraction (the heart)** — decode once, downmix mono, resample; run through **CLAP** (primary — joint audio+text space enables *both* by-example *and* by-name/zero-shot genre seeding), with PANNs/CNN14 or OpenL3 as audio-only fallback. Embed windows across the **whole track** (fixing the repos' first-3s mistake), mean-pool to one L2-normalized vector. **Cache every embedding to Parquet/SQLite keyed by file hash** → after the first pass, all re-classification is instant.
3. **Example-based custom sub-genre matching** — a genre registry where each genre = `{name, optional taxonomy id, centroid, exemplars, threshold}`. Seed **by example** (folder of reference tracks → centroid) or **by name** (CLAP text-encode the genre description, zero-shot). Classify by cosine similarity → top-k + confidence; below threshold → "unknown / needs review." Adding/editing a genre re-runs only this step against cached embeddings.
4. **Human confirm** — review list (top-1 + top-3 + confidence + any external-API corroboration); bulk-accept above a confidence cutoff; accepted pairs become new exemplars (active-learning loop). Optional: route only low-confidence cases through an LLM (Claude skill) to canonicalize the label and explain it.
5. **Metadata write-back** — only for accepted suggestions, write with **mutagen** (ID3 `TCON` for MP3, Vorbis `GENRE` for FLAC/OGG, MP4 `©gen` for M4A), preserving existing tags, with **dry-run + backup/undo**. (Net-new — no repo does this.)

### What to actually reuse from the repos
- **joeseesun's `references/*.json`** — the 5,947-genre RateYourMusic hierarchy (with parent links + descriptions) as your canonical **genre vocabulary / tag values** AND as **CLAP text-anchors** for zero-shot seeding. (Some fields are Chinese — normalize to English.) Its 3-tier progressive-disclosure design + skill packaging is a good template for the optional LLM layer.
- **lilgallon's `genres.json`** — the niche sub-genre **label list** (Neurofunk, Liquid DnB, Chillstep, Trap, Phonk, Lofi, Emo rap…) as starter custom genres. (Discard the YouTube channel IDs.)
- **ruijramos / ruohoruotsi librosa recipes** — as a CPU-only **fallback** feature set if a user can't run torch; but fix the librosa **positional-arg calls** (break on librosa ≥0.10) and embed the whole track.
- **The feature-cache idea** — generalize to an embedding cache keyed by file hash.
- **mlachmish's finding** — mel-spectrogram beats raw MFCC (47% vs 17%); favors mel/pretrained spectrogram embeddings.

### Anti-patterns to deliberately avoid
Fixed-width softmax heads · filename/path-derived labels · first-3s-only analysis · re-opening the audio file per window · retraining on every prediction · keying genre off web/uploader metadata.

## Suggested next steps
1. **Embedding spike (an afternoon):** run laion-CLAP (+ OpenL3 baseline) over ~50 of *your own* tracks across 5–6 target sub-genres; mean-pool whole-track embeddings; t-SNE/UMAP to confirm the sub-genres separate.
2. **Few-shot loop:** 3 custom sub-genres × 5 example tracks → centroids → classify a held-out set → measure top-1/top-3 + threshold behavior.
3. **Ingestion + cache layer:** folder walk, mutagen read, decode-once batched embedding, Parquet/SQLite cache; benchmark CPU vs GPU.
4. **Genre registry** (by-example + by-name seed modes).
5. **Confirm → mutagen write-back** (dry-run, backup, undo).
6. **Eval harness on YOUR library** (not GTZAN): per-genre precision/recall; capture accept/override as new exemplars.
