# 🎧 Musik — your AI music hub

Drop a folder of music in and Musik figures out what everything is, lets you
build your **own** genres and subgenres **by example**, and sorts your library
into clean `Genre/Subgenre` folders with Rekordbox-ready tags — all locally, on
your GPU. It's built for DJs and deep electronic libraries where "house" isn't
enough and you actually care about deep-house vs tech-house vs minimal.

```
drag music in ─▶ auto-classify ─▶ review / relabel / make new genres ─▶ organize into your folder tree
        └─ browse sound-alikes · cluster the dump · write genre tags ─┘   (dry-run + undo on everything)
```

---

## ⚡ Get started (Windows, NVIDIA GPU)

**One-time setup** — installs everything (engine, CUDA PyTorch, the music model, the app) and drops a **Musik** shortcut on your desktop:

```powershell
# from the project folder, in PowerShell:
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Then just **double-click the Musik icon on your desktop**. The app opens and the
engine starts automatically — nothing else to run. (First launch compiles the
app once, ~1–2 min; after that it's instant.)

Prerequisites the setup assumes are installed: **Python 3.13**, **Node 18+**, and
**Rust** (for the desktop shell). The RTX 5080 / any NVIDIA GPU is used
automatically via the CUDA PyTorch build (`cu128`).

---

## 🎚️ How to use it

1. **Set your destination** — in the top bar, type the **Organized library** folder (where sorted music should go) and pick **copy** or **move**.
2. **Drag a selection of files or whole folders onto the window** → they import and auto-embed (or paste a path into *Import*). The model fingerprints each track.
3. **Teach it your genres** (first time):
   - *Genres → Create genre by example* — check a few reference tracks, name the genre.
   - or *Clusters → Find clusters → Make genre from cluster* — let it group sound-alikes for you.
   - or *Seed taxonomy* — load 2,600+ RateYourMusic genres as a starting vocabulary.
4. **Classify & review** — hit **Suggest**; every track gets a genre + confidence. Sort by confidence, **relabel inline** (click a genre cell), eyeball the **Map**, or play tracks in-app.
5. **Apply** — write genre **tags** (Rekordbox reads them) and **Organize** into `Destination/Genre/Subgenre/`. **Undo** reverses anything.

Everything is reversible: the app's database is the source of truth, files are
only touched when you accept, every change has a dry-run preview and one-click undo.

### Pick the model
- **`mert`** (recommended) — a music-trained model; sharp on fine electronic subgenres, GPU-accelerated. Set it in the top bar after `setup.ps1`.
- **`baseline`** — zero-install, rough; fine for a first look.
- **`discogs` / `clap`** — Discogs-style labels / define-a-genre-by-text. See `engine/README.md`.

---

## ✨ What it does today

- **Auto-classify** audio into fine-grained **subgenres** you define
- **Define genres BY EXAMPLE** — a few reference tracks, no retraining
- **Bulk drag-and-drop** import + **organize into `Genre/Subgenre` folders**
- **Write Rekordbox-ready genre tags**
- **Sound-alike** discovery ("tracks like this") + a 2D **similarity map**
- **Clustering** of an unsorted dump into groups you can name
- **Inline relabel + active learning** — your corrections sharpen the next pass
- Local + private; runs on your GPU; dry-run + undo everywhere

- **AI Set Builder** — describe a vibe ("light groovy house at sunset, start slow,
  build punchier, end deep & minimal") → an ordered set following that energy/BPM arc
- **Identify a track by its sound** + **Radio** (auto-advancing play-next queue)
- **Auto-sort** one-click pipeline, or work through it interactively

## 🧠 Under the hood (how the classification works)

Musik uses the modern MIR recipe: **frozen self-supervised music embeddings + a light probe**, not a brittle fixed-label classifier.

- **Embeddings** — each track is encoded by a music foundation model (**MERT** `m-a-p/MERT-v1-330M`, GPU) into a vector that captures timbre/rhythm/texture; whole-track windows are mean-pooled + L2-normalized and cached by content hash.
- **Classification by example** — a "genre" is the **centroid (or k-NN over exemplars)** of the embeddings of a few reference tracks. Classify = cosine similarity to those centroids. Adding a custom subgenre = drop in examples, no retraining. Your confirmations become new exemplars (active learning).
- **Open vocabulary** — **CLAP** (LAION) puts audio and free text in one space, enabling define-a-genre-by-name and (next) free-text attribute search.
- **Analysis** — BPM (onset-autocorrelation), musical key (Krumhansl-Schmuckler chroma), energy/danceability — numpy/scipy, no heavy deps.

### What's next (research-grounded — see [the design spec](docs/superpowers/specs/2026-06-16-music-understanding-search-design.md))
A rich per-song **"understanding" record** + **open-vocabulary search** ("give me songs with cowbells"):
- **AudioSet-527 tagging** (EfficientAT / AST) — instruments, vocals, and a literal *Cowbell* class
- **Vocal vs instrumental + gender** (Essentia heads on the Discogs-EffNet embedding), **mood** (arousal/valence), **sung-language** (Whisper on a separated vocal stem)
- **Open-vocab attribute search** (CLAP mean + per-chunk-max embeddings, prompt-ensemble, per-query calibration so you can "return ALL matches") with a router that sends known classes to the precise tagger and free text to CLAP
- **Source separation** (HTDemucs) as an opt-in "deep analysis" pass for quiet percussion / vocal technique / language
- An **LLM-over-tags "understanding compiler"** that fuses all model outputs into canonical tags + a per-song caption, and feeds an attribute-aware set-builder (Camelot key + mood continuity)
- (Honest scope: voice-type/SATB and region/dialect are **not recoverable from a mix** — we ship a coarse register *hint* and drop region.)

## 🧱 How it's built

- **Engine** (`engine/`, Python) — embeddings, by-example classification, clustering,
  BPM/key/energy analysis, set-builder, identify, tags + folder organize, a local FastAPI
  sidecar. **156 tests.** See `engine/README.md`.
- **App** (`app/`, Tauri v2 + React) — the desktop UI; the Rust shell auto-starts the
  sidecar. See `app/README.md`.
- **Research** (`research/FINDINGS.md` + `docs/superpowers/specs/`) — the deep-dives that shaped the design.

## 🔒 Privacy
Everything runs on your machine. Audio never leaves your computer (model weights
download once from Hugging Face on first use; classification is fully local).
