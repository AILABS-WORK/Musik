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

## 🛣️ Roadmap (next)

These are designed-for, not done yet — see `docs/superpowers/`:
- **Play-next / radio** — auto-suggest what to play after a track
- **Identify a track from its sound** (within your library; AcoustID for unknowns)
- **AI set builder** — describe a vibe in words ("light groovy house at sunset,
  start slow, build punchier, end deep & minimal") → an ordered set following that
  energy/BPM arc (needs per-track BPM/key/energy analysis + an LLM planner)
- **Two workflow modes** — full-auto ("do everything, I'll edit after") vs.
  interactive ("show me your guess + alternatives, keep refining")

## 🧱 How it's built

- **Engine** (`engine/`, Python) — embeddings, by-example classification, clustering,
  tags + folder organize, a local FastAPI sidecar. 110 tests. See `engine/README.md`.
- **App** (`app/`, Tauri v2 + React) — the desktop UI; the Rust shell auto-starts the
  sidecar. See `app/README.md`.
- **Research** (`research/FINDINGS.md`) — the deep-dive that shaped the design.
- **Specs/plans** (`docs/superpowers/`).

## 🔒 Privacy
Everything runs on your machine. Audio never leaves your computer (model weights
download once from Hugging Face on first use; classification is fully local).
