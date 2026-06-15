# MGC Desktop App (Phase 2)

A Tauri v2 desktop app over the Phase 1 `mgc` engine. It runs a local Python
**sidecar** (FastAPI over the engine) and a **React** UI in a native window:
scan a folder → embed → define genres by example → review/confirm → browse
sound-alikes → write tags + organize, all with dry-run + undo.

```
Tauri (Rust) ──spawns──▶ Python sidecar (uvicorn mgc.server:app, :8000)
     │                          ▲
     └─ WebView2 ─ React UI ─ HTTP
```

## Prerequisites
- The engine installed in its venv (see `../engine/README.md`): `pip install -e "../engine[server]"`.
- Node 18+ and Rust (for Tauri). WebView2 (preinstalled on Win11).

## Run (dev)
```bash
cd app
npm install
npm run tauri dev
```
The Rust shell auto-starts the sidecar by locating `engine/.venv` (walks up from
the app, or set `MGC_PYTHON` to the venv's python). If it can't find it, start the
sidecar yourself in another terminal:
```bash
cd ../engine && .venv\Scripts\activate && mgc serve   # http://127.0.0.1:8000
```
The UI shows **"sidecar offline"** (amber) until the API is reachable.

### Web-only (no native window)
```bash
cd ../engine && mgc serve            # terminal 1
cd app && npm run dev                # terminal 2 → open http://localhost:5173
```

## Use it
1. In the top bar, set your **library path** + **model** (`baseline` works out of the box; `mert`/`clap`/`discogs` need `engine[models]`), click **Save**.
2. **Scan** → **Embed** (progress bar) → optionally **Seed taxonomy** (point at the RYM `references` dir).
3. Check a few tracks, open **Genres → Create genre by example**, name it, **Create from N selected**.
4. **Suggest** to classify everything; sort by confidence and **review**.
5. **Apply** tab: preview then write tags, preview then organize into `Genre/Subgenre`, or **Undo**.

## Build a shippable installer (with bundled sidecar)
To ship the app without requiring a Python install on the user's machine, bundle
the sidecar as a standalone binary:

> ⚠️ **Windows Defender note:** PyInstaller's onefile bootloader exe is a frequent
> antivirus false-positive — Defender may quarantine `mgc-sidecar.exe` the instant
> it's written (the build then fails with `FileNotFoundError ... mgc-sidecar.exe`).
> Add a Defender **exclusion** for the output/build folder (Settings → Virus &
> threat protection → Exclusions) — or build on macOS/Linux — before step 1.

```bash
# 1) Build the standalone sidecar (baseline engine; ~tens of MB)
engine/.venv/Scripts/python engine/packaging/build_sidecar.py     # -> dist/mgc-sidecar.exe

# 2) Place it as a Tauri sidecar (target-triple suffix; find yours with `rustc -vV`)
mkdir -p app/src-tauri/binaries
cp dist/mgc-sidecar.exe app/src-tauri/binaries/mgc-sidecar-x86_64-pc-windows-msvc.exe

# 3) Add it to app/src-tauri/tauri.conf.json:  "bundle": { "externalBin": ["binaries/mgc-sidecar"] }

# 4) Build the installer
cd app && npm run tauri build
```
The Rust shell prefers a bundled `mgc-sidecar` next to the app executable and
falls back to the engine venv in dev — so the same build works both ways.

> The bundled sidecar is the **baseline** engine only. For the heavy model
> backends (torch/essentia) the user still installs `engine[models]` and selects
> the model in the app; bundling multi-GB CUDA wheels is out of scope.

## Dev build (no bundling)
```bash
npm run tauri build   # bundles frontend + Rust shell; expects the engine venv at runtime
```

## Config
The sidecar reads `MGC_CONFIG` (or `mgc.config.json` in its cwd). The in-app
**Save** button (`POST /api/config`) updates library/model/db at runtime.
