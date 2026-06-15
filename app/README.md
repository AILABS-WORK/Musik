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

## Build an installer
```bash
npm run tauri build
```
> Note: this bundles the frontend + Rust shell. Shipping the Python sidecar as a
> standalone (PyInstaller) binary inside the installer is a later step; for now
> the app expects the engine venv to be present (dev workflow).

## Config
The sidecar reads `MGC_CONFIG` (or `mgc.config.json` in its cwd). The in-app
**Save** button (`POST /api/config`) updates library/model/db at runtime.
