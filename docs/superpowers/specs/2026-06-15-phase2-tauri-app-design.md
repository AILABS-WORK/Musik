# Design Spec — Music Genre Classifier: Phase 2 Desktop App

- **Date:** 2026-06-15
- **Status:** Building
- **Builds on:** Phase 1 engine (`engine/`, package `mgc`) — already complete, 109 tests green.

## Goal
Wrap the proven Phase 1 engine in a **Tauri v2 desktop app** (cross-OS, fast, native) that lets the user drive the whole cold-start workflow visually: scan → embed → define genres by example → review/confirm → browse sound-alikes → write tags + organize, with dry-run + undo.

## Architecture
```
┌─ Tauri v2 shell (Rust) ─────────────────────────────┐
│  on startup: spawn the Python sidecar; kill on exit  │
│  loads the web UI in a WebView2 window               │
│                                                      │
│   React + Vite frontend  ── HTTP ──▶  FastAPI sidecar│
│   (app/src)                          (mgc.server)    │
│                                          │            │
│                                       Engine facade   │
│                                       (mgc.api)       │
└──────────────────────────────────────────────────────┘
```
- **Sidecar (`mgc/server.py`):** FastAPI over the `Engine`; single instance, all access serialized by a lock, sqlite `check_same_thread=False`. Embedding runs on a background thread that locks per-track and reports progress via `/api/progress`. Audio is streamed from `/api/audio/{id}` for in-app playback.
- **Frontend (`app/src`):** React + TypeScript + Vite. One window, no router; a tracks table + a tabbed side panel (Genres / Similar / Apply). Talks to the sidecar via `src/api.ts`.
- **Shell (`app/src-tauri`):** Rust; `find_python()` walks up to the engine venv (or `$MGC_PYTHON`), spawns `uvicorn mgc.server:app` on `127.0.0.1:8000`, and kills it on `ExitRequested`.

## API surface (sidecar)
`GET /api/health` · `GET|POST /api/config` · `POST /api/scan` · `POST /api/embed` + `GET /api/progress` · `GET /api/tracks` · `GET /api/audio/{id}` · `GET /api/genres` · `POST /api/genres` · `POST /api/genres/by-example` · `POST /api/suggest` · `GET /api/review` · `POST /api/confirm` · `GET /api/similar/{id}` · `POST /api/cluster` · `GET /api/project` · `POST /api/write-tags` · `POST /api/organize` · `POST /api/undo` · `POST /api/seed-taxonomy`.

## Screens (MVP)
1. **Top bar** — health badge (model · tracks · genres), config (library path + model + save), actions: Scan, Embed (progress bar), Suggest, optional Seed-taxonomy.
2. **Tracks table** — name, assigned genre, confidence bar, play (▶ streams audio), row-select, checkboxes for by-example selection, filter + sort-by-confidence.
3. **Side panel tabs** — *Genres* (tree + "create genre by example from checked tracks"); *Similar* (sound-alikes for the selected track); *Apply* (preview/apply tag writes, preview/apply organize, undo).
4. **Status bar** — last result / errors; "sidecar offline" when the API is unreachable.

## Verification
- Sidecar: pytest via FastAPI `TestClient` (full flow) — green.
- Frontend: `npm run build` + `tsc --noEmit` clean; then drive the live UI (Vite dev + sidecar) with Playwright and screenshot the real screens.
- Shell: `cargo check` the Rust.

## MVP boundary / deferred
- **Distribution bundling** (PyInstaller sidecar binary + `tauri build` installer) is deferred — dev runs via `mgc serve` + `npm run tauri dev` (Rust also auto-spawns the sidecar).
- Similarity **map** (2D projection) endpoint exists (`/api/project`); a rich interactive map is a later polish.
- Auth/multi-user: N/A (single-user local app).

## Non-goals
Cloud sync, mobile, packaging signing — not now.
