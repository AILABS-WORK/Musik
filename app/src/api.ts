import type {
  AppConfig, Genre, Health, Progress, SimilarItem, Track,
} from "./types";

const BASE: string =
  (import.meta as any).env?.VITE_API ?? "http://127.0.0.1:8000";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  base: BASE,
  health: () => req<Health>("GET", "/api/health"),
  getConfig: () => req<AppConfig>("GET", "/api/config"),
  setConfig: (b: Partial<AppConfig>) => req<AppConfig>("POST", "/api/config", b),

  scan: () => req<{ scanned: number; total: number }>("POST", "/api/scan"),
  embed: (force = false) => req<{ started: boolean }>("POST", `/api/embed?force=${force}`),
  /** Import dropped files/folders (absolute paths) and auto-embed them. */
  importPaths: (paths: string[], embed = true) =>
    req<{ added: number; files_seen: number; total: number; embedding: boolean }>(
      "POST", "/api/import", { paths, embed }),
  /** Browser drag-drop upload: base64 file bytes saved server-side + imported. */
  upload: (files: { name: string; data_base64: string }[]) =>
    req<{ added: number; files_seen: number; total: number; embedding: boolean }>(
      "POST", "/api/upload", { files }),
  /** Identify a recorded clip (base64 wav) against the library, no import. */
  identifyUpload: (name: string, data_base64: string, n = 5) =>
    req<{ matches: { track_id: number; name: string; score: number }[]; error?: string }>(
      "POST", "/api/identify-upload", { name, data_base64, n }),
  progress: () => req<Progress>("GET", "/api/progress"),

  tracks: () => req<Track[]>("GET", "/api/tracks"),
  genres: () => req<Genre[]>("GET", "/api/genres"),
  addGenre: (b: { name: string; parent_id?: number | null; level?: string }) =>
    req<{ genre_id: number }>("POST", "/api/genres", b),
  byExample: (b: { name: string; track_ids: number[]; parent_id?: number | null; level?: string }) =>
    req<{ genre_id: number }>("POST", "/api/genres/by-example", b),

  suggest: () => req<{ count: number; known: number }>("POST", "/api/suggest"),
  review: (limit = 50) => req<any[]>("GET", `/api/review?limit=${limit}`),
  confirm: (b: { track_id: number; genre_id: number }) =>
    req<{ ok: boolean }>("POST", "/api/confirm", b),

  similar: (id: number, n = 12) => req<SimilarItem[]>("GET", `/api/similar/${id}?n=${n}`),
  cluster: (minSize = 2) => req<any[]>("POST", `/api/cluster?min_size=${minSize}`),
  project: (method = "pca") => req<{ points: any[] }>("GET", `/api/project?method=${method}`),

  analyze: () => req<{ started: boolean }>("POST", "/api/analyze"),
  tag: () => req<{ started: boolean }>("POST", "/api/tag"),
  deep: () => req<{ started: boolean }>("POST", "/api/deep"),
  fuse: () => req<{ started: boolean }>("POST", "/api/fuse"),
  deepOne: (id: number) =>
    req<{ ok: boolean; stems?: string[]; language?: { language: string; confidence: number } | null }>(
      "POST", `/api/deep/${id}`),
  search: (query: string, threshold?: number | null, n = 80) =>
    req<{ results: { track_id: number; name: string; score: number }[]; method: string; matched_label?: string | null; note?: string }>(
      "POST", "/api/search", { query, n, threshold: threshold ?? null }),
  understanding: (id: number) =>
    req<{ track_id: number; top_tags: { label: string; prob: number }[]; vocal?: any; mood?: any; caption?: string | null }>(
      "GET", `/api/understanding/${id}`),
  buildSet: (description: string, length?: number | null) =>
    req<{ track_ids: number[]; names: string[]; arc: number[]; reasons: string[]; parsed: any }>(
      "POST", "/api/set-build", { description, length: length ?? null }),
  identify: (path: string, n = 5) =>
    req<{ matches: { track_id: number; name: string; score: number }[] }>(
      "POST", "/api/identify", { path, n }),
  radio: (trackId: number, n = 20) =>
    req<{ queue: { track_id: number; name: string }[] }>("GET", `/api/radio/${trackId}?n=${n}`),
  identifyMix: (path: string) =>
    req<{ segments: { start: number; end: number; track_id: number; name: string; score: number }[] }>(
      "POST", "/api/identify-mix", { path }),
  region: (artist: string, title?: string) =>
    req<{ region: Record<string, unknown> }>("POST", "/api/region", { artist, title: title ?? null }),
  /** Seed by-example genres from MusicBrainz labels on your own tracks (background). */
  mbSeed: () => req<{ started: boolean }>("POST", "/api/mb/seed"),
  mbLookup: (artist: string, title?: string) =>
    req<{ result: Record<string, unknown> }>("POST", "/api/mb/lookup", { artist, title: title ?? null }),
  relatedGenres: (genre: string, n = 25) =>
    req<{ related: { genre: string; weight: number }[] }>(
      "GET", `/api/genre/related?genre=${encodeURIComponent(genre)}&n=${n}`),
  // ---- segment-level similarity (waveform region -> matching parts) ----
  segmentIndex: () => req<{ started: boolean }>("POST", "/api/segment/index"),
  segmentSearch: (track_id: number, start: number, end: number, n = 20) =>
    req<{ matches: { track_id: number; name: string; score: number; start: number; end: number }[] }>(
      "POST", "/api/segment/search", { track_id, start, end, n }),
  segmentSave: (b: { track_id: number; start: number; end: number; label?: string; note?: string; genre_id?: number | null }) =>
    req<{ ok: boolean; segment_id?: number }>("POST", "/api/segment/save", b),
  segments: (genreId?: number) =>
    req<{ segments: { id: number; track_id: number; start: number; end: number; label: string | null; note: string | null }[] }>(
      "GET", `/api/segments${genreId != null ? `?genre_id=${genreId}` : ""}`),
  segmentMakeGenre: (b: { track_id: number; start: number; end: number; name: string; parent_id?: number | null; n?: number }) =>
    req<{ ok: boolean; genre_id?: number; examples?: number[]; matches?: { track_id: number; name: string; score: number }[] }>(
      "POST", "/api/segment/make-genre", b),
  // ---- the genre blend (multi-label with scores) ----
  trackSuggestions: (id: number) =>
    req<{ suggestions: { genre_id: number; name: string; confidence: number; rank: number }[] }>(
      "GET", `/api/track/${id}/suggestions`),

  writeTags: (dry_run: boolean) =>
    req<{ count: number; plans: any[]; applied: boolean }>("POST", "/api/write-tags", { dry_run }),
  organize: (dry_run: boolean) =>
    req<{ count: number; plan: any[]; applied: boolean }>("POST", "/api/organize", { dry_run }),
  undo: () => req<{ tags: number; organize: number }>("POST", "/api/undo"),
  seedTaxonomy: (refs_dir: string) => req<{ seeded: number }>("POST", "/api/seed-taxonomy", { refs_dir }),

  audioUrl: (id: number) => `${BASE}/api/audio/${id}`,
};
