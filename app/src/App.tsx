import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import type { Genre, Progress, Track } from "./types";
import { api } from "./api";
import { TopBar } from "./components/TopBar";
import { TrackTable } from "./components/TrackTable";
import { MapView } from "./components/MapView";
import { SidePanel } from "./components/SidePanel";
import type { SideTab } from "./components/SidePanel";
import { GenrePanel } from "./components/GenrePanel";
import { SimilarPanel } from "./components/SimilarPanel";
import { SongPanel } from "./components/SongPanel";
import { WavePanel } from "./components/WavePanel";
import { ClustersPanel } from "./components/ClustersPanel";
import { ApplyPanel } from "./components/ApplyPanel";
import { StatusBar } from "./components/StatusBar";
import { AddMusic } from "./components/AddMusic";
import { SearchBar } from "./components/SearchBar";
import { SetBuilder } from "./components/SetBuilder";
import { IdentifyPanel } from "./components/IdentifyPanel";
import { MixPanel } from "./components/MixPanel";
import { SelectionBar } from "./components/SelectionBar";
import { PlayerBar } from "./components/PlayerBar";
import { EmptyState } from "./components/EmptyState";
import { JobBanner } from "./components/JobBanner";
import type { JobKind } from "./components/JobBanner";
import { MobileRecord } from "./components/MobileRecord";

type MainView = "table" | "map";

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Mobile companion mode: the phone-first "Record & Identify" view replaces the
 * whole desktop UI when the URL asks for it (#/record or ?record=1 — the PWA
 * start_url). Read once at module load so the desktop path is untouched.
 */
function wantsMobileRecord(): boolean {
  if (typeof window === "undefined") return false;
  const { hash, search } = window.location;
  return hash === "#/record" || new URLSearchParams(search).has("record");
}

export default function App() {
  // Computed once; mobile mode is a static URL decision, not reactive state.
  const [mobileRecord] = useState(wantsMobileRecord);

  const [tracks, setTracks] = useState<Track[]>([]);
  const [genres, setGenres] = useState<Genre[]>([]);
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);


  const [tab, setTab] = useState<SideTab>("genres");
  const [view, setView] = useState<MainView>("table");

  const [status, setStatus] = useState("ready");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);

  const [progress, setProgress] = useState<Progress | null>(null);
  // What kind of background job is running, so the banner can speak plainly
  // ("Tagging sounds…" vs "Analyzing…"). null when idle.
  const [jobKind, setJobKind] = useState<JobKind | null>(null);
  // A short-lived "done"/"error" note shown after a job finishes.
  const [jobNote, setJobNote] = useState<{ kind: JobKind; error: string | null } | null>(null);
  const embedPollRef = useRef<number | null>(null);
  const jobNoteTimerRef = useRef<number | null>(null);
  // Read the latest jobKind from inside the poll callback without re-creating it.
  const jobKindRef = useRef<JobKind | null>(null);
  jobKindRef.current = jobKind;

  // ---- open-vocab search ----
  const [searchResults, setSearchResults] = useState<
    { track_id: number; score: number }[] | null
  >(null);
  const [searchMeta, setSearchMeta] = useState<{
    method: string;
    matched_label?: string | null;
    note?: string;
  } | null>(null);

  // ---- active play queue (set builder / radio) ----
  const [queue, setQueue] = useState<number[]>([]);
  const [queuePos, setQueuePos] = useState(0);
  // keep latest queue/pos readable from the audio onEnded handler
  const queueRef = useRef<number[]>([]);
  const queuePosRef = useRef(0);
  queueRef.current = queue;
  queuePosRef.current = queuePos;

  // guard against double-running the auto pipeline
  const autoRunningRef = useRef(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // ---- browser drag-drop loading ----
  const [dragActive, setDragActive] = useState(false);
  // depth counter so nested dragenter/leave events don't flicker the overlay
  const dragDepthRef = useRef(0);
  // hidden <input type=file> shared by EmptyState's "Browse" CTA
  const heroFileRef = useRef<HTMLInputElement | null>(null);

  const report = useCallback((msg: string, isError = false) => {
    setStatus(msg);
    setStatusError(isError);
  }, []);

  // ---- data loaders ----
  const loadTracks = useCallback(async () => {
    try {
      setTracks(await api.tracks());
    } catch (e) {
      report(`load tracks failed: ${errMsg(e)}`, true);
    }
  }, [report]);

  const loadGenres = useCallback(async () => {
    try {
      setGenres(await api.genres());
    } catch (e) {
      report(`load genres failed: ${errMsg(e)}`, true);
    }
  }, [report]);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadTracks(), loadGenres()]);
  }, [loadTracks, loadGenres]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  // Briefly surface a "done"/"error" banner after a job finishes, then clear
  // the running job. The note auto-dismisses after a few seconds.
  const flashJobNote = useCallback((kind: JobKind, error: string | null) => {
    setJobNote({ kind, error });
    if (jobNoteTimerRef.current !== null) {
      window.clearTimeout(jobNoteTimerRef.current);
    }
    jobNoteTimerRef.current = window.setTimeout(
      () => setJobNote(null),
      error ? 9000 : 4500,
    );
  }, []);

  // ---- embed progress polling ----
  const stopEmbedPoll = useCallback(() => {
    if (embedPollRef.current !== null) {
      window.clearInterval(embedPollRef.current);
      embedPollRef.current = null;
    }
  }, []);

  const startEmbedPoll = useCallback(() => {
    stopEmbedPoll();
    embedPollRef.current = window.setInterval(async () => {
      try {
        const p = await api.progress();
        setProgress(p);
        if (!p.running) {
          stopEmbedPoll();
          const kind = jobKindRef.current ?? "embed";
          if (p.error) {
            report(`${kind} error: ${p.error}`, true);
          } else {
            report(`${kind} finished · ${p.done}/${p.total}`);
          }
          flashJobNote(kind, p.error);
          setJobKind(null);
          void refreshAll();
        }
      } catch (e) {
        stopEmbedPoll();
        setJobKind(null);
        report(`progress poll failed: ${errMsg(e)}`, true);
      }
    }, 700);
  }, [stopEmbedPoll, report, refreshAll, flashJobNote]);

  useEffect(() => stopEmbedPoll, [stopEmbedPoll]);

  // Clear the job-note timer on unmount so it can't fire into a dead tree.
  useEffect(
    () => () => {
      if (jobNoteTimerRef.current !== null) {
        window.clearTimeout(jobNoteTimerRef.current);
      }
    },
    [],
  );

  // ---- actions ----
  const handleScan = useCallback(async () => {
    setBusy(true);
    report("scanning library…");
    try {
      const r = await api.scan();
      report(`scanned ${r.scanned} files · ${r.total} tracks total`);
      await refreshAll();
    } catch (e) {
      report(`scan failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, refreshAll]);

  const handleEmbed = useCallback(async () => {
    setBusy(true);
    report("starting embed…");
    try {
      const r = await api.embed(false);
      if (r.started) {
        report("embedding…");
        setJobKind("embed");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("embed not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`embed failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  const handleImport = useCallback(
    async (paths: string[]) => {
      if (!paths.length) return;
      setBusy(true);
      report(`importing ${paths.length} path(s)…`);
      try {
        const r = await api.importPaths(paths);
        report(`imported ${r.added} new track(s) from ${r.files_seen} file(s)`);
        if (r.embedding) {
          setJobKind("embed");
          setJobNote(null);
          setProgress({ running: true, done: 0, total: 0, last: "", error: null });
          startEmbedPoll();
        } else {
          await refreshAll();
        }
      } catch (e) {
        report(`import failed: ${errMsg(e)}`, true);
      } finally {
        setBusy(false);
      }
    },
    [report, startEmbedPoll, refreshAll],
  );

  // ---- browser drag-drop / Browse upload ----
  const handleUploadFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setBusy(true);
      report(`reading ${files.length} file(s)…`);
      try {
        // Read one File as a base64 data URL (api.upload tolerates the prefix).
        const readOne = (f: File) =>
          new Promise<{ name: string; data_base64: string }>((resolve, reject) => {
            const fr = new FileReader();
            fr.onload = () => resolve({ name: f.name, data_base64: String(fr.result) });
            fr.onerror = () => reject(fr.error ?? new Error(`read failed: ${f.name}`));
            fr.readAsDataURL(f);
          });
        // Upload in small batches — base64-ing a whole folder into one JSON body
        // overflows V8's max string length ("Invalid string length").
        const BATCH = 3;
        let added = 0, seen = 0, embedding = false;
        for (let i = 0; i < files.length; i += BATCH) {
          const slice = files.slice(i, i + BATCH);
          report(`uploading ${Math.min(i + BATCH, files.length)}/${files.length} file(s)…`);
          const payload = await Promise.all(slice.map(readOne));
          const r = await api.upload(payload);
          added += r.added;
          seen += r.files_seen;
          embedding = embedding || r.embedding;
        }
        report(`added ${added} new track(s) from ${seen} file(s)`);
        if (embedding) {
          setJobKind("embed");
          setJobNote(null);
          setProgress({ running: true, done: 0, total: 0, last: "", error: null });
          startEmbedPoll();
        } else {
          await refreshAll();
        }
      } catch (e) {
        report(`upload failed: ${errMsg(e)}`, true);
      } finally {
        setBusy(false);
      }
    },
    [report, startEmbedPoll, refreshAll],
  );

  // Trigger the hidden hero file input (used by the empty-state CTA).
  const openHeroPicker = useCallback(() => {
    heroFileRef.current?.click();
  }, []);

  // ---- native OS drag-drop under Tauri (real absolute paths) ----
  // Browsers can't expose absolute paths, so this only runs in the desktop
  // shell; the browser File overlay above covers the web/dev case.
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let active = true;
    let unlisten: (() => void) | undefined;
    void (async () => {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        const un = await getCurrentWebview().onDragDropEvent((event) => {
          const payload = event.payload as { type: string; paths?: string[] };
          if (payload.type === "enter" || payload.type === "over") {
            setDragActive(true);
          } else if (payload.type === "leave") {
            setDragActive(false);
          } else if (payload.type === "drop") {
            setDragActive(false);
            const paths = payload.paths ?? [];
            if (paths.length) void handleImport(paths);
          }
        });
        if (active) unlisten = un;
        else un();
      } catch {
        /* not under Tauri — browser overlay handles it */
      }
    })();
    return () => {
      active = false;
      if (unlisten) unlisten();
    };
  }, [handleImport]);

  // ---- full-window drag overlay ----
  // Only react to drags that carry files (not text / in-app element drags).
  const dragHasFiles = (e: DragEvent): boolean =>
    Array.from(e.dataTransfer?.types ?? []).includes("Files");

  const onDragEnter = useCallback((e: DragEvent) => {
    if (!dragHasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setDragActive(true);
  }, []);

  const onDragOver = useCallback((e: DragEvent) => {
    if (!dragHasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const onDragLeave = useCallback((e: DragEvent) => {
    if (!dragHasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setDragActive(false);
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      dragDepthRef.current = 0;
      setDragActive(false);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length === 0) return; // Tauri native-path drops are handled elsewhere
      e.preventDefault();
      void handleUploadFiles(files);
    },
    [handleUploadFiles],
  );

  const handleSuggest = useCallback(async () => {
    setBusy(true);
    report("suggesting genres…");
    try {
      const r = await api.suggest();
      report(`suggested ${r.count} · ${r.known} known`);
      await refreshAll();
    } catch (e) {
      report(`suggest failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, refreshAll]);

  // ---- analyze (BPM / key / energy) — reuses the embed progress flow ----
  const handleAnalyze = useCallback(async () => {
    setBusy(true);
    report("starting analysis…");
    try {
      const r = await api.analyze();
      if (r.started) {
        report("analyzing…");
        setJobKind("analyze");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("analyze not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`analyze failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  // ---- tag (AudioSet-527 sound tags) — reuses the embed progress flow ----
  const handleTag = useCallback(async () => {
    setBusy(true);
    report("starting tagging…");
    try {
      const r = await api.tag();
      if (r.started) {
        report("tagging…");
        setJobKind("tag");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("tag not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`tag failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  // ---- deep pass (stems + sung-language) — reuses the progress flow ----
  const handleDeep = useCallback(async () => {
    setBusy(true);
    report("starting deep analysis…");
    try {
      const r = await api.deep();
      if (r.started) {
        report("deep analysis…");
        setJobKind("deep");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("deep not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`deep failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  // ---- fuse (sharper grouping) — reuses the progress flow ----
  const handleFuse = useCallback(async () => {
    setBusy(true);
    report("starting fusion…");
    try {
      const r = await api.fuse();
      if (r.started) {
        report("fusing…");
        setJobKind("fuse");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("fuse not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`fuse failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  // ---- seed genres from MusicBrainz (uses artist/title tags) ----
  const handleMbSeed = useCallback(async () => {
    setBusy(true);
    report("starting MusicBrainz seed…");
    try {
      const r = await api.mbSeed();
      if (r.started) {
        report("seeding genres from MusicBrainz…");
        setJobKind("mbseed");
        setJobNote(null);
        setProgress({ running: true, done: 0, total: 0, last: "", error: null });
        startEmbedPoll();
      } else {
        report("MB seed not started (already running or nothing to do)");
      }
    } catch (e) {
      report(`MB seed failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  }, [report, startEmbedPoll]);

  // ---- open-vocab search ----
  const handleSearch = useCallback(
    async (query: string, threshold: number | null) => {
      if (!query.trim()) {
        setSearchResults(null);
        setSearchMeta(null);
        return;
      }
      report(`searching “${query.trim()}”…`);
      try {
        const r = await api.search(query, threshold);
        setSearchResults(
          r.results.map((x) => ({ track_id: x.track_id, score: x.score })),
        );
        setSearchMeta({ method: r.method, matched_label: r.matched_label, note: r.note });
        report(`search · ${r.results.length} result(s)`);
      } catch (e) {
        report(`search failed: ${errMsg(e)}`, true);
      }
    },
    [report],
  );

  const handleClearSearch = useCallback(() => {
    setSearchResults(null);
    setSearchMeta(null);
  }, []);

  // ---- find tracks with a similar frequency fingerprint (bass/mids/highs) ----
  const handleSimilarSound = useCallback(async (id: number) => {
    report("finding tracks with a similar frequency profile…");
    try {
      const r = await api.spectralSimilar(id);
      if (!r.matches.length) {
        report("building the frequency index — press ‘similar sound’ again in ~1 min");
        await api.spectralIndex();
        return;
      }
      setSearchResults(r.matches.map((m) => ({ track_id: m.track_id, score: m.score })));
      setSearchMeta({
        method: "spectral",
        matched_label: "similar frequency profile (bass/mids/highs)",
      });
      setView("table");
      report(`similar sound · ${r.matches.length} result(s)`);
    } catch (e) {
      report(`similar sound failed: ${errMsg(e)}`, true);
    }
  }, [report]);

  // ---- one-click auto pipeline: embed → analyze → suggest ----
  const waitForProgress = useCallback(async () => {
    // Poll until the background job reports !running.
    for (;;) {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 600));
      try {
        const p = await api.progress();
        setProgress(p);
        if (!p.running) return;
      } catch {
        // transient failure — stop waiting rather than spin forever
        return;
      }
    }
  }, []);

  const handleAuto = useCallback(async () => {
    if (autoRunningRef.current) return;
    autoRunningRef.current = true;
    setBusy(true);
    setJobNote(null);
    try {
      report("auto: embedding…");
      setJobKind("embed");
      setProgress({ running: true, done: 0, total: 0, last: "", error: null });
      await api.embed();
      await waitForProgress();
      report("auto: analyzing tempo + key…");
      setJobKind("analyze");
      setProgress({ running: true, done: 0, total: 0, last: "", error: null });
      await api.analyze();
      await waitForProgress();
      report("auto: tagging sounds…");
      setJobKind("tag");
      setProgress({ running: true, done: 0, total: 0, last: "", error: null });
      await api.tag();
      await waitForProgress();
      // identify by fingerprint -> real MusicBrainz genres (best-effort: needs a key)
      report("auto: identifying tracks (AcoustID)…");
      setJobKind("identify");
      setProgress({ running: true, done: 0, total: 0, last: "", error: null });
      try {
        await api.identifyAll();
        await waitForProgress();
      } catch {
        report("auto: skipped AcoustID (no key) — using sound-tag names");
      }
      // cluster into major genre folders + subgenres (dry-run preview first)
      report("auto: grouping into genres…");
      setJobKind("auto");
      setProgress(null);
      const preview = await api.autoOrganize(false);
      const tree = preview.tree
        .map((t) => `${t.major} (${t.size})`)
        .join(", ");
      report(`auto: ${preview.tree.length} genres — ${tree}`);
      // copy every track into root/<Major>/<Subgenre>/ (originals untouched; undo available)
      report("auto: sorting files into folders…");
      const done = await api.autoOrganize(true);
      await refreshAll();
      report(`auto: sorted ${done.count} files into ${preview.tree.length} genre folders. Undo available.`);
      flashJobNote("auto", null);
    } catch (e) {
      report(`auto-sort failed: ${errMsg(e)}`, true);
      flashJobNote("auto", errMsg(e));
    } finally {
      stopEmbedPoll();
      setProgress(null);
      setJobKind(null);
      setBusy(false);
      autoRunningRef.current = false;
    }
  }, [report, waitForProgress, refreshAll, stopEmbedPoll, flashJobNote]);

  // ---- re-sort using by-example labels (no re-embed; updates genres + assignments) ----
  const handleResort = useCallback(async () => {
    report("re-sorting with your labels…");
    setJobKind("auto");
    try {
      const r = await api.autoOrganize(false);
      await refreshAll();
      const real = r.tree
        .flatMap((t) => t.subgenres)
        .filter((s) => !s.name.includes("Unsorted"))
        .reduce((a, s) => a + s.size, 0);
      report(`re-sorted · ${real} in subgenres across ${r.tree.length} genres (your labels propagated)`);
      flashJobNote("auto", null);
    } catch (e) {
      report(`re-sort failed: ${errMsg(e)}`, true);
    } finally {
      setJobKind(null);
    }
  }, [report, refreshAll, flashJobNote]);

  const handleSortFromLabels = useCallback(async () => {
    report("learning your subgenres + sorting the library…");
    setJobKind("auto");
    try {
      const r = await api.propagateLabels();
      if (r.error) {
        report(
          r.need
            ? `Sort from labels: ${r.need} (you have ${r.ready_classes ?? 0} ready). Label a few more, then retry.`
            : `Sort from labels: ${r.error}`,
          true,
        );
      } else {
        await refreshAll();
        report(`sorted ${r.assigned} tracks into your ${r.classes} subgenres (your ${r.labelled} labels kept exact)`);
      }
      flashJobNote("auto", null);
    } catch (e) {
      report(`sort from labels failed: ${errMsg(e)}`, true);
    } finally {
      setJobKind(null);
    }
  }, [report, refreshAll, flashJobNote]);

  // ---- selection / checks ----
  const toggleCheck = useCallback((id: number) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleAll = useCallback((ids: number[], check: boolean) => {
    setChecked((prev) => {
      const next = new Set(prev);
      for (const id of ids) {
        if (check) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  }, []);

  // ---- selection action bar → existing tabs (selection stays intact) ----
  const handleBuildFromSelection = useCallback(() => {
    setTab("set");
    report(`${checked.size} track(s) ready — describe the set you want`);
  }, [checked.size, report]);

  const handleUseAsExamples = useCallback(() => {
    setTab("genres");
    report(`${checked.size} track(s) ready — name a genre to create by example`);
  }, [checked.size, report]);

  const clearChecked = useCallback(() => setChecked(new Set()), []);

  // SimilarPanel fetches its own detailed matches; selecting just sets the id.
  const selectTrack = useCallback((id: number) => setSelectedId(id), []);

  // ---- audio ----
  const play = useCallback(
    (id: number) => {
      const el = audioRef.current;
      if (!el) return;
      try {
        el.src = api.audioUrl(id);
        void el.play();
        setPlayingId(id);
        report(`playing track ${id}`);
      } catch (e) {
        report(`play failed: ${errMsg(e)}`, true);
      }
    },
    [report],
  );

  // ---- active play queue (set builder / radio) ----
  const playQueue = useCallback(
    (ids: number[]) => {
      const list = ids.filter((id) => Number.isFinite(id));
      if (list.length === 0) {
        report("queue is empty", true);
        return;
      }
      setQueue(list);
      setQueuePos(0);
      report(`queue · 1/${list.length}`);
      play(list[0] as number);
    },
    [play, report],
  );

  // advance to the next queued track when the current one ends
  const playNext = useCallback(() => {
    const q = queueRef.current;
    const next = queuePosRef.current + 1;
    if (next < q.length) {
      setQueuePos(next);
      report(`queue · ${next + 1}/${q.length}`);
      play(q[next] as number);
    }
  }, [play, report]);

  const handleRadio = useCallback(
    async (trackId: number) => {
      report("tuning radio…");
      try {
        const r = await api.radio(trackId);
        const ids = r.queue.map((q) => q.track_id);
        if (ids.length === 0) {
          report("radio returned no tracks", true);
          return;
        }
        report(`radio · ${ids.length} tracks`);
        playQueue(ids);
      } catch (e) {
        report(`radio failed: ${errMsg(e)}`, true);
      }
    },
    [playQueue, report],
  );

  // ---- create genre by example ----
  const createByExample = useCallback(
    async (args: { name: string; parentId: number | null; level: string }) => {
      const trackIds = Array.from(checked);
      try {
        await api.byExample({
          name: args.name,
          track_ids: trackIds,
          parent_id: args.parentId,
          level: args.level,
        });
        report(`created genre “${args.name}” from ${trackIds.length} tracks`);
        setChecked(new Set());
        await refreshAll();
      } catch (e) {
        report(`create genre failed: ${errMsg(e)}`, true);
      }
    },
    [checked, report, refreshAll],
  );

  // ---- inline confirm / relabel from the track table ----
  const confirmGenre = useCallback(
    async (trackId: number, genreId: number) => {
      try {
        await api.confirm({ track_id: trackId, genre_id: genreId });
        const gName = genres.find((g) => g.id === genreId)?.name ?? `#${genreId}`;
        report(`assigned track ${trackId} → “${gName}”`);
        await refreshAll();
      } catch (e) {
        report(`confirm failed: ${errMsg(e)}`, true);
      }
    },
    [genres, report, refreshAll],
  );

  const selectedTrack = tracks.find((t) => t.id === selectedId) ?? null;
  const checkedIds = Array.from(checked);
  const embedRunning = progress?.running ?? false;

  // When a search is active, show its results in result order; otherwise
  // show all tracks. Map each result id to its Track, dropping any misses.
  const displayedTracks: Track[] =
    searchResults === null
      ? tracks
      : searchResults
          .map((r) => tracks.find((t) => t.id === r.track_id))
          .filter((t): t is Track => t !== undefined);

  const libraryEmpty = tracks.length === 0 && searchResults === null;

  // Phone-first companion: full-screen Record & Identify replaces the desktop
  // UI when the URL requests it. Everything below is the unchanged desktop app.
  if (mobileRecord) {
    return <MobileRecord />;
  }

  return (
    <div
      className="app"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <TopBar
        busy={busy}
        progress={progress}
        jobKind={jobKind}
        report={report}
        onScan={handleScan}
        onEmbed={handleEmbed}
        onAnalyze={handleAnalyze}
        onTag={handleTag}
        onDeep={handleDeep}
        onFuse={handleFuse}
        onMbSeed={handleMbSeed}
        onSuggest={handleSuggest}
        onAuto={handleAuto}
        onResort={handleResort}
        onSortFromLabels={handleSortFromLabels}
      />

      <div className="app__body">
        <main className="app__main">
          <JobBanner kind={jobKind} progress={progress} note={jobNote} />
          {libraryEmpty ? (
            <div className="app__main-scroll">
              <AddMusic
                onImport={(p) => void handleImport(p)}
                onUploadFiles={(f) => void handleUploadFiles(f)}
                report={report}
              />
              <EmptyState onBrowse={openHeroPicker} />
            </div>
          ) : (
            <>
              <SearchBar
                onSearch={(q, t) => void handleSearch(q, t)}
                onClear={handleClearSearch}
                meta={searchMeta}
                active={searchResults !== null}
                count={displayedTracks.length}
              />
              <AddMusic
                onImport={(p) => void handleImport(p)}
                onUploadFiles={(f) => void handleUploadFiles(f)}
                report={report}
              />
              <div className="viewtoggle">
                <div className="seg">
                  <button
                    className={view === "table" ? "seg__btn active" : "seg__btn"}
                    onClick={() => setView("table")}
                  >
                    Table
                  </button>
                  <button
                    className={view === "map" ? "seg__btn active" : "seg__btn"}
                    onClick={() => setView("map")}
                  >
                    Map
                  </button>
                </div>
              </div>
              {view === "table" ? (
                <>
                  {searchResults !== null && (
                    <div className="search-banner">
                      showing{" "}
                      <strong>{displayedTracks.length}</strong> result
                      {displayedTracks.length === 1 ? "" : "s"} for your search
                      <button
                        className="btn btn--xs search-banner__clear"
                        onClick={handleClearSearch}
                      >
                        Clear
                      </button>
                    </div>
                  )}
                  <TrackTable
                    tracks={displayedTracks}
                    genres={genres}
                    checked={checked}
                    selectedId={selectedId}
                    onToggleCheck={toggleCheck}
                    onToggleAll={toggleAll}
                    onSelect={selectTrack}
                    onPlay={play}
                    onConfirm={(trackId, genreId) => {
                      void confirmGenre(trackId, genreId);
                    }}
                  />
                </>
              ) : (
                <MapView
                  tracks={tracks}
                  selectedId={selectedId}
                  onSelect={selectTrack}
                />
              )}
            </>
          )}
        </main>

        <SidePanel active={tab} onTab={setTab}>
          {tab === "genres" && (
            <GenrePanel
              genres={genres}
              checkedIds={checkedIds}
              onByExample={createByExample}
            />
          )}
          {tab === "similar" && (
            <SimilarPanel
              selected={selectedTrack}
              genres={genres}
              onPlay={play}
              onRadio={(id) => void handleRadio(id)}
              report={report}
              onChanged={refreshAll}
            />
          )}
          {tab === "song" && (
            <SongPanel
              track={tracks.find((t) => t.id === selectedId) ?? null}
              genres={genres}
              onPlay={play}
              onChanged={refreshAll}
            />
          )}
          {tab === "parts" && (
            <WavePanel
              track={tracks.find((t) => t.id === selectedId) ?? null}
              report={report}
              onChanged={refreshAll}
            />
          )}
          {tab === "clusters" && (
            <ClustersPanel
              tracks={tracks}
              genres={genres}
              report={report}
              refresh={refreshAll}
              onPlay={play}
              onSelect={setSelectedId}
            />
          )}
          {tab === "set" && (
            <SetBuilder
              report={report}
              onPlayQueue={playQueue}
              onPlay={play}
            />
          )}
          {tab === "id" && <IdentifyPanel report={report} onPlay={play} />}
          {tab === "mix" && <MixPanel report={report} onPlay={play} />}
          {tab === "apply" && (
            <ApplyPanel report={report} onMutated={refreshAll} />
          )}
        </SidePanel>
      </div>

      {/* bottom transport: RGB spectral waveform + playhead + similar-sound */}
      <PlayerBar
        audioRef={audioRef}
        trackId={playingId}
        trackName={tracks.find((t) => t.id === playingId)?.name ?? null}
        bpm={tracks.find((t) => t.id === playingId)?.bpm ?? null}
        onSimilarSound={handleSimilarSound}
      />

      <SelectionBar
        count={checked.size}
        onBuildSet={handleBuildFromSelection}
        onUseAsExamples={handleUseAsExamples}
        onClear={clearChecked}
      />

      <StatusBar
        message={status}
        isError={statusError}
        busy={busy || embedRunning}
      />

      {/* full-window drag-and-drop overlay (browser File drops) */}
      {dragActive && (
        <div className="dropzone" aria-hidden="true">
          <div className="dropzone__card">
            <div className="dropzone__glyph">⤓</div>
            <div className="dropzone__title">Drop your music to add it</div>
            <div className="dropzone__sub">
              wav · mp3 · flac · aiff · m4a · ogg
            </div>
          </div>
        </div>
      )}

      {/* hidden file input shared by the empty-state "Browse" CTA */}
      <input
        ref={heroFileRef}
        type="file"
        multiple
        accept="audio/*,.flac,.wav,.mp3,.aiff,.m4a,.ogg"
        hidden
        onChange={(e) => {
          const list = e.target.files;
          if (list && list.length > 0) void handleUploadFiles(Array.from(list));
          e.target.value = "";
        }}
      />

      {/* subtle "up next" indicator while a queue is active */}
      {queue.length > 0 && queuePos + 1 < queue.length && (
        <div className="upnext" title="up next in queue">
          up next ·{" "}
          <span className="upnext__name">
            {tracks.find((t) => t.id === queue[queuePos + 1])?.name ??
              `#${queue[queuePos + 1]}`}
          </span>
          <span className="upnext__pos">
            {queuePos + 2}/{queue.length}
          </span>
        </div>
      )}

      {/* one shared audio element used by every play button */}
      <audio ref={audioRef} onEnded={playNext} />
    </div>
  );
}
