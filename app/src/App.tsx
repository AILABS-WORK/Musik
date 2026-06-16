import { useCallback, useEffect, useRef, useState } from "react";
import type { Genre, Progress, SimilarItem, Track } from "./types";
import { api } from "./api";
import { TopBar } from "./components/TopBar";
import { TrackTable } from "./components/TrackTable";
import { MapView } from "./components/MapView";
import { SidePanel } from "./components/SidePanel";
import type { SideTab } from "./components/SidePanel";
import { GenrePanel } from "./components/GenrePanel";
import { SimilarPanel } from "./components/SimilarPanel";
import { ClustersPanel } from "./components/ClustersPanel";
import { ApplyPanel } from "./components/ApplyPanel";
import { StatusBar } from "./components/StatusBar";
import { ImportBar } from "./components/ImportBar";
import { SearchBar } from "./components/SearchBar";
import { SetBuilder } from "./components/SetBuilder";
import { IdentifyPanel } from "./components/IdentifyPanel";
import { MixPanel } from "./components/MixPanel";

type MainView = "table" | "map";

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export default function App() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [genres, setGenres] = useState<Genre[]>([]);
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [similar, setSimilar] = useState<SimilarItem[]>([]);
  const [similarLoading, setSimilarLoading] = useState(false);

  const [tab, setTab] = useState<SideTab>("genres");
  const [view, setView] = useState<MainView>("table");

  const [status, setStatus] = useState("ready");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);

  const [progress, setProgress] = useState<Progress | null>(null);
  const embedPollRef = useRef<number | null>(null);

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
          if (p.error) {
            report(`embed error: ${p.error}`, true);
          } else {
            report(`embed finished · ${p.done}/${p.total}`);
          }
          void refreshAll();
        }
      } catch (e) {
        stopEmbedPoll();
        report(`progress poll failed: ${errMsg(e)}`, true);
      }
    }, 700);
  }, [stopEmbedPoll, report, refreshAll]);

  useEffect(() => stopEmbedPoll, [stopEmbedPoll]);

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
    try {
      report("auto: embedding…");
      await api.embed();
      await waitForProgress();
      report("auto: analyzing…");
      await api.analyze();
      await waitForProgress();
      report("auto: classifying…");
      await api.suggest();
      await refreshAll();
      report("auto: done");
    } catch (e) {
      report(`auto-sort failed: ${errMsg(e)}`, true);
    } finally {
      stopEmbedPoll();
      setProgress(null);
      setBusy(false);
      autoRunningRef.current = false;
    }
  }, [report, waitForProgress, refreshAll, stopEmbedPoll]);

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

  const selectTrack = useCallback(
    async (id: number) => {
      setSelectedId(id);
      setSimilarLoading(true);
      setSimilar([]);
      try {
        setSimilar(await api.similar(id));
      } catch (e) {
        report(`similar failed: ${errMsg(e)}`, true);
      } finally {
        setSimilarLoading(false);
      }
    },
    [report],
  );

  // ---- audio ----
  const play = useCallback(
    (id: number) => {
      const el = audioRef.current;
      if (!el) return;
      try {
        el.src = api.audioUrl(id);
        void el.play();
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

  return (
    <div className="app">
      <TopBar
        busy={busy}
        progress={progress}
        report={report}
        onScan={handleScan}
        onEmbed={handleEmbed}
        onAnalyze={handleAnalyze}
        onTag={handleTag}
        onSuggest={handleSuggest}
        onAuto={handleAuto}
      />

      <div className="app__body">
        <main className="app__main">
          <SearchBar
            onSearch={(q, t) => void handleSearch(q, t)}
            onClear={handleClearSearch}
            meta={searchMeta}
            active={searchResults !== null}
            count={displayedTracks.length}
          />
          <ImportBar onImport={(p) => void handleImport(p)} report={report} />
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
              items={similar}
              loading={similarLoading}
              onPlay={play}
              onRadio={(id) => void handleRadio(id)}
            />
          )}
          {tab === "clusters" && (
            <ClustersPanel
              tracks={tracks}
              genres={genres}
              report={report}
              refresh={refreshAll}
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

      <StatusBar
        message={status}
        isError={statusError}
        busy={busy || embedRunning}
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
