import { useCallback, useEffect, useRef, useState } from "react";
import type { Genre, Progress, SimilarItem, Track } from "./types";
import { api } from "./api";
import { TopBar } from "./components/TopBar";
import { TrackTable } from "./components/TrackTable";
import { SidePanel } from "./components/SidePanel";
import type { SideTab } from "./components/SidePanel";
import { GenrePanel } from "./components/GenrePanel";
import { SimilarPanel } from "./components/SimilarPanel";
import { ApplyPanel } from "./components/ApplyPanel";
import { StatusBar } from "./components/StatusBar";

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

  const [status, setStatus] = useState("ready");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);

  const [progress, setProgress] = useState<Progress | null>(null);
  const embedPollRef = useRef<number | null>(null);

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

  const selectedTrack = tracks.find((t) => t.id === selectedId) ?? null;
  const checkedIds = Array.from(checked);
  const embedRunning = progress?.running ?? false;

  return (
    <div className="app">
      <TopBar
        busy={busy}
        progress={progress}
        report={report}
        onScan={handleScan}
        onEmbed={handleEmbed}
        onSuggest={handleSuggest}
      />

      <div className="app__body">
        <main className="app__main">
          <TrackTable
            tracks={tracks}
            checked={checked}
            selectedId={selectedId}
            onToggleCheck={toggleCheck}
            onToggleAll={toggleAll}
            onSelect={selectTrack}
            onPlay={play}
          />
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
            />
          )}
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

      {/* one shared audio element used by every play button */}
      <audio ref={audioRef} />
    </div>
  );
}
