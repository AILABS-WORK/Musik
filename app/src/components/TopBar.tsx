import { useCallback, useEffect, useState } from "react";
import type { Health, Progress } from "../types";
import type { JobKind } from "./JobBanner";
import { api } from "../api";

const MODELS = ["baseline", "discogs", "mert", "clap"];

/** Friendly present-tense verb for the compact top-bar progress chip. */
const JOB_VERB: Record<JobKind, string> = {
  embed: "listening",
  analyze: "analyzing",
  tag: "tagging",
  deep: "deep-analyzing",
  suggest: "classifying",
  auto: "auto-sorting",
};

interface TopBarProps {
  busy: boolean;
  onScan: () => void;
  onEmbed: () => void;
  onAnalyze: () => void;
  onTag: () => void;
  onDeep: () => void;
  onSuggest: () => void;
  onAuto: () => void;
  /** Live embed progress; null when no embed has run / not running. */
  progress: Progress | null;
  /** Which background job is running, for a plain-language label. */
  jobKind?: JobKind | null;
  /** Bubble a status string (and error flag) up to the shared status bar. */
  report: (msg: string, isError?: boolean) => void;
  /** Notify parent of config so it can react (e.g. show library path). */
  onConfigLoaded?: (model: string, library: string | null) => void;
}

export function TopBar({
  busy,
  onScan,
  onEmbed,
  onAnalyze,
  onTag,
  onDeep,
  onSuggest,
  onAuto,
  progress,
  jobKind,
  report,
  onConfigLoaded,
}: TopBarProps) {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const [libraryPath, setLibraryPath] = useState("");
  const [model, setModel] = useState("baseline");
  const [refsDir, setRefsDir] = useState("");
  const [saving, setSaving] = useState(false);
  const [seeding, setSeeding] = useState(false);

  // Poll health every few seconds.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        setHealth(h);
        setOffline(false);
      } catch {
        if (!alive) return;
        setOffline(true);
      }
    };
    void tick();
    const t = window.setInterval(tick, 4000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);

  // Load config once on mount.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const cfg = await api.getConfig();
        if (!alive) return;
        setLibraryPath(cfg.library_root ?? "");
        setModel(cfg.active_model || "baseline");
        onConfigLoaded?.(cfg.active_model, cfg.library_root);
      } catch (e) {
        report(`config load failed: ${errMsg(e)}`, true);
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const saveConfig = useCallback(async () => {
    setSaving(true);
    try {
      const cfg = await api.setConfig({
        library_root: libraryPath.trim() || null,
        active_model: model,
      });
      setLibraryPath(cfg.library_root ?? "");
      setModel(cfg.active_model || "baseline");
      onConfigLoaded?.(cfg.active_model, cfg.library_root);
      report(`config saved · model ${cfg.active_model}`);
    } catch (e) {
      report(`save config failed: ${errMsg(e)}`, true);
    } finally {
      setSaving(false);
    }
  }, [libraryPath, model, onConfigLoaded, report]);

  const seed = useCallback(async () => {
    if (!refsDir.trim()) {
      report("refs dir is empty", true);
      return;
    }
    setSeeding(true);
    try {
      const r = await api.seedTaxonomy(refsDir.trim());
      report(`seeded taxonomy · ${r.seeded} genres`);
    } catch (e) {
      report(`seed taxonomy failed: ${errMsg(e)}`, true);
    } finally {
      setSeeding(false);
    }
  }, [refsDir, report]);

  const embedRunning = progress?.running ?? false;
  const total = progress?.total ?? 0;
  const done = progress?.done ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

  return (
    <header className="topbar">
      <div className="topbar__row">
        <div className="brand">
          <span className="brand__logo">Musik</span>
          <span className="brand__sub">your DJ crate, organized by sound</span>
        </div>

        <div className={healthClass(offline)}>
          <span className="health__dot" />
          {offline ? (
            <span>sidecar offline</span>
          ) : health ? (
            <span>
              {health.model} · {health.tracks} tracks · {health.genres} genres
            </span>
          ) : (
            <span>connecting…</span>
          )}
        </div>

        <div className="spacer" />

        <div className="btn-group" role="group" aria-label="Library">
          <span className="btn-group__label">Library</span>
          <button
            className="btn btn--accent"
            onClick={onScan}
            disabled={busy || embedRunning}
            title="Scan the library folder for audio files"
          >
            Scan
          </button>
          <button
            className="btn btn--accent"
            onClick={onEmbed}
            disabled={busy || embedRunning}
            title="Compute sound embeddings for new tracks"
          >
            Embed
          </button>
        </div>

        <div className="btn-group" role="group" aria-label="Analyze">
          <span className="btn-group__label">Analyze</span>
          <button
            className="btn btn--accent"
            onClick={onAnalyze}
            disabled={busy || embedRunning}
            title="Detect BPM, musical key and energy"
          >
            Analyze
          </button>
          <button
            className="btn btn--accent"
            onClick={onTag}
            disabled={busy || embedRunning}
            title="Compute AudioSet-527 sound tags in the background"
          >
            Tag
          </button>
          <button
            className="btn btn--accent"
            onClick={onDeep}
            disabled={busy || embedRunning}
            title="Deep pass: separate stems for finer percussion + detect sung language (slower, GPU)"
          >
            Deep
          </button>
        </div>

        <div className="btn-group" role="group" aria-label="Classify">
          <span className="btn-group__label">Classify</span>
          <button
            className="btn btn--accent"
            onClick={onSuggest}
            disabled={busy || embedRunning}
            title="Suggest genres for unclassified tracks"
          >
            Suggest
          </button>
          <button
            className="btn btn--go"
            onClick={onAuto}
            disabled={busy || embedRunning}
            title="Embed → Analyze → Suggest, all in one"
          >
            Auto-sort
          </button>
        </div>
      </div>

      <div className="topbar__row">
        <div className="field" style={{ flex: "1 1 280px" }}>
          <span className="field__label">Library</span>
          <input
            type="text"
            className="input--path"
            placeholder="/path/to/music/library"
            value={libraryPath}
            onChange={(e) => setLibraryPath(e.target.value)}
          />
        </div>

        <div className="field">
          <span className="field__label">Model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <button className="btn" onClick={saveConfig} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>

        <div className="field">
          <span className="field__label">Refs</span>
          <input
            type="text"
            className="input--path"
            placeholder="/path/to/reference/genres"
            value={refsDir}
            onChange={(e) => setRefsDir(e.target.value)}
            style={{ minWidth: 160 }}
          />
        </div>
        <button className="btn" onClick={seed} disabled={seeding}>
          {seeding ? "Seeding…" : "Seed taxonomy"}
        </button>
      </div>

      {embedRunning && (
        <div className="progress" role="status" aria-live="polite">
          <span className="progress__label">
            {jobKind ? JOB_VERB[jobKind] : "working"}{" "}
            {total > 0 ? `${done}/${total}` : "…"}
          </span>
          <div className="progress__track">
            <div
              className={
                total > 0
                  ? "progress__fill"
                  : "progress__fill progress__fill--indeterminate"
              }
              style={total > 0 ? { width: `${pct}%` } : undefined}
            />
          </div>
          <span className="progress__last" title={progress?.last ?? ""}>
            {progress?.last || "…"}
          </span>
        </div>
      )}
    </header>
  );
}

function healthClass(offline: boolean): string {
  return offline ? "health health--off" : "health health--ok";
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
