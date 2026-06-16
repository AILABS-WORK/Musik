import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Track } from "../types";
import { api } from "../api";
import "./wave.css";

interface WavePanelProps {
  track: Track | null;
  report: (m: string, e?: boolean) => void;
  /** Called after a subgenre is created so the parent can refresh. */
  onChanged?: () => void;
}

/** One result row from segment search. */
interface Match {
  track_id: number;
  name: string;
  score: number;
  start: number;
  end: number;
}

/** A drag-selected region in seconds. null start means "no selection". */
interface Region {
  start: number;
  end: number;
}

/** How many peak bars to draw across the canvas (kept modest for the ~360px panel). */
const PEAK_BARS = 720;

/** A single, lazily-created AudioContext shared across every WavePanel mount. */
let sharedCtx: AudioContext | null = null;
function getAudioContext(): AudioContext {
  if (sharedCtx === null) {
    // Safari still ships only the webkit-prefixed constructor.
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) throw new Error("Web Audio API is not available in this browser");
    sharedCtx = new Ctor();
  }
  return sharedCtx;
}

/** Format seconds as M:SS (e.g. 73 -> "1:13"). */
function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Downsample a channel's samples into `bars` peak magnitudes (0..1). For each
 * output bar we take the max absolute sample in its source window — that keeps
 * transients visible rather than averaging them away.
 */
function computePeaks(samples: Float32Array, bars: number): Float32Array {
  const peaks = new Float32Array(bars);
  if (samples.length === 0) return peaks;
  const per = samples.length / bars;
  for (let i = 0; i < bars; i++) {
    const from = Math.floor(i * per);
    const to = Math.min(samples.length, Math.floor((i + 1) * per));
    let max = 0;
    for (let j = from; j < to; j++) {
      const v = Math.abs(samples[j] as number);
      if (v > max) max = v;
    }
    peaks[i] = max;
  }
  // Normalize so the loudest bar fills the height.
  let peak = 0;
  for (let i = 0; i < bars; i++) {
    const v = peaks[i] as number;
    if (v > peak) peak = v;
  }
  if (peak > 0) {
    for (let i = 0; i < bars; i++) peaks[i] = (peaks[i] as number) / peak;
  }
  return peaks;
}

export function WavePanel({ track, report, onChanged }: WavePanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Decoded peak magnitudes for the current track (null while loading/unloaded).
  const peaksRef = useRef<Float32Array | null>(null);

  const [loading, setLoading] = useState(false);
  const [decodeErr, setDecodeErr] = useState<string | null>(null);
  // Bumped after peaks land so the draw effect re-runs.
  const [peaksReady, setPeaksReady] = useState(0);
  const [duration, setDuration] = useState(0);

  const [region, setRegion] = useState<Region | null>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);

  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);

  const [matches, setMatches] = useState<Match[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [makingGenre, setMakingGenre] = useState(false);

  // Private audio elements: one for previewing the selection, one for results.
  const selAudioRef = useRef<HTMLAudioElement | null>(null);
  const matchAudioRef = useRef<HTMLAudioElement | null>(null);
  // Timers that stop selection / match playback at the region end.
  const selStopRef = useRef<number | null>(null);
  const matchStopRef = useRef<number | null>(null);
  // rAF id driving the playhead while the selection previews.
  const rafRef = useRef<number | null>(null);

  // Drag bookkeeping: x in [0,1] of the canvas at pointer-down, and whether the
  // pointer actually moved (so a plain click clears instead of selecting).
  const dragStartRef = useRef<number | null>(null);
  const draggedRef = useRef(false);

  // ---- audio element lifecycle (create once, clean up on unmount) ----
  useEffect(() => {
    const sel = new Audio();
    const match = new Audio();
    sel.preload = "none";
    match.preload = "none";
    selAudioRef.current = sel;
    matchAudioRef.current = match;
    return () => {
      sel.pause();
      sel.src = "";
      match.pause();
      match.src = "";
      selAudioRef.current = null;
      matchAudioRef.current = null;
      if (selStopRef.current !== null) window.clearTimeout(selStopRef.current);
      if (matchStopRef.current !== null) window.clearTimeout(matchStopRef.current);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ---- stop any in-flight playback (used when switching tracks/unmounting) ----
  const stopSelPlayback = useCallback(() => {
    const el = selAudioRef.current;
    if (el) el.pause();
    if (selStopRef.current !== null) {
      window.clearTimeout(selStopRef.current);
      selStopRef.current = null;
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setPlayhead(null);
  }, []);

  const stopMatchPlayback = useCallback(() => {
    const el = matchAudioRef.current;
    if (el) el.pause();
    if (matchStopRef.current !== null) {
      window.clearTimeout(matchStopRef.current);
      matchStopRef.current = null;
    }
  }, []);

  // ---- load + decode the track's audio whenever the selected track changes ----
  useEffect(() => {
    // Reset all per-track state.
    stopSelPlayback();
    stopMatchPlayback();
    peaksRef.current = null;
    setRegion(null);
    setMatches(null);
    setDecodeErr(null);
    setDuration(0);
    setPeaksReady((n) => n + 1);

    if (track === null) {
      setLoading(false);
      return;
    }

    let alive = true;
    const id = track.id;
    setLoading(true);

    void (async () => {
      try {
        const res = await fetch(api.audioUrl(id));
        if (!res.ok) throw new Error(`audio ${res.status}`);
        const buf = await res.arrayBuffer();
        const ctx = getAudioContext();
        // decodeAudioData detaches the buffer; that's fine, we don't reuse it.
        const audio = await ctx.decodeAudioData(buf);
        if (!alive) return;
        const ch = audio.getChannelData(0);
        peaksRef.current = computePeaks(ch, PEAK_BARS);
        setDuration(audio.duration);
        setLoading(false);
        setPeaksReady((n) => n + 1);
      } catch (e) {
        if (!alive) return;
        peaksRef.current = null;
        setLoading(false);
        setDecodeErr(e instanceof Error ? e.message : String(e));
        setPeaksReady((n) => n + 1);
      }
    })();

    return () => {
      alive = false;
    };
  }, [track, stopSelPlayback, stopMatchPlayback]);

  // ---- draw the waveform + selection + playhead ----
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match the backing store to the displayed size (crisp on HiDPI).
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 1;
    const cssH = canvas.clientHeight || 1;
    const wantW = Math.round(cssW * dpr);
    const wantH = Math.round(cssH * dpr);
    if (canvas.width !== wantW || canvas.height !== wantH) {
      canvas.width = wantW;
      canvas.height = wantH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // dark background
    ctx.fillStyle = "#0b0d12";
    ctx.fillRect(0, 0, cssW, cssH);

    const peaks = peaksRef.current;
    const mid = cssH / 2;

    // center line
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(cssW, mid);
    ctx.stroke();

    if (peaks && peaks.length > 0) {
      const n = peaks.length;
      const barW = cssW / n;
      ctx.fillStyle = "#2dd4bf";
      for (let i = 0; i < n; i++) {
        const h = Math.max(1, (peaks[i] as number) * (cssH - 4));
        const x = i * barW;
        ctx.fillRect(x, mid - h / 2, Math.max(1, barW * 0.8), h);
      }
    }

    // selection overlay
    if (region && duration > 0) {
      const x1 = (region.start / duration) * cssW;
      const x2 = (region.end / duration) * cssW;
      const left = Math.min(x1, x2);
      const w = Math.abs(x2 - x1);
      ctx.fillStyle = "rgba(45, 212, 191, 0.18)";
      ctx.fillRect(left, 0, w, cssH);
      ctx.strokeStyle = "rgba(45, 212, 191, 0.7)";
      ctx.lineWidth = 1;
      ctx.strokeRect(left + 0.5, 0.5, Math.max(0, w - 1), cssH - 1);
    }

    // playhead
    if (playhead !== null && duration > 0) {
      const x = (playhead / duration) * cssW;
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, cssH);
      ctx.stroke();
    }
  }, [region, playhead, duration]);

  // Redraw whenever inputs change (peaks landing bumps peaksReady).
  useEffect(() => {
    draw();
  }, [draw, peaksReady]);

  // Keep the canvas crisp on container resize.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  // ---- region drag selection ----
  const xToSeconds = useCallback(
    (clientX: number): number => {
      const canvas = canvasRef.current;
      if (!canvas || duration <= 0) return 0;
      const rect = canvas.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return frac * duration;
    },
    [duration],
  );

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (duration <= 0 || !peaksRef.current) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      dragStartRef.current = xToSeconds(e.clientX);
      draggedRef.current = false;
      stopSelPlayback();
    },
    [duration, xToSeconds, stopSelPlayback],
  );

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      const start = dragStartRef.current;
      if (start === null) return;
      const cur = xToSeconds(e.clientX);
      // Treat tiny moves as a click, not a drag.
      if (Math.abs(cur - start) > 0.05) draggedRef.current = true;
      const a = Math.min(start, cur);
      const b = Math.max(start, cur);
      setRegion({ start: a, end: b });
    },
    [xToSeconds],
  );

  const onPointerUp = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      const start = dragStartRef.current;
      dragStartRef.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      if (start === null) return;
      if (!draggedRef.current) {
        // A plain click clears the selection.
        setRegion(null);
        setMatches(null);
      }
    },
    [],
  );

  // ---- preview the selection (private audio element) ----
  const playSelection = useCallback(() => {
    if (!region || !track) return;
    const el = selAudioRef.current;
    if (!el) return;
    stopMatchPlayback();
    stopSelPlayback();
    const { start, end } = region;
    el.src = api.audioUrl(track.id);
    el.currentTime = start;
    void el.play().catch((err: unknown) => {
      report(`preview failed: ${err instanceof Error ? err.message : String(err)}`, true);
    });
    // Stop at the region end.
    const ms = Math.max(50, (end - start) * 1000);
    selStopRef.current = window.setTimeout(() => stopSelPlayback(), ms);
    // Drive the playhead.
    const tick = () => {
      const cur = el.currentTime;
      if (cur >= end || el.paused) {
        setPlayhead(null);
        rafRef.current = null;
        return;
      }
      setPlayhead(cur);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [region, track, report, stopSelPlayback, stopMatchPlayback]);

  // ---- play a match from its own start ----
  const playMatch = useCallback(
    (m: Match) => {
      const el = matchAudioRef.current;
      if (!el) return;
      stopSelPlayback();
      stopMatchPlayback();
      el.src = api.audioUrl(m.track_id);
      el.currentTime = m.start;
      void el.play().catch((err: unknown) => {
        report(`play failed: ${err instanceof Error ? err.message : String(err)}`, true);
      });
      const span = Math.max(0, m.end - m.start);
      const ms = span > 0 ? span * 1000 : 6000;
      matchStopRef.current = window.setTimeout(() => stopMatchPlayback(), Math.max(1000, ms));
    },
    [report, stopSelPlayback, stopMatchPlayback],
  );

  // ---- save the labelled region ----
  const saveSound = useCallback(async () => {
    if (!region || !track) return;
    setSaving(true);
    try {
      await api.segmentSave({
        track_id: track.id,
        start: region.start,
        end: region.end,
        label: label.trim() || undefined,
        note: note.trim() || undefined,
      });
      report(`saved “${label.trim() || "sound"}” · ${fmtTime(region.start)}–${fmtTime(region.end)}`);
    } catch (e) {
      report(`save failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      setSaving(false);
    }
  }, [region, track, label, note, report]);

  // ---- find this part elsewhere ----
  const findPart = useCallback(async () => {
    if (!region || !track) return;
    setSearching(true);
    setMatches(null);
    report("searching for matching parts…");
    try {
      const r = await api.segmentSearch(track.id, region.start, region.end);
      setMatches(r.matches);
      if (r.matches.length === 0) {
        report("no matching parts found — try “Index parts” first", false);
      } else {
        report(`found ${r.matches.length} matching part(s)`);
      }
    } catch (e) {
      report(`segment search failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      setSearching(false);
    }
  }, [region, track, report]);

  // ---- define a subgenre from this sound ----
  const makeGenre = useCallback(async () => {
    if (!region || !track) return;
    const name = label.trim();
    if (!name) return;
    setMakingGenre(true);
    report(`creating subgenre “${name}”…`);
    try {
      const r = await api.segmentMakeGenre({
        track_id: track.id,
        start: region.start,
        end: region.end,
        name,
      });
      if (r.ok) {
        report(`created subgenre “${name}” from ${r.examples?.length ?? 0} track(s)`);
        onChanged?.();
      } else {
        report("make subgenre failed", true);
      }
    } catch (e) {
      report(`make subgenre failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      setMakingGenre(false);
    }
  }, [region, track, label, report, onChanged]);

  // ---- build the per-window index, polling progress until done ----
  const indexParts = useCallback(async () => {
    setIndexing(true);
    report("starting parts index…");
    try {
      await api.segmentIndex();
      // Poll progress every ~700ms until the job stops running.
      await new Promise<void>((resolve, reject) => {
        const poll = window.setInterval(() => {
          void (async () => {
            try {
              const p = await api.progress();
              if (p.error) {
                window.clearInterval(poll);
                reject(new Error(p.error));
                return;
              }
              report(`indexing parts… ${p.done}/${p.total}`);
              if (!p.running) {
                window.clearInterval(poll);
                resolve();
              }
            } catch (err) {
              window.clearInterval(poll);
              reject(err instanceof Error ? err : new Error(String(err)));
            }
          })();
        }, 700);
      });
      report("parts indexed");
    } catch (e) {
      report(`index failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      setIndexing(false);
    }
  }, [report]);

  if (track === null) {
    return <div className="hint">Select a track to inspect its waveform.</div>;
  }

  const hasRegion = region !== null && region.end - region.start > 0.02;
  const stateMsg = loading
    ? "Loading waveform…"
    : decodeErr !== null
      ? `Could not decode audio: ${decodeErr}`
      : peaksRef.current === null
        ? "No waveform"
        : null;

  return (
    <div className="wave">
      <div className="wave-head">
        <span className="wave-head__name" title={track.name}>
          {track.name}
        </span>
      </div>

      <div className="wave-canvas-wrap">
        <canvas
          ref={canvasRef}
          className="wave-canvas"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
        {stateMsg !== null && (
          <div
            className={
              decodeErr !== null
                ? "wave-canvas-state wave-canvas-state--error"
                : "wave-canvas-state"
            }
          >
            {stateMsg}
          </div>
        )}
      </div>

      <div className="wave-times">
        {hasRegion && region ? (
          <span className="wave-times__sel">
            {fmtTime(region.start)} – {fmtTime(region.end)}
          </span>
        ) : (
          <span className="wave-times__hint">drag on the waveform to select a part</span>
        )}
        <span className="wave-times__dur">{fmtTime(duration)}</span>
      </div>

      <div className="wave-transport">
        <button
          className="btn btn--play btn--xs"
          onClick={playSelection}
          disabled={!hasRegion}
          title="Play the selected region"
        >
          ▶ Play part
        </button>
      </div>

      <div className="wave-form">
        <input
          className="wave-input"
          type="text"
          placeholder="what is this sound? (e.g. electroclash cowbell)"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
        />
        <input
          className="wave-input"
          type="text"
          placeholder="note (optional)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
        <div className="wave-actions">
          <button
            className="btn btn--accent btn--sm"
            onClick={() => void saveSound()}
            disabled={!hasRegion || saving}
          >
            {saving ? "Saving…" : "Save this sound"}
          </button>
          <button
            className="btn btn--go btn--sm"
            onClick={() => void findPart()}
            disabled={!hasRegion || searching}
          >
            {searching ? "Finding…" : "Find this part elsewhere"}
          </button>
          <button
            className="btn btn--accent btn--sm"
            onClick={() => void makeGenre()}
            disabled={!hasRegion || !label.trim() || makingGenre}
            title="Create a subgenre seeded from every track that has this sound"
          >
            {makingGenre ? "Creating…" : "Make subgenre from this"}
          </button>
        </div>
      </div>

      <div className="wave-index">
        <button
          className="btn btn--xs"
          onClick={() => void indexParts()}
          disabled={indexing}
        >
          {indexing ? "Indexing…" : "Index parts"}
        </button>
        <span className="wave-index__note">
          Builds the part index used by search. Run once after adding tracks.
        </span>
      </div>

      {matches !== null && (
        <div className="wave-matches">
          <div className="wave-matches__head">
            {matches.length > 0
              ? `Matches (${matches.length})`
              : "No matches"}
          </div>
          {matches.length === 0 ? (
            <div className="hint">
              Nothing found. If you haven’t yet, hit “Index parts” and try again.
            </div>
          ) : (
            matches.map((m, i) => {
              const score = Math.max(0, Math.min(1, m.score));
              return (
                <div className="wave-match" key={`${m.track_id}-${m.start}-${i}`}>
                  <div className="wave-match__body">
                    <span className="wave-match__name" title={m.name}>
                      {m.name}
                    </span>
                    <div className="wave-match__row">
                      <span className="wave-match__bar">
                        <span
                          className="wave-match__fill"
                          style={{ width: `${score * 100}%` }}
                        />
                      </span>
                      <span className="wave-match__at">at {fmtTime(m.start)}</span>
                    </div>
                  </div>
                  <button
                    className="btn btn--play btn--xs wave-match__play"
                    onClick={() => playMatch(m)}
                    title={`Play ${m.name} from ${fmtTime(m.start)}`}
                    aria-label={`Play ${m.name}`}
                  >
                    ▶
                  </button>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
