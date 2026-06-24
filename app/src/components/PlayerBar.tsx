import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

interface PlayerBarProps {
  audioRef: React.RefObject<HTMLAudioElement | null>;
  trackId: number | null;
  trackName: string | null;
  /** BPM of the playing track, so the zoom window can be ~4 bars. */
  bpm?: number | null;
  /** Find tracks with a similar frequency fingerprint to the playing track. */
  onSimilarSound?: (trackId: number) => void;
}

type Wave = { bass: number[]; mid: number[]; high: number[]; start: number; end: number };

const BASS = "rgb(235,70,70)";
const MID = "rgb(80,205,95)";
const HIGH = "rgb(80,150,235)";

/**
 * Bottom transport: play/pause and a Rekordbox/Serato-style RGB spectral waveform
 * (low=red, mid=green, high=blue, height=loudness). Two views: a whole-track overview,
 * and a high-resolution "4 bars" zoom that scrolls past a fixed centre playhead as the
 * track plays (the engine renders a high-res slice for the visible window).
 */
export function PlayerBar({ audioRef, trackId, trackName, bpm, onSimilarSound }: PlayerBarProps) {
  const [full, setFull] = useState<Wave | null>(null); // whole-track overview
  const [buf, setBuf] = useState<Wave | null>(null); // hi-res window buffer (zoom)
  const [zoom, setZoom] = useState(false);
  const [showWave, setShowWave] = useState(true);
  const [pos, setPos] = useState(0); // 0..1
  const [playing, setPlaying] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const durRef = useRef(0);
  const fetchingRef = useRef(false);

  // zoom window length in seconds (~4 bars = 16 beats), from BPM when known.
  const winSec = bpm && bpm > 0 ? (16 * 60) / bpm : 8;

  // whole-track hi-res overview on track change
  useEffect(() => {
    setFull(null);
    setBuf(null);
    setPos(0);
    if (trackId == null) return;
    let alive = true;
    api.waveform(trackId, 2000).then((w) => { if (alive) setFull(w as Wave); }).catch(() => {});
    return () => { alive = false; };
  }, [trackId]);

  // mirror the <audio> element + capture duration
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => { durRef.current = el.duration || 0; setPos(el.duration ? el.currentTime / el.duration : 0); };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [audioRef]);

  // smooth playhead/scroll while playing (timeupdate alone is too choppy)
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const el = audioRef.current;
      if (el && el.duration) setPos(el.currentTime / el.duration);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, audioRef]);

  // in zoom mode keep a hi-res buffer around the playhead; refetch as it nears the edge
  useEffect(() => {
    if (!zoom || trackId == null) return;
    const dur = durRef.current;
    if (!dur) return;
    const t = pos * dur;
    const need = !buf || t < buf.start + winSec * 0.25 || t > buf.end - winSec * 0.25;
    if (need && !fetchingRef.current) {
      fetchingRef.current = true;
      const s = Math.max(0, t - winSec * 1.5);
      const e = Math.min(dur, t + winSec * 1.5);
      api.waveform(trackId, 1400, s, e)
        .then((w) => setBuf(w as Wave))
        .catch(() => {})
        .finally(() => { fetchingRef.current = false; });
    }
  }, [zoom, pos, trackId, buf, winSec]);

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !showWave) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    c.width = w * dpr;
    c.height = h * dpr;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const dur = durRef.current;

    // choose data + visible array fraction + playhead x
    let data: Wave | null;
    let f0 = 0;
    let f1 = 1;
    let headX = pos * w;
    if (zoom && buf && dur) {
      data = buf;
      const t = pos * dur;
      const span = (buf.end - buf.start) || 1;
      f0 = (t - winSec / 2 - buf.start) / span;
      f1 = (t + winSec / 2 - buf.start) / span;
      headX = w / 2; // playhead fixed at centre, waveform scrolls past
    } else {
      data = full;
    }
    if (!data || data.bass.length === 0) return;
    const N = data.bass.length;
    const i0 = Math.max(0, Math.floor(f0 * N));
    const i1 = Math.min(N, Math.ceil(f1 * N));
    const count = Math.max(1, i1 - i0);
    let peak = 1e-6;
    for (let i = i0; i < i1; i++) peak = Math.max(peak, data.bass[i] + data.mid[i] + data.high[i]);
    const bw = w / count;
    const cy = h / 2;
    for (let k = 0; k < count; k++) {
      const i = i0 + k;
      const b = data.bass[i] ?? 0;
      const m = data.mid[i] ?? 0;
      const hi = data.high[i] ?? 0;
      const tot = b + m + hi;
      if (tot <= 0) continue;
      const barH = (tot / peak) * h;
      const x = k * bw;
      const bwp = Math.max(1, bw - 0.3);
      let y = cy + barH / 2;
      for (const [val, col] of [[b, BASS], [m, MID], [hi, HIGH]] as [number, string][]) {
        const seg = (val / tot) * barH;
        ctx.fillStyle = col;
        ctx.fillRect(x, y - seg, bwp, seg);
        y -= seg;
      }
    }
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillRect(headX - 1, 0, 2, h);
  }, [full, buf, zoom, pos, showWave, winSec]);

  const seek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const el = audioRef.current;
    if (!el || !el.duration) return;
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    if (zoom && durRef.current) {
      const t = pos * durRef.current;
      el.currentTime = Math.max(0, Math.min(durRef.current, t - winSec / 2 + frac * winSec));
    } else {
      el.currentTime = frac * el.duration;
    }
  }, [audioRef, zoom, pos, winSec]);

  const toggle = useCallback(() => {
    const el = audioRef.current;
    if (!el || !el.src) return;
    if (playing) el.pause(); else void el.play();
  }, [audioRef, playing]);

  return (
    <div className="playerbar">
      <button className="playerbar__btn" onClick={toggle} title={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "►"}
      </button>
      <div className="playerbar__name" title={trackName ?? ""}>
        {trackName ?? "nothing playing"}
      </div>
      <div className="playerbar__track" onClick={seek}>
        {showWave ? (
          <canvas ref={canvasRef} className="playerbar__canvas" />
        ) : (
          <div className="playerbar__line" />
        )}
      </div>
      <button
        className={zoom ? "playerbar__toggle active" : "playerbar__toggle"}
        onClick={() => setZoom((z) => !z)}
        title="Zoom to ~4 bars (scrolls as it plays)"
      >
        {zoom ? "⊟ 4 bars" : "⊞ zoom"}
      </button>
      <button className="playerbar__toggle" onClick={() => setShowWave((s) => !s)} title="Toggle waveform">
        {showWave ? "～ wave" : "— line"}
      </button>
      {onSimilarSound && trackId != null && (
        <button className="playerbar__toggle" title="Find tracks with a similar frequency profile"
                onClick={() => onSimilarSound(trackId)}>
          ≈ similar sound
        </button>
      )}
    </div>
  );
}
