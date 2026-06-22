import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

interface PlayerBarProps {
  audioRef: React.RefObject<HTMLAudioElement | null>;
  trackId: number | null;
  trackName: string | null;
  /** Find tracks with a similar frequency fingerprint to the playing track. */
  onSimilarSound?: (trackId: number) => void;
}

type Wave = { bass: number[]; mid: number[]; high: number[] };

/**
 * Bottom transport bar: play/pause, a Rekordbox-style RGB spectral waveform
 * (low=red, mid=green, high=blue, brightness=loudness) with a live playhead and
 * click-to-seek, a waveform/line toggle, and a "similar sound" action.
 */
export function PlayerBar({ audioRef, trackId, trackName, onSimilarSound }: PlayerBarProps) {
  const [wave, setWave] = useState<Wave | null>(null);
  const [showWave, setShowWave] = useState(true);
  const [pos, setPos] = useState(0); // 0..1
  const [playing, setPlaying] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // fetch the spectral waveform whenever the track changes
  useEffect(() => {
    setWave(null);
    setPos(0);
    if (trackId == null) return;
    let alive = true;
    api.waveform(trackId).then((w) => { if (alive) setWave(w); }).catch(() => {});
    return () => { alive = false; };
  }, [trackId]);

  // mirror the <audio> element's state
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => setPos(el.duration ? el.currentTime / el.duration : 0);
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

  // draw the RGB waveform
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !wave || !showWave) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const n = wave.bass.length;
    const bw = w / n;
    for (let i = 0; i < n; i++) {
      const b = wave.bass[i], m = wave.mid[i], hi = wave.high[i];
      const amp = Math.max(b, m, hi);
      const bh = Math.max(1, amp * h);
      ctx.fillStyle = `rgb(${Math.round(40 + b * 215)},${Math.round(30 + m * 200)},${Math.round(50 + hi * 205)})`;
      ctx.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 0.3), bh);
    }
  }, [wave, showWave]);

  const seek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const el = audioRef.current;
    if (!el || !el.duration) return;
    const r = e.currentTarget.getBoundingClientRect();
    el.currentTime = ((e.clientX - r.left) / r.width) * el.duration;
  }, [audioRef]);

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
        {showWave && wave ? (
          <canvas ref={canvasRef} className="playerbar__canvas" />
        ) : (
          <div className="playerbar__line" />
        )}
        <div className="playerbar__playhead" style={{ left: `${pos * 100}%` }} />
      </div>
      <button className="playerbar__toggle" onClick={() => setShowWave((s) => !s)}
              title="Toggle waveform">
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
