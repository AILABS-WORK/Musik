import { useEffect, useState } from "react";
import type { Track } from "../types";
import { api } from "../api";
import { Meter } from "./Meter";

interface SongPanelProps {
  track: Track | null;
  onPlay: (id: number) => void;
}

/**
 * Local narrowing of the per-song understanding record. `api.understanding`
 * returns a loosely-typed object (vocal/mood are `any`, instruments and
 * tags_canonical are not on its signature), so we describe the runtime shape
 * here and read it defensively — every field is optional and validated before
 * use, so nothing untyped leaks into the render logic below.
 */
interface VocalInfo {
  voice_instrumental?: string | null;
  gender?: string | null;
  gender_conf?: number | null;
  sung_score?: number | null;
}

interface MoodInfo {
  arousal?: number | null;
  valence?: number | null;
}

interface Understanding {
  track_id: number;
  top_tags: { label: string; prob: number }[];
  instruments?: Record<string, number> | null;
  vocal?: VocalInfo | null;
  mood?: MoodInfo | null;
  caption?: string | null;
  tags_canonical?: string[] | null;
  deep_done?: number;
}

/** Finite-number guard that also rejects null/undefined. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Format a 0..1 energy value as a whole percentage, or "—". */
function pct(v: number | null): string {
  return v === null ? "—" : `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="song-stat">
      <span className="song-stat__label">{label}</span>
      <span className="song-stat__value mono">{value}</span>
    </div>
  );
}

/** A small valence/arousal pad with a glowing dot at the mood coordinate. */
function MoodPad({ valence, arousal }: { valence: number; arousal: number }) {
  const size = 120;
  const vx = Math.max(0, Math.min(1, valence));
  const ay = Math.max(0, Math.min(1, arousal));
  const cx = vx * size;
  const cy = (1 - ay) * size; // arousal up => smaller y
  return (
    <svg
      className="song-mood__pad"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`mood — valence ${vx.toFixed(2)}, arousal ${ay.toFixed(2)}`}
    >
      <rect x={0} y={0} width={size} height={size} className="song-mood__bg" />
      {/* faint quadrant grid */}
      <line x1={size / 2} y1={0} x2={size / 2} y2={size} className="song-mood__grid" />
      <line x1={0} y1={size / 2} x2={size} y2={size / 2} className="song-mood__grid" />
      {/* glowing dot */}
      <circle cx={cx} cy={cy} r={9} className="song-mood__glow" />
      <circle cx={cx} cy={cy} r={4} className="song-mood__dot" />
      {/* subtle axis labels */}
      <text x={size - 3} y={size / 2 - 4} className="song-mood__axis song-mood__axis--end">
        valence+
      </text>
      <text x={3} y={12} className="song-mood__axis">
        arousal+
      </text>
    </svg>
  );
}

export function SongPanel({ track, onPlay }: SongPanelProps) {
  const [u, setU] = useState<Understanding | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (track === null) {
      setU(null);
      setErr(null);
      return;
    }
    let alive = true;
    const id = track.id;
    setLoading(true);
    setErr(null);
    void (async () => {
      try {
        const res = (await api.understanding(id)) as Understanding | null;
        if (alive) setU(res);
      } catch (e) {
        if (alive) {
          setU(null);
          setErr(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [track]);

  if (track === null) {
    return <div className="hint">Select a track to see its sound profile.</div>;
  }

  const bpm = num(track.bpm);
  const energy = num(track.energy);

  // Stat tiles are always available from the Track itself.
  const tiles = (
    <div className="song-stats">
      <StatTile label="BPM" value={bpm === null ? "—" : String(Math.round(bpm))} />
      <StatTile label="Key" value={track.music_key ?? "—"} />
      <StatTile label="Energy" value={pct(energy)} />
    </div>
  );

  const header = (
    <div className="song-head">
      <span className="song-head__name" title={track.name}>
        {track.name}
      </span>
      <button
        className="btn btn--play btn--xs"
        onClick={() => onPlay(track.id)}
        aria-label={`Play ${track.name}`}
      >
        ▶
      </button>
    </div>
  );

  if (loading) {
    return (
      <div className="song-card">
        {header}
        {tiles}
        <div className="hint">Loading…</div>
      </div>
    );
  }

  const instruments = u?.instruments ?? null;
  const hasInstruments = !!instruments && Object.keys(instruments).length > 0;
  const vocal = u?.vocal ?? null;
  const untagged = !u || (!hasInstruments && !vocal);

  if (untagged) {
    return (
      <div className="song-card">
        {header}
        {tiles}
        {err !== null ? (
          <div className="hint">Could not load sound profile: {err}</div>
        ) : (
          <div className="hint">Not tagged yet — hit Tag to analyze the sound.</div>
        )}
      </div>
    );
  }

  // ---- tagged: build the rich detail body ----
  const caption = u?.caption ?? null;

  const voiceKind = vocal?.voice_instrumental ?? null;
  const gender = vocal?.gender ?? null;
  const showGender = gender !== null && gender.toLowerCase() !== "unknown";
  const sung = num(vocal?.sung_score);

  const instrChips: { name: string; prob: number }[] = hasInstruments
    ? Object.entries(instruments as Record<string, number>)
        .map(([name, prob]) => ({ name, prob: num(prob) ?? 0 }))
        .sort((a, b) => b.prob - a.prob)
    : [];

  const valence = num(u?.mood?.valence);
  const arousal = num(u?.mood?.arousal);
  const hasMood = valence !== null && arousal !== null;

  const tags = (u?.tags_canonical ?? []).filter((t) => t.length > 0);

  return (
    <div className="song-card">
      {header}

      {caption !== null && caption.length > 0 && (
        <blockquote className="song-caption">{caption}</blockquote>
      )}

      {tiles}

      <div className="song-block">
        <h3 className="song-block__title">Vocal</h3>
        {voiceKind !== null || sung !== null ? (
          <div className="song-vocal">
            <div className="song-vocal__row">
              <span className="song-vocal__kind">{voiceKind ?? "—"}</span>
              {showGender && <span className="song-chip song-chip--gender">{gender}</span>}
            </div>
            {sung !== null && (
              <div className="song-vocal__meter">
                <span className="song-vocal__meter-label">sung</span>
                <Meter value={sung} color="var(--accent-2)" />
              </div>
            )}
          </div>
        ) : (
          <div className="dash">—</div>
        )}
      </div>

      <div className="song-block">
        <h3 className="song-block__title">Instruments</h3>
        {instrChips.length > 0 ? (
          <div className="song-chips">
            {instrChips.map((it) => (
              <span
                className="song-chip song-chip--instr"
                key={it.name}
                style={{ opacity: 0.4 + 0.6 * Math.max(0, Math.min(1, it.prob)) }}
                title={`${it.name} · ${it.prob.toFixed(2)}`}
              >
                <span className="song-chip__label">{it.name}</span>
                <span className="song-chip__prob mono">{it.prob.toFixed(2)}</span>
              </span>
            ))}
          </div>
        ) : (
          <div className="dash">—</div>
        )}
      </div>

      <div className="song-block">
        <h3 className="song-block__title">Mood</h3>
        {hasMood ? (
          <div className="song-mood">
            <MoodPad valence={valence} arousal={arousal} />
            <div className="song-mood__readout mono">
              <div>val {valence.toFixed(2)}</div>
              <div>aro {arousal.toFixed(2)}</div>
            </div>
          </div>
        ) : (
          <div className="dash">—</div>
        )}
      </div>

      <div className="song-block">
        <h3 className="song-block__title">Tags</h3>
        {tags.length > 0 ? (
          <div className="song-chips">
            {tags.map((t) => (
              <span className="song-chip song-chip--tag" key={t}>
                {t}
              </span>
            ))}
          </div>
        ) : (
          <div className="dash">—</div>
        )}
      </div>
    </div>
  );
}
