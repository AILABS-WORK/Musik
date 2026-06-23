import { useEffect, useState } from "react";
import type { Genre, Track } from "../types";
import { api } from "../api";
import { Meter } from "./Meter";

interface SongPanelProps {
  track: Track | null;
  /** All genres, for the "label as" picker (datalist of existing subgenres). */
  genres?: Genre[];
  onPlay: (id: number) => void;
  /** Called after a relabel so the parent (e.g. the track table) can refresh. */
  onChanged?: () => void;
  /** Surface a status message (errors when isError=true). */
  report?: (m: string, e?: boolean) => void;
}

/** One genre suggestion from the blend (multi-label, rank 0 = primary/stored). */
interface Suggestion {
  genre_id: number;
  name: string;
  confidence: number;
  rank: number;
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
  /** Human-readable named moods, e.g. ["driving","hypnotic","dark"]. */
  tags?: string[] | null;
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

export function SongPanel({ track, genres = [], onPlay, onChanged, report }: SongPanelProps) {
  const [u, setU] = useState<Understanding | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // ---- genre blend (multi-label suggestions, best first) ----
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [relabeling, setRelabeling] = useState<number | null>(null);
  // ---- "label as" picker (assign to any existing or a brand-new subgenre) ----
  const [labelInput, setLabelInput] = useState("");
  const [assigning, setAssigning] = useState(false);

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

  // Fetch the genre blend whenever the selected track changes.
  useEffect(() => {
    if (track === null) {
      setSuggestions([]);
      return;
    }
    let alive = true;
    const id = track.id;
    void (async () => {
      try {
        const res = await api.trackSuggestions(id);
        if (alive) setSuggestions(res.suggestions);
      } catch {
        // A missing blend isn't an error worth surfacing — show the hint instead.
        if (alive) setSuggestions([]);
      }
    })();
    return () => {
      alive = false;
    };
  }, [track]);

  // Relabel: confirm a non-primary genre, then re-fetch the blend and notify.
  const relabel = async (genreId: number, name: string) => {
    if (track === null) return;
    const id = track.id;
    setRelabeling(genreId);
    try {
      await api.confirm({ track_id: id, genre_id: genreId });
      const res = await api.trackSuggestions(id);
      setSuggestions(res.suggestions);
      onChanged?.();
      report?.(`relabeled to ${name}`);
    } catch (e) {
      report?.(
        `relabel failed: ${e instanceof Error ? e.message : String(e)}`,
        true,
      );
    } finally {
      setRelabeling(null);
    }
  };

  // Label the track as a subgenre (existing -> confirm + becomes an exemplar/anchor;
  // new -> create by example). Either way it anchors the by-example re-sort.
  const assignLabel = async () => {
    if (track === null) return;
    const name = labelInput.trim();
    if (!name) return;
    setAssigning(true);
    try {
      const existing = genres.find(
        (g) => g.level === "subgenre" && g.name.toLowerCase() === name.toLowerCase(),
      );
      if (existing) {
        await api.confirm({ track_id: track.id, genre_id: existing.id });
      } else {
        await api.byExample({ name, track_ids: [track.id], level: "subgenre" });
      }
      setLabelInput("");
      onChanged?.();
      report?.(`labeled “${track.name}” as ${name} — Re-sort to propagate`);
    } catch (e) {
      report?.(`label failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      setAssigning(false);
    }
  };

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
        title={`Play “${track.name}”`}
      >
        ▶
      </button>
    </div>
  );

  // ---- genre blend: primary chip (rank 0) + clickable alternatives ----
  // Shown near the top so it's always available, regardless of tag state.
  const genreBlock = (
    <div className="song-block">
      <h3 className="song-block__title">Genre</h3>
      <div className="song-curgenre">
        <span className="song-curgenre__lab">now</span>
        <span className="song-curgenre__val">{track.genre ?? "unsorted"}</span>
        {track.confidence != null && (
          <span className="song-curgenre__conf mono">
            {Math.round(Math.max(0, Math.min(1, track.confidence)) * 100)}%
          </span>
        )}
      </div>
      {suggestions.length > 0 && (
        <div className="song-chips">
          {suggestions.map((s) => {
            const primary = s.rank === 0;
            const score = `${Math.round(Math.max(0, Math.min(1, s.confidence)) * 100)}%`;
            return (
              <button
                key={s.genre_id}
                type="button"
                className={primary ? "song-genre song-genre--primary" : "song-genre song-genre--alt"}
                disabled={primary || relabeling !== null}
                onClick={primary ? undefined : () => void relabel(s.genre_id, s.name)}
                title={primary ? `${s.name} — current genre` : `Relabel to “${s.name}”`}
              >
                <span className="song-genre__name">{s.name}</span>
                <span className="song-genre__score mono">{score}</span>
              </button>
            );
          })}
        </div>
      )}
      <div className="song-label">
        <input
          className="song-label__input"
          list="song-subgenres"
          placeholder="label as subgenre…"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") void assignLabel(); }}
        />
        <datalist id="song-subgenres">
          {genres.filter((g) => g.level === "subgenre").map((g) => (
            <option key={g.id} value={g.name} />
          ))}
        </datalist>
        <button
          className="btn btn--xs btn--accent"
          disabled={assigning || !labelInput.trim()}
          onClick={() => void assignLabel()}
        >
          {assigning ? "…" : "Label"}
        </button>
      </div>
      <div className="song-label__hint">
        pick an existing subgenre or type a new one — then Re-sort to propagate by sound
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="song-card">
        {header}
        {genreBlock}
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
        {genreBlock}
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

  // Named, human-readable moods (e.g. "driving", "hypnotic", "dark").
  const moodTags = (u?.mood?.tags ?? []).filter(
    (t): t is string => typeof t === "string" && t.length > 0,
  );

  const tags = (u?.tags_canonical ?? []).filter((t) => t.length > 0);

  return (
    <div className="song-card">
      {header}

      {caption !== null && caption.length > 0 && (
        <blockquote className="song-caption">{caption}</blockquote>
      )}

      {genreBlock}

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
        {moodTags.length > 0 && (
          <div className="song-chips song-moodtags">
            {moodTags.map((t) => (
              <span className="song-chip song-chip--mood" key={t}>
                {t}
              </span>
            ))}
          </div>
        )}
        {hasMood ? (
          <div className="song-mood">
            <MoodPad valence={valence} arousal={arousal} />
            <div className="song-mood__readout mono">
              <div>val {valence.toFixed(2)}</div>
              <div>aro {arousal.toFixed(2)}</div>
            </div>
          </div>
        ) : moodTags.length === 0 ? (
          <div className="dash">—</div>
        ) : null}
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
