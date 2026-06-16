import { useState } from "react";
import { api } from "../api";

interface MixPanelProps {
  report: (msg: string, isError?: boolean) => void;
  onPlay: (trackId: number) => void;
}

interface Seg {
  start: number;
  end: number;
  track_id: number;
  name: string;
  score: number;
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Drop a whole mix/DJ-set -> a timestamped tracklist of tracks from YOUR library. */
export function MixPanel({ report, onPlay }: MixPanelProps) {
  const [path, setPath] = useState("");
  const [segs, setSegs] = useState<Seg[] | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    const p = path.trim().replace(/^["']|["']$/g, "");
    if (!p) return;
    setBusy(true);
    try {
      const r = await api.identifyMix(p);
      setSegs(r.segments);
      report(`mix tracklist · ${r.segments.length} segment(s)`);
    } catch (e) {
      report(`mix id failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h3 className="panel-h">Mix tracklist</h3>
      <div className="importbar__row">
        <input
          className="importbar__input"
          placeholder="paste a mix / set file path…"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") run();
          }}
        />
        <button className="btn btn--go btn--xs" onClick={run} disabled={busy || !path.trim()}>
          {busy ? "…" : "Tracklist"}
        </button>
      </div>
      <p className="panel-hint">Finds tracks from <b>your library</b> inside a mix, with timestamps.</p>

      {segs &&
        (segs.length === 0 ? (
          <p className="panel-hint">No library tracks recognised in this mix.</p>
        ) : (
          <ol className="mixlist">
            {segs.map((s, i) => (
              <li className="mixlist__item" key={i}>
                <button className="mixlist__play" onClick={() => onPlay(s.track_id)} title="play this track">
                  ▶
                </button>
                <span className="mixlist__time">
                  {fmt(s.start)}–{fmt(s.end)}
                </span>
                <span className="mixlist__name" title={s.name}>
                  {s.name}
                </span>
                <span className="mixlist__score">{s.score.toFixed(2)}</span>
              </li>
            ))}
          </ol>
        ))}
    </>
  );
}
