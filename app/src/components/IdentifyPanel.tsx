import { useState } from "react";
import { api } from "../api";
import { Meter } from "./Meter";

interface IdentifyPanelProps {
  report: (msg: string, isError?: boolean) => void;
  /** Play a matched track via the shared audio element. */
  onPlay: (id: number) => void;
}

interface Match {
  track_id: number;
  name: string;
  score: number;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function IdentifyPanel({ report, onPlay }: IdentifyPanelProps) {
  const [path, setPath] = useState("");
  const [identifying, setIdentifying] = useState(false);
  const [matches, setMatches] = useState<Match[] | null>(null);

  const identify = async () => {
    const p = path.trim().replace(/^["']|["']$/g, "");
    if (!p) {
      report("paste a file path to identify", true);
      return;
    }
    setIdentifying(true);
    report("identifying…");
    try {
      const r = await api.identify(p);
      setMatches(r.matches);
      report(`identified · ${r.matches.length} match(es)`);
    } catch (e) {
      report(`identify failed: ${errMsg(e)}`, true);
    } finally {
      setIdentifying(false);
    }
  };

  return (
    <>
      <div className="panel-section">
        <h3>Identify</h3>
        <div className="form">
          <div className="form__row">
            <label>File path</label>
            <input
              type="text"
              className="input--path"
              placeholder="paste a file path to identify…"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void identify();
              }}
            />
          </div>
          <button
            className="btn btn--go btn--xs"
            onClick={() => void identify()}
            disabled={identifying || !path.trim()}
          >
            {identifying ? "Identifying…" : "Identify"}
          </button>
        </div>
        <div className="hint">Matches against tracks already in your library.</div>
      </div>

      {matches && (
        <div className="panel-section">
          <h3>Matches ({matches.length})</h3>
          {matches.length === 0 ? (
            <div className="hint">No matches found.</div>
          ) : (
            <div className="sim-list">
              {matches.map((m) => (
                <div className="sim-row" key={m.track_id}>
                  <button
                    className="btn btn--play btn--xs"
                    onClick={() => onPlay(m.track_id)}
                    aria-label={`Play ${m.name}`}
                  >
                    ▶
                  </button>
                  <span className="sim-row__name" title={m.name}>
                    {m.name}
                  </span>
                  <Meter value={m.score} color="#2dd4bf" />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}
