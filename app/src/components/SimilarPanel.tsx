import type { SimilarItem, Track } from "../types";
import { Meter } from "./Meter";

interface SimilarPanelProps {
  selected: Track | null;
  items: SimilarItem[];
  loading: boolean;
  onPlay: (id: number) => void;
}

export function SimilarPanel({
  selected,
  items,
  loading,
  onPlay,
}: SimilarPanelProps) {
  if (!selected) {
    return (
      <div className="hint">
        Select a track in the table to see similar tracks here.
      </div>
    );
  }

  return (
    <div className="panel-section">
      <h3>Similar to “{selected.name}”</h3>
      {loading ? (
        <div className="hint">Loading…</div>
      ) : items.length === 0 ? (
        <div className="hint">No similar tracks found.</div>
      ) : (
        <div className="sim-list">
          {items.map((it) => (
            <div className="sim-row" key={it.track_id}>
              <button
                className="btn btn--play btn--xs"
                onClick={() => onPlay(it.track_id)}
                aria-label={`Play ${it.name}`}
              >
                ▶
              </button>
              <span className="sim-row__name" title={it.name}>
                {it.name}
              </span>
              <Meter value={it.score} color="#2dd4bf" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
