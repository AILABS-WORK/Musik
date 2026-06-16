import type { SimilarItem, Track } from "../types";
import { Meter } from "./Meter";

interface SimilarPanelProps {
  selected: Track | null;
  items: SimilarItem[];
  loading: boolean;
  onPlay: (id: number) => void;
  /** Start an auto-advancing radio queue seeded from the selected track. */
  onRadio: (id: number) => void;
}

export function SimilarPanel({
  selected,
  items,
  loading,
  onPlay,
  onRadio,
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
      <div className="apply-group__head">
        <span className="apply-group__title">Similar to “{selected.name}”</span>
        <button
          className="btn btn--accent btn--xs"
          onClick={() => onRadio(selected.id)}
          title="Auto-advancing radio seeded from this track"
        >
          📻 Radio
        </button>
      </div>
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
                title={`Play “${it.name}”`}
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
