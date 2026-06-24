import { useState } from "react";
import type { SimilarItem, Track } from "../types";
import { api } from "../api";
import { Meter } from "./Meter";

interface SimilarPanelProps {
  selected: Track | null;
  items: SimilarItem[];
  loading: boolean;
  onPlay: (id: number) => void;
  /** Start an auto-advancing radio queue seeded from the selected track. */
  onRadio: (id: number) => void;
}

interface BandMatch {
  name: string;
  a: number;
  b: number;
  match: number;
}

interface Explanation {
  score: number | null;
  shared: string[];
  different: string[];
  bands: BandMatch[];
}

export function SimilarPanel({
  selected,
  items,
  loading,
  onPlay,
  onRadio,
}: SimilarPanelProps) {
  // which row's "why" is open, and the cached explanation per other-track id
  const [openId, setOpenId] = useState<number | null>(null);
  const [explain, setExplain] = useState<Record<number, Explanation | "loading">>({});

  if (!selected) {
    return (
      <div className="hint">
        Select a track in the table to see similar tracks here.
      </div>
    );
  }

  const sel = selected;
  const toggleWhy = async (otherId: number) => {
    if (openId === otherId) {
      setOpenId(null);
      return;
    }
    setOpenId(otherId);
    if (explain[otherId] === undefined) {
      setExplain((m) => ({ ...m, [otherId]: "loading" }));
      try {
        const r = await api.explain(sel.id, otherId);
        setExplain((m) => ({
          ...m,
          [otherId]: { score: r.score, shared: r.shared, different: r.different, bands: r.bands ?? [] },
        }));
      } catch {
        setExplain((m) => ({ ...m, [otherId]: { score: null, shared: [], different: [], bands: [] } }));
      }
    }
  };

  return (
    <div className="panel-section">
      <div className="apply-group__head">
        <span className="apply-group__title">Similar to “{sel.name}”</span>
        <button
          className="btn btn--accent btn--xs"
          onClick={() => onRadio(sel.id)}
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
          {items.map((it) => {
            const ex = explain[it.track_id];
            const open = openId === it.track_id;
            return (
              <div className="sim-rowwrap" key={it.track_id}>
                <div className="sim-row">
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
                  <button
                    className="sim-why"
                    onClick={() => void toggleWhy(it.track_id)}
                    title="Why is this similar?"
                  >
                    {open ? "×" : "why"}
                  </button>
                </div>
                {open && (
                  <div className="sim-whybox">
                    {ex === undefined || ex === "loading" ? (
                      <span className="hint">comparing…</span>
                    ) : ex.shared.length === 0 && ex.different.length === 0 && ex.bands.length === 0 ? (
                      <span className="hint">Tag + Analyze both tracks for a detailed comparison.</span>
                    ) : (
                      <>
                        {ex.bands.length > 0 && (
                          <div className="sim-bands" title="Where the two tracks match across the frequency spectrum">
                            {ex.bands.map((bd) => {
                              const pct = Math.round(bd.match * 100);
                              // green = matched, red = differs, for this frequency range
                              const hue = Math.round(bd.match * 130);
                              return (
                                <div className="sim-band" key={bd.name}>
                                  <span className="sim-band__name">{bd.name}</span>
                                  <span className="sim-band__bar">
                                    <span
                                      className="sim-band__fill"
                                      style={{ width: `${pct}%`, background: `hsl(${hue},70%,50%)` }}
                                    />
                                  </span>
                                  <span className="sim-band__pct mono">{pct}</span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                        {ex.shared.length > 0 && (
                          <div className="sim-shares">
                            <span className="sim-tag sim-tag--yes">shares</span>{" "}
                            {ex.shared.join(" · ")}
                          </div>
                        )}
                        {ex.different.length > 0 && (
                          <div className="sim-diffs">
                            <span className="sim-tag sim-tag--no">differs</span>{" "}
                            {ex.different.join(" · ")}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
