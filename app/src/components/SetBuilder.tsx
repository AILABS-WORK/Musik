import { useState } from "react";
import { api } from "../api";

interface SetBuilderProps {
  report: (msg: string, isError?: boolean) => void;
  /** Start an auto-advancing queue of the built set's track ids. */
  onPlayQueue: (ids: number[]) => void;
  /** Play a single track via the shared audio element. */
  onPlay: (id: number) => void;
}

interface SetResult {
  trackIds: number[];
  names: string[];
  arc: number[];
  reasons: string[];
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** A tiny inline energy-arc sparkline drawn as an SVG polyline. */
function ArcSparkline({ arc }: { arc: number[] }) {
  if (arc.length < 2) return null;
  const W = 100;
  const H = 28;
  const PAD = 2;
  let min = Infinity;
  let max = -Infinity;
  for (const v of arc) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min || 1;
  const stepX = (W - PAD * 2) / (arc.length - 1);
  const pts = arc
    .map((v, i) => {
      const x = PAD + i * stepX;
      const y = PAD + (1 - (v - min) / span) * (H - PAD * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      className="set-arc"
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      preserveAspectRatio="none"
      aria-label="energy arc"
    >
      <polyline
        points={pts}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function SetBuilder({ report, onPlayQueue, onPlay }: SetBuilderProps) {
  const [text, setText] = useState("");
  const [length, setLength] = useState("");
  const [building, setBuilding] = useState(false);
  const [result, setResult] = useState<SetResult | null>(null);
  const [dragIdx, setDragIdx] = useState<number | null>(null);

  // Manual edits to the built set: drag to reorder, × to remove. The arc is the
  // planned shape, so we leave it as-is when the order changes.
  const reorder = (from: number, to: number) => {
    if (from === to) return;
    setResult((r) => {
      if (!r) return r;
      const move = <T,>(arr: T[]): T[] => {
        const a = [...arr];
        const [x] = a.splice(from, 1);
        a.splice(to, 0, x);
        return a;
      };
      return { ...r, trackIds: move(r.trackIds), names: move(r.names), reasons: move(r.reasons) };
    });
  };
  const removeAt = (i: number) => {
    setResult((r) =>
      r
        ? {
            ...r,
            trackIds: r.trackIds.filter((_, j) => j !== i),
            names: r.names.filter((_, j) => j !== i),
            reasons: r.reasons.filter((_, j) => j !== i),
          }
        : r,
    );
  };

  const build = async () => {
    const desc = text.trim();
    if (!desc) {
      report("describe the set you want first", true);
      return;
    }
    setBuilding(true);
    report("building set…");
    try {
      const lenNum = length.trim() ? Number(length.trim()) : null;
      const len = lenNum !== null && Number.isFinite(lenNum) ? lenNum : null;
      const r = await api.buildSet(desc, len);
      setResult({
        trackIds: r.track_ids,
        names: r.names,
        arc: r.arc,
        reasons: r.reasons,
      });
      report(`built set · ${r.track_ids.length} tracks`);
    } catch (e) {
      report(`build set failed: ${errMsg(e)}`, true);
    } finally {
      setBuilding(false);
    }
  };

  return (
    <>
      <div className="panel-section">
        <h3>Set builder</h3>
        <div className="form">
          <div className="form__row">
            <label>Describe the set</label>
            <textarea
              className="set-input"
              rows={3}
              placeholder="light groovy house at sunset — start slow, build punchier, then slow down deep & minimal"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div className="set-controls">
            <div className="field">
              <span className="field__label">Length</span>
              <input
                type="number"
                className="set-length"
                min={1}
                placeholder="auto"
                value={length}
                onChange={(e) => setLength(e.target.value)}
              />
            </div>
            <button
              className="btn btn--go btn--xs"
              onClick={() => void build()}
              disabled={building || !text.trim()}
            >
              {building ? "Building…" : "Build set"}
            </button>
          </div>
        </div>
      </div>

      {result && (
        <div className="panel-section">
          <div className="apply-group__head">
            <span className="apply-group__title">
              Set · {result.trackIds.length} tracks
            </span>
            <button
              className="btn btn--accent btn--xs"
              onClick={() => onPlayQueue(result.trackIds)}
              disabled={result.trackIds.length === 0}
              title="Play the whole set in order, auto-advancing"
            >
              ▶ Play set
            </button>
          </div>

          {result.arc.length >= 2 && (
            <div className="set-arc-wrap" title="energy arc">
              <ArcSparkline arc={result.arc} />
            </div>
          )}

          <div className="set-hint">drag ⠿ to reorder · × to remove · then Play set</div>
          <ol className="set-list">
            {result.names.map((name, i) => {
              const id = result.trackIds[i];
              return (
                <li
                  className={dragIdx === i ? "set-item set-item--drag" : "set-item"}
                  key={`${id ?? "x"}-${i}`}
                  draggable
                  onDragStart={() => setDragIdx(i)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (dragIdx !== null) reorder(dragIdx, i);
                    setDragIdx(null);
                  }}
                  onDragEnd={() => setDragIdx(null)}
                >
                  <div className="set-item__head">
                    <span className="set-item__grip" title="Drag to reorder">⠿</span>
                    <span className="set-item__pos mono">{i + 1}</span>
                    {id !== undefined && (
                      <button
                        className="btn btn--play btn--xs"
                        onClick={() => onPlay(id)}
                        aria-label={`Play ${name}`}
                        title={`Play “${name}”`}
                      >
                        ▶
                      </button>
                    )}
                    <span className="set-item__name" title={name}>
                      {name}
                    </span>
                    <button
                      className="set-item__rm"
                      onClick={() => removeAt(i)}
                      title="Remove from set"
                      aria-label={`Remove ${name}`}
                    >
                      ×
                    </button>
                  </div>
                  {result.reasons[i] && (
                    <div className="set-item__reason">{result.reasons[i]}</div>
                  )}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </>
  );
}
