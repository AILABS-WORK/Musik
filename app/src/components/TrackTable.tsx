import { useMemo, useState } from "react";
import type { Genre, Track } from "../types";
import { Meter } from "./Meter";

type SortDir = "none" | "asc" | "desc";

interface TrackTableProps {
  tracks: Track[];
  genres: Genre[];
  checked: Set<number>;
  selectedId: number | null;
  onToggleCheck: (id: number) => void;
  onToggleAll: (ids: number[], check: boolean) => void;
  onSelect: (id: number) => void;
  onPlay: (id: number) => void;
  /** Confirm a genre assignment for a track, then refresh + report. */
  onConfirm: (trackId: number, genreId: number) => void;
}

export function TrackTable({
  tracks,
  genres,
  checked,
  selectedId,
  onToggleCheck,
  onToggleAll,
  onSelect,
  onPlay,
  onConfirm,
}: TrackTableProps) {
  const [filter, setFilter] = useState("");
  const [sortDir, setSortDir] = useState<SortDir>("none");
  const [editing, setEditing] = useState<number | null>(null);

  const genreByName = useMemo(() => {
    const m = new Map<string, number>();
    for (const g of genres) m.set(g.name, g.id);
    return m;
  }, [genres]);

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    let rows = q
      ? tracks.filter((t) => t.name.toLowerCase().includes(q))
      : tracks.slice();
    if (sortDir !== "none") {
      rows = rows.slice().sort((a, b) => {
        const av = a.confidence ?? -1;
        const bv = b.confidence ?? -1;
        return sortDir === "asc" ? av - bv : bv - av;
      });
    }
    return rows;
  }, [tracks, filter, sortDir]);

  const visibleIds = useMemo(() => visible.map((t) => t.id), [visible]);
  const allChecked =
    visibleIds.length > 0 && visibleIds.every((id) => checked.has(id));

  const cycleSort = () =>
    setSortDir((d) => (d === "desc" ? "asc" : d === "asc" ? "none" : "desc"));

  const sortLabel =
    sortDir === "desc"
      ? "conf ↓"
      : sortDir === "asc"
        ? "conf ↑"
        : "sort by confidence";

  return (
    <>
      <div className="toolbar">
        <input
          type="text"
          placeholder="Filter by name…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ minWidth: 200 }}
        />
        <button className="btn" onClick={cycleSort}>
          {sortLabel}
        </button>
        <div className="spacer" />
        <span className="toolbar__counts">
          <strong>{checked.size}</strong> checked ·{" "}
          <strong>{visible.length}</strong> shown · {tracks.length} total
          {selectedId !== null ? " · 1 selected" : ""}
        </span>
      </div>

      <div className="table-wrap">
        <table className="tracks">
          <thead>
            <tr>
              <th className="col-check">
                <input
                  type="checkbox"
                  checked={allChecked}
                  onChange={(e) => onToggleAll(visibleIds, e.target.checked)}
                  aria-label="Select all visible"
                />
              </th>
              <th className="col-name">Name</th>
              <th className="col-bpm">BPM</th>
              <th className="col-key">Key</th>
              <th className="col-energy">Energy</th>
              <th className="col-genre">Genre</th>
              <th className="col-conf sortable" onClick={cycleSort}>
                Confidence
              </th>
              <th className="col-play">Play</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((t) => {
              const isSel = t.id === selectedId;
              return (
                <tr
                  key={t.id}
                  className={isSel ? "selected" : undefined}
                  onClick={() => onSelect(t.id)}
                >
                  <td
                    className="col-check"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      checked={checked.has(t.id)}
                      onChange={() => onToggleCheck(t.id)}
                      aria-label={`Select ${t.name}`}
                    />
                  </td>
                  <td className="col-name">
                    <div className="cell-name" title={t.path}>
                      {t.name}
                    </div>
                  </td>
                  <td className="col-bpm mono">
                    {t.bpm !== null ? (
                      Math.round(t.bpm)
                    ) : (
                      <span className="dash">—</span>
                    )}
                  </td>
                  <td className="col-key mono">
                    {t.music_key ? (
                      t.music_key
                    ) : (
                      <span className="dash">—</span>
                    )}
                  </td>
                  <td className="col-energy">
                    {t.energy !== null ? (
                      <div
                        className="energy-bar"
                        title={`energy ${t.energy.toFixed(2)}`}
                      >
                        <div
                          className="energy-bar__fill"
                          style={{
                            width: `${Math.max(0, Math.min(1, t.energy)) * 100}%`,
                          }}
                        />
                      </div>
                    ) : (
                      <span className="dash">—</span>
                    )}
                  </td>
                  <td
                    className="col-genre"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {editing === t.id ? (
                      <select
                        className="genre-edit"
                        autoFocus
                        defaultValue={
                          t.genre && genreByName.has(t.genre)
                            ? String(genreByName.get(t.genre))
                            : ""
                        }
                        onBlur={() => setEditing(null)}
                        onChange={(e) => {
                          const v = e.target.value;
                          setEditing(null);
                          if (v) onConfirm(t.id, Number(v));
                        }}
                      >
                        <option value="">— pick genre —</option>
                        {genres.map((g) => (
                          <option key={g.id} value={String(g.id)}>
                            {g.name}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <button
                        type="button"
                        className="genre-cell"
                        onClick={() => setEditing(t.id)}
                        title={t.genre ?? "click to assign a genre"}
                      >
                        {t.genre ? (
                          <span className="genre-pill">{t.genre}</span>
                        ) : (
                          <span className="dash">—</span>
                        )}
                        {t.assignment_status === "confirmed" && (
                          <span className="genre-check" title="confirmed">
                            ✓
                          </span>
                        )}
                      </button>
                    )}
                  </td>
                  <td className="col-conf">
                    <Meter value={t.confidence} />
                  </td>
                  <td
                    className="col-play"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      className="btn btn--play btn--xs"
                      onClick={() => onPlay(t.id)}
                      aria-label={`Play ${t.name}`}
                    >
                      ▶
                    </button>
                  </td>
                </tr>
              );
            })}
            {visible.length === 0 && (
              <tr>
                <td colSpan={8}>
                  <div className="hint">
                    {tracks.length === 0
                      ? "No tracks yet. Set a library path and hit Scan."
                      : "No tracks match the filter."}
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
