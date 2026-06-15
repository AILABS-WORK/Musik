import { useMemo, useState } from "react";
import type { Track } from "../types";
import { Meter } from "./Meter";

type SortDir = "none" | "asc" | "desc";

interface TrackTableProps {
  tracks: Track[];
  checked: Set<number>;
  selectedId: number | null;
  onToggleCheck: (id: number) => void;
  onToggleAll: (ids: number[], check: boolean) => void;
  onSelect: (id: number) => void;
  onPlay: (id: number) => void;
}

export function TrackTable({
  tracks,
  checked,
  selectedId,
  onToggleCheck,
  onToggleAll,
  onSelect,
  onPlay,
}: TrackTableProps) {
  const [filter, setFilter] = useState("");
  const [sortDir, setSortDir] = useState<SortDir>("none");

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
                  <td className="col-genre">
                    {t.genre ? (
                      <span className="genre-pill" title={t.genre}>
                        {t.genre}
                      </span>
                    ) : (
                      <span className="dash">—</span>
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
                <td colSpan={5}>
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
