import { useState } from "react";
import type { Track } from "../types";
import { api } from "../api";

interface PlaylistPanelProps {
  tracks: Track[];
  checked: Set<number>;
  onPlay: (id: number) => void;
  onPlayQueue: (ids: number[]) => void;
  report: (msg: string, isError?: boolean) => void;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * A hand-built playlist: drag tracks in from the table (or "+ checked"), reorder by drag,
 * then "Make folder" to copy/move the files into a numbered folder on disk (the XDJ then
 * sees them in order).
 */
export function PlaylistPanel({ tracks, checked, onPlay, onPlayQueue, report }: PlaylistPanelProps) {
  const [items, setItems] = useState<number[]>([]);
  const [name, setName] = useState("");
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);

  const nameById = (id: number) => tracks.find((t) => t.id === id)?.name ?? `#${id}`;
  const add = (ids: number[]) =>
    setItems((prev) => [...prev, ...ids.filter((id) => !prev.includes(id))]);
  const addChecked = () => {
    if (checked.size === 0) {
      report("check some tracks in the table first", true);
      return;
    }
    add(Array.from(checked));
  };
  const removeAt = (i: number) => setItems((prev) => prev.filter((_, j) => j !== i));
  const reorder = (from: number, to: number) =>
    setItems((prev) => {
      const a = [...prev];
      const [x] = a.splice(from, 1);
      a.splice(to, 0, x);
      return a;
    });

  const makeFolder = async (mode: "copy" | "move") => {
    if (items.length === 0) {
      report("playlist is empty", true);
      return;
    }
    setBusy(true);
    try {
      const r = await api.playlistFolder(items, name.trim() || "Playlist", mode);
      report(`${mode === "move" ? "moved" : "copied"} ${r.copied} tracks → ${r.folder}`);
    } catch (e) {
      report(`make folder failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel-section">
      <div className="apply-group__head">
        <span className="apply-group__title">Playlist · {items.length}</span>
        <button className="btn btn--xs" onClick={addChecked} title="Add the tracks checked in the table">
          + checked ({checked.size})
        </button>
        {items.length > 0 && (
          <button className="btn btn--accent btn--xs" onClick={() => onPlayQueue(items)} title="Play the playlist in order">
            ▶ Play
          </button>
        )}
      </div>

      <div
        className={over ? "pl-drop pl-drop--over" : "pl-drop"}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const id = Number(e.dataTransfer.getData("text/track-id"));
          if (id) add([id]);
        }}
      >
        drag tracks here, or use “+ checked”
      </div>

      {items.length > 0 && (
        <>
          <ol className="pl-list">
            {items.map((id, i) => (
              <li
                className={dragIdx === i ? "pl-item pl-item--drag" : "pl-item"}
                key={id}
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
                <span className="pl-item__grip" title="Drag to reorder">⠿</span>
                <span className="pl-item__pos mono">{i + 1}</span>
                <button className="btn btn--play btn--xs" onClick={() => onPlay(id)} aria-label="Play">▶</button>
                <span className="pl-item__name" title={nameById(id)}>{nameById(id)}</span>
                <button className="pl-item__rm" onClick={() => removeAt(i)} title="Remove" aria-label="Remove">×</button>
              </li>
            ))}
          </ol>
          <div className="pl-make">
            <input
              className="song-label__input"
              placeholder="folder name…"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <button className="btn btn--accent btn--xs" disabled={busy} onClick={() => void makeFolder("copy")}>
              {busy ? "…" : "Make folder"}
            </button>
            <button
              className="btn btn--xs"
              disabled={busy}
              onClick={() => void makeFolder("move")}
              title="Move the files (removes them from their source location)"
            >
              move
            </button>
          </div>
        </>
      )}
    </div>
  );
}
