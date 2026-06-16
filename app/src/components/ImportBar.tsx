import { useEffect, useState } from "react";
import { api } from "../api";

interface ImportBarProps {
  onImport: (paths: string[]) => void;
  report: (msg: string, isError?: boolean) => void;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Drag-and-drop import (native paths under Tauri) + an always-on
 *  add-by-path fallback, plus the Organized-library destination picker. */
export function ImportBar({ onImport, report }: ImportBarProps) {
  const [pathInput, setPathInput] = useState("");
  const [dragging, setDragging] = useState(false);

  const [dest, setDest] = useState("");
  const [mode, setMode] = useState("copy");
  const [loaded, setLoaded] = useState(false);

  // load current destination from the sidecar
  useEffect(() => {
    let alive = true;
    api
      .getConfig()
      .then((c) => {
        if (!alive) return;
        setDest(c.organize_root ?? "");
        setMode(c.organize_mode ?? "copy");
        setLoaded(true);
      })
      .catch(() => alive && setLoaded(true));
    return () => {
      alive = false;
    };
  }, []);

  // native OS file drag-drop (Tauri only — browsers can't expose absolute paths)
  useEffect(() => {
    if (!isTauri()) return;
    let active = true;
    let unlisten: (() => void) | undefined;
    (async () => {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        const un = await getCurrentWebview().onDragDropEvent((event) => {
          const payload = event.payload as { type: string; paths?: string[] };
          if (payload.type === "enter" || payload.type === "over") {
            setDragging(true);
          } else if (payload.type === "leave") {
            setDragging(false);
          } else if (payload.type === "drop") {
            setDragging(false);
            const paths = payload.paths ?? [];
            if (paths.length) onImport(paths);
          }
        });
        if (active) unlisten = un;
        else un();
      } catch {
        /* not running under Tauri — the path-input fallback covers it */
      }
    })();
    return () => {
      active = false;
      if (unlisten) unlisten();
    };
  }, [onImport]);

  const addByPath = () => {
    const p = pathInput.trim().replace(/^["']|["']$/g, "");
    if (!p) return;
    onImport([p]);
    setPathInput("");
  };

  const saveDest = async () => {
    try {
      await api.setConfig({ organize_root: dest.trim() || null, organize_mode: mode });
      report(`organized library → ${dest.trim() || "(default)"} · ${mode}`);
    } catch (e) {
      report(`set destination failed: ${errMsg(e)}`, true);
    }
  };

  return (
    <div className="importbar">
      <div className="importbar__row">
        <span className="importbar__label">Import</span>
        <input
          className="importbar__input"
          placeholder={
            isTauri()
              ? "Drop music files/folders anywhere — or paste a path…"
              : "Paste a file or folder path, then Add…"
          }
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") addByPath();
          }}
        />
        <button className="btn btn--go btn--xs" onClick={addByPath} disabled={!pathInput.trim()}>
          Add
        </button>
      </div>

      <div className="importbar__row">
        <span className="importbar__label">Organized&nbsp;library</span>
        <input
          className="importbar__input"
          placeholder="destination folder for organized music…"
          value={dest}
          disabled={!loaded}
          onChange={(e) => setDest(e.target.value)}
        />
        <select className="importbar__select" value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="copy">copy</option>
          <option value="move">move</option>
        </select>
        <button className="btn btn--xs" onClick={saveDest}>
          Set
        </button>
      </div>

      {dragging && (
        <div className="dropoverlay">
          <div className="dropoverlay__inner">Drop music to import</div>
        </div>
      )}
    </div>
  );
}
