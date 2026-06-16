import { useEffect, useRef, useState } from "react";
import { api } from "../api";

interface AddMusicProps {
  /** Import absolute file/folder paths (folders, bulk). */
  onImport: (paths: string[]) => void;
  /** Upload browser File objects (drag-drop / Browse picker). */
  onUploadFiles: (files: File[]) => void;
  report: (msg: string, isError?: boolean) => void;
  /** Render compact (used inside the empty-state hero). */
  compact?: boolean;
}

const AUDIO_ACCEPT = "audio/*,.flac,.wav,.mp3,.aiff,.m4a,.ogg";

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * "Add music" area — the primary loading surface.
 *  • Browse files  → hidden <input type=file multiple> → upload path
 *  • a drop-zone hint (the real drop target is the full-window overlay in App)
 *  • a collapsible "paste a folder path" row for bulk folder import
 *  • the Organized-library destination picker (preserved from the old ImportBar)
 */
export function AddMusic({ onImport, onUploadFiles, report, compact }: AddMusicProps) {
  const fileRef = useRef<HTMLInputElement | null>(null);

  const [pathInput, setPathInput] = useState("");
  const [showPath, setShowPath] = useState(false);

  const [dest, setDest] = useState("");
  const [mode, setMode] = useState("copy");
  const [loaded, setLoaded] = useState(false);
  const [showDest, setShowDest] = useState(false);

  // load current organize destination from the sidecar
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

  const pickFiles = () => fileRef.current?.click();

  const onFilesChosen = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    onUploadFiles(Array.from(list));
    if (fileRef.current) fileRef.current.value = "";
  };

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
    <div className={compact ? "addmusic addmusic--compact" : "addmusic"}>
      <input
        ref={fileRef}
        type="file"
        multiple
        accept={AUDIO_ACCEPT}
        hidden
        onChange={(e) => onFilesChosen(e.target.files)}
      />

      {!compact && <div className="addmusic__title">Add music</div>}

      <div className="addmusic__main">
        <button
          className="btn btn--accent addmusic__browse"
          onClick={pickFiles}
          title="Choose audio files from your computer"
        >
          <span className="addmusic__browse-icon" aria-hidden="true">⤓</span>
          Browse files
        </button>
        <div className="addmusic__hint">
          <span className="addmusic__hint-strong">…or drop audio anywhere</span>
          <span className="addmusic__hint-dim">
            {isTauri() ? "files or folders" : "wav · mp3 · flac · aiff · m4a · ogg"}
          </span>
        </div>
      </div>

      {!compact && (
        <div className="addmusic__rows">
          <button
            type="button"
            className="addmusic__toggle"
            onClick={() => setShowPath((v) => !v)}
            aria-expanded={showPath}
          >
            <span className="addmusic__chev">{showPath ? "▾" : "▸"}</span>
            or paste a folder path
          </button>
          {showPath && (
            <div className="addmusic__row">
              <input
                className="addmusic__input"
                placeholder={
                  isTauri()
                    ? "/path/to/folder — drop also works"
                    : "/path/to/folder or single file…"
                }
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") addByPath();
                }}
              />
              <button
                className="btn btn--go btn--xs"
                onClick={addByPath}
                disabled={!pathInput.trim()}
              >
                Add
              </button>
            </div>
          )}

          <button
            type="button"
            className="addmusic__toggle"
            onClick={() => setShowDest((v) => !v)}
            aria-expanded={showDest}
          >
            <span className="addmusic__chev">{showDest ? "▾" : "▸"}</span>
            organized-library destination
          </button>
          {showDest && (
            <div className="addmusic__row">
              <input
                className="addmusic__input"
                placeholder="destination folder for organized music…"
                value={dest}
                disabled={!loaded}
                onChange={(e) => setDest(e.target.value)}
              />
              <select
                className="addmusic__select"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="copy">copy</option>
                <option value="move">move</option>
              </select>
              <button className="btn btn--xs" onClick={saveDest}>
                Set
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
