import { useEffect, useRef, useState } from "react";

export interface SearchMeta {
  method: string;
  matched_label?: string | null;
  note?: string;
}

interface SearchBarProps {
  /** Run a search. threshold is null when the slider is at 0 (off). */
  onSearch: (query: string, threshold: number | null) => void;
  /** Clear the active search and restore the full track list. */
  onClear: () => void;
  /** Meta from the most recent search; null when no search is active. */
  meta: SearchMeta | null;
  /** Whether a search is currently active (results are being shown). */
  active: boolean;
  /** Number of results currently shown (for the meta line). */
  count: number;
}

const EXAMPLES = [
  "songs with cowbells",
  "female vocals",
  "electric guitar",
  "dark melodic techno",
  "lo-fi piano",
  "punchy drums",
];

export function SearchBar({ onSearch, onClear, meta, active, count }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0);
  const [exampleIdx, setExampleIdx] = useState(0);

  // Keep the latest query readable from the rotating-placeholder timer
  // without restarting the interval on every keystroke.
  const queryRef = useRef(query);
  queryRef.current = query;

  // Cycle the placeholder examples every ~3s while the box is empty.
  useEffect(() => {
    const t = window.setInterval(() => {
      if (queryRef.current === "") {
        setExampleIdx((i) => (i + 1) % EXAMPLES.length);
      }
    }, 3000);
    return () => window.clearInterval(t);
  }, []);

  const runSearch = () => {
    onSearch(query, threshold > 0 ? threshold : null);
  };

  const clear = () => {
    setQuery("");
    onClear();
  };

  const chipLabel = meta
    ? meta.matched_label
      ? `AudioSet · ${meta.matched_label}`
      : "open-vocab (CLAP)"
    : "";

  return (
    <div className="searchbar">
      <div className="searchbar__title">
        Find tracks by sound
        <span className="searchbar__title-hint">
          describe a vibe in plain words — no tags needed
        </span>
      </div>
      <div className="searchbar__row">
        <span className="searchbar__icon" aria-hidden="true" title="Sound search">
          ⌕
        </span>
        <input
          className="searchbar__input"
          type="text"
          value={query}
          placeholder={`Search by sound — e.g. "${EXAMPLES[exampleIdx]}"`}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") runSearch();
          }}
          aria-label="Open-vocabulary sound search"
        />

        <label className="searchbar__thresh" title="minimum match score">
          <span className="searchbar__thresh-label">min score</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <span className="searchbar__thresh-val mono">
            {threshold > 0 ? threshold.toFixed(2) : "off"}
          </span>
        </label>

        <button
          className="btn btn--go"
          onClick={runSearch}
          title="Find tracks that match what you described"
        >
          Search
        </button>
        <button
          className="btn"
          onClick={clear}
          disabled={!active && query === ""}
          title="Clear the search and show all tracks"
        >
          Clear
        </button>
      </div>

      {active && meta && (
        <div className="searchbar__meta">
          <span className="searchbar__chip">{chipLabel}</span>
          <span className="searchbar__count mono">
            {count} result{count === 1 ? "" : "s"}
          </span>
          {meta.note && <span className="searchbar__note">{meta.note}</span>}
        </div>
      )}
    </div>
  );
}
