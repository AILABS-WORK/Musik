interface SelectionBarProps {
  count: number;
  onBuildSet: () => void;
  onUseAsExamples: () => void;
  onClear: () => void;
}

/**
 * Action bar that slides in from the bottom whenever ≥1 track is checked.
 * Wires the checked Set (owned by App) to the existing Set / Genres tabs.
 */
export function SelectionBar({
  count,
  onBuildSet,
  onUseAsExamples,
  onClear,
}: SelectionBarProps) {
  if (count < 1) return null;
  return (
    <div className="selbar" role="region" aria-label="Selection actions">
      <span className="selbar__count">
        <strong className="mono">{count}</strong> selected
      </span>
      <div className="selbar__actions">
        <button
          className="btn btn--accent btn--sm"
          onClick={onUseAsExamples}
          title="Open Genres and create a sub-genre from these tracks"
        >
          Use as genre examples
        </button>
        <button
          className="btn btn--go btn--sm"
          onClick={onBuildSet}
          title="Open the Set tab to build a DJ set"
        >
          Build set from these
        </button>
        <button
          className="btn btn--sm selbar__clear"
          onClick={onClear}
          title="Clear selection"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
