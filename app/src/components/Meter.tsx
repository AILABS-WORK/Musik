interface MeterProps {
  /** 0..1 value; null renders an empty placeholder. */
  value: number | null;
  /** Color of the fill. Defaults to a confidence gradient based on value. */
  color?: string;
  /** Decimal places for the printed number. */
  digits?: number;
}

/** Pick a color along amber -> teal -> green by magnitude. */
function autoColor(v: number): string {
  if (v >= 0.66) return "#4ade80";
  if (v >= 0.33) return "#2dd4bf";
  return "#f59e0b";
}

/** A small horizontal bar + the numeric value (monospace, 2dp by default). */
export function Meter({ value, color, digits = 2 }: MeterProps) {
  if (value === null || Number.isNaN(value)) {
    return (
      <div className="meter">
        <div className="meter__track" />
        <span className="meter__num dash">—</span>
      </div>
    );
  }
  const clamped = Math.max(0, Math.min(1, value));
  const fill = color ?? autoColor(clamped);
  return (
    <div className="meter">
      <div className="meter__track">
        <div
          className="meter__fill"
          style={{ width: `${clamped * 100}%`, background: fill }}
        />
      </div>
      <span className="meter__num">{value.toFixed(digits)}</span>
    </div>
  );
}
