import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Track } from "../types";
import { api } from "../api";

interface MapViewProps {
  tracks: Track[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

type ProjMethod = "pca" | "umap";

interface ProjPoint {
  track_id: number;
  x: number;
  y: number;
  genre: string | null;
  major: string | null;
}

interface PlacedPoint extends ProjPoint {
  px: number;
  py: number;
  color: string;
  name: string;
}

interface HoverState {
  point: PlacedPoint;
  cx: number;
  cy: number;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const NULL_COLOR = "#5c6573";

/** Fixed, evenly-spread palette; stable per genre name via a string hash. */
const PALETTE = [
  "#2dd4bf",
  "#4ade80",
  "#f59e0b",
  "#60a5fa",
  "#c084fc",
  "#f472b6",
  "#fb7185",
  "#facc15",
  "#34d399",
  "#a3e635",
  "#38bdf8",
  "#e879f9",
];

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function colorForGenre(genre: string | null): string {
  if (genre === null || genre === "") return NULL_COLOR;
  return PALETTE[hashString(genre) % PALETTE.length] ?? NULL_COLOR;
}

/** Coerce one raw API point into a typed ProjPoint, or null if unusable. */
function toProjPoint(raw: unknown): ProjPoint | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const id = r.track_id;
  const x = r.x;
  const y = r.y;
  if (typeof id !== "number" || typeof x !== "number" || typeof y !== "number") {
    return null;
  }
  const genre = typeof r.genre === "string" ? r.genre : null;
  const major = typeof r.major === "string" ? r.major : null;
  return { track_id: id, x, y, genre, major };
}

const PAD = 28;

export function MapView({ tracks, selectedId, onSelect }: MapViewProps) {
  const [method, setMethod] = useState<ProjMethod>("pca");
  const [colorBy, setColorBy] = useState<"major" | "genre">("major");
  const [points, setPoints] = useState<ProjPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 600, h: 400 });
  const [hover, setHover] = useState<HoverState | null>(null);

  const wrapRef = useRef<HTMLDivElement | null>(null);

  const nameById = useMemo(() => {
    const m = new Map<number, string>();
    for (const t of tracks) m.set(t.id, t.name);
    return m;
  }, [tracks]);

  const recompute = useCallback(
    async (m: ProjMethod) => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.project(m);
        const raw = Array.isArray(res.points) ? res.points : [];
        const parsed: ProjPoint[] = [];
        for (const p of raw) {
          const pt = toProjPoint(p);
          if (pt) parsed.push(pt);
        }
        setPoints(parsed);
        setLoaded(true);
      } catch (e) {
        setError(errMsg(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Initial projection on mount.
  useEffect(() => {
    void recompute("pca");
  }, [recompute]);

  // Track container size so the SVG fills the viewport.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const placed = useMemo<PlacedPoint[]>(() => {
    if (points.length === 0) return [];
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const innerW = Math.max(1, size.w - PAD * 2);
    const innerH = Math.max(1, size.h - PAD * 2);
    return points.map((p) => ({
      ...p,
      px: PAD + ((p.x - minX) / spanX) * innerW,
      py: PAD + (1 - (p.y - minY) / spanY) * innerH,
      color: colorForGenre(colorBy === "major" ? p.major : p.genre),
      name: nameById.get(p.track_id) ?? `#${p.track_id}`,
    }));
  }, [points, size, nameById, colorBy]);

  // Legend: unique color-key (major or subgenre) in encounter order, + "unknown".
  const legend = useMemo(() => {
    const seen = new Map<string, string>();
    let hasNull = false;
    for (const p of points) {
      const key = colorBy === "major" ? p.major : p.genre;
      if (key === null || key === "") {
        hasNull = true;
      } else if (!seen.has(key)) {
        seen.set(key, colorForGenre(key));
      }
    }
    const out = Array.from(seen, ([name, color]) => ({ name, color }));
    if (hasNull) out.push({ name: "unknown", color: NULL_COLOR });
    return out;
  }, [points, colorBy]);

  return (
    <>
      <div className="toolbar">
        <span className="map__heading">Similarity map</span>
        <div className="seg">
          <button
            className={method === "pca" ? "seg__btn active" : "seg__btn"}
            onClick={() => {
              setMethod("pca");
              void recompute("pca");
            }}
            disabled={loading}
          >
            PCA
          </button>
          <button
            className={method === "umap" ? "seg__btn active" : "seg__btn"}
            onClick={() => {
              setMethod("umap");
              void recompute("umap");
            }}
            disabled={loading}
          >
            UMAP
          </button>
        </div>
        <button
          className="btn btn--accent btn--xs"
          onClick={() => void recompute(method)}
          disabled={loading}
        >
          {loading ? "Projecting…" : "Recompute"}
        </button>
        <div className="seg">
          <button
            className={colorBy === "major" ? "seg__btn active" : "seg__btn"}
            onClick={() => setColorBy("major")}
            title="Colour dots by major genre"
          >
            by major
          </button>
          <button
            className={colorBy === "genre" ? "seg__btn active" : "seg__btn"}
            onClick={() => setColorBy("genre")}
            title="Colour dots by subgenre"
          >
            by sub
          </button>
        </div>
        <div className="spacer" />
        <span className="toolbar__counts">
          <strong>{placed.length}</strong> points
        </span>
      </div>

      <div className="map-wrap" ref={wrapRef}>
        {error ? (
          <div className="hint">projection failed: {error}</div>
        ) : !loaded && loading ? (
          <div className="hint">Projecting…</div>
        ) : placed.length <= 2 ? (
          <div className="hint">
            Not enough points to map yet. Embed your tracks and suggest a few
            genres first, then recompute.
          </div>
        ) : (
          <svg
            className="map-svg"
            width={size.w}
            height={size.h}
            onMouseLeave={() => setHover(null)}
          >
            {placed.map((p) => {
              const isSel = p.track_id === selectedId;
              return (
                <circle
                  key={p.track_id}
                  cx={p.px}
                  cy={p.py}
                  r={isSel ? 6.5 : 4}
                  fill={p.color}
                  stroke={isSel ? "#d7dce3" : "rgba(13,15,18,0.7)"}
                  strokeWidth={isSel ? 2 : 1}
                  className="map-pt"
                  onMouseEnter={() =>
                    setHover({ point: p, cx: p.px, cy: p.py })
                  }
                  onClick={() => onSelect(p.track_id)}
                />
              );
            })}
          </svg>
        )}

        {hover && (
          <div
            className="map-tooltip"
            style={{
              left: Math.min(hover.cx + 12, size.w - 8),
              top: Math.max(hover.cy - 8, 4),
            }}
          >
            <span className="map-tooltip__name">{hover.point.name}</span>
            <span className="map-tooltip__genre">
              {hover.point.genre ?? "unknown"}
            </span>
          </div>
        )}

        {legend.length > 0 && (
          <div className="map-legend">
            {legend.map((g) => (
              <div className="map-legend__item" key={g.name} title={g.name}>
                <span
                  className="map-legend__swatch"
                  style={{ background: g.color }}
                />
                <span className="map-legend__label">{g.name}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
