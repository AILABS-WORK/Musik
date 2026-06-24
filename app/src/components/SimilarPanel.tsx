import { useEffect, useMemo, useState } from "react";
import type { Genre, Track } from "../types";
import { api } from "../api";

interface SimilarPanelProps {
  selected: Track | null;
  genres: Genre[];
  onPlay: (id: number) => void;
  onRadio: (id: number) => void;
  report: (msg: string, isError?: boolean) => void;
  /** Refresh tracks + genres after a bulk assign. */
  onChanged: () => void;
}

interface BandMatch {
  name: string;
  a: number;
  b: number;
  match: number;
}

interface Match {
  track_id: number;
  name: string;
  score: number;
  bands: BandMatch[];
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Six tiny per-range bars (sub..highs); green = the two tracks match there. */
function BandStrip({ bands }: { bands: BandMatch[] }) {
  if (bands.length === 0) return null;
  return (
    <span className="simrow__bands" title="per-range match: sub · bass · low-mid · mid · high-mid · highs">
      {bands.map((b) => (
        <span
          key={b.name}
          className="simrow__band"
          title={`${b.name}: ${Math.round(b.match * 100)}%`}
          style={{ background: `hsl(${Math.round(b.match * 130)},70%,45%)`, opacity: 0.35 + b.match * 0.65 }}
        />
      ))}
    </span>
  );
}

export function SimilarPanel({ selected, genres, onPlay, onRadio, report, onChanged }: SimilarPanelProps) {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(false);
  const [checked, setChecked] = useState<Set<number>>(new Set());
  // genre picker (major -> subgenre, with create-new)
  const [majorSel, setMajorSel] = useState("");
  const [subSel, setSubSel] = useState("");
  const [newName, setNewName] = useState("");
  const [assigning, setAssigning] = useState(false);

  useEffect(() => {
    setMatches([]);
    setChecked(new Set());
    if (!selected) return;
    let alive = true;
    setLoading(true);
    const id = selected.id;
    void (async () => {
      try {
        const r = await api.similarDetailed(id, 30);
        if (alive) setMatches(r.matches);
      } catch (e) {
        if (alive) report(`similar failed: ${errMsg(e)}`, true);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [selected, report]);

  const majors = useMemo(() => genres.filter((g) => g.level === "genre"), [genres]);
  const subsForMajor = useMemo(
    () =>
      majorSel && majorSel !== "__new__"
        ? genres.filter((g) => g.level === "subgenre" && g.parent_id === Number(majorSel))
        : genres.filter((g) => g.level === "subgenre"),
    [genres, majorSel],
  );

  const toggle = (id: number) =>
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });

  const allChecked = matches.length > 0 && checked.size === matches.length;
  const toggleAll = () =>
    setChecked(allChecked ? new Set() : new Set(matches.map((m) => m.track_id)));

  // Assign the checked tracks (plus the selected seed) to a subgenre under a major.
  const assign = async () => {
    if (!selected) return;
    const ids = Array.from(checked);
    if (ids.length === 0) {
      report("tick some similar tracks first", true);
      return;
    }
    setAssigning(true);
    try {
      const parentId: number | null = majorSel ? Number(majorSel) : null;
      if (subSel && subSel !== "__new__") {
        for (const tid of ids) await api.confirm({ track_id: tid, genre_id: Number(subSel) });
        report(`assigned ${ids.length} track(s)`);
      } else {
        const name = newName.trim();
        if (!name) { report("name the subgenre", true); return; }
        await api.byExample({ name, track_ids: ids, parent_id: parentId, level: "subgenre" });
        report(`created “${name}” from ${ids.length} track(s)`);
      }
      setChecked(new Set());
      setNewName("");
      setSubSel("");
      onChanged();
    } catch (e) {
      report(`assign failed: ${errMsg(e)}`, true);
    } finally {
      setAssigning(false);
    }
  };

  if (!selected) {
    return <div className="hint">Select a track in the table to see what sounds like it.</div>;
  }

  return (
    <div className="panel-section">
      <div className="apply-group__head">
        <span className="apply-group__title">Sounds like “{selected.name}”</span>
        <button className="btn btn--accent btn--xs" onClick={() => onRadio(selected.id)} title="Auto-advancing radio from this track">
          📻 Radio
        </button>
      </div>

      {/* bulk assign bar */}
      <div className="sim-assign">
        <label className="sim-assign__all">
          <input type="checkbox" checked={allChecked} onChange={toggleAll} />
          {checked.size > 0 ? `${checked.size} selected` : "select all"}
        </label>
        <select className="song-pick__sel" value={majorSel} onChange={(e) => { setMajorSel(e.target.value); setSubSel(""); }}>
          <option value="">major…</option>
          {majors.map((m) => <option key={m.id} value={String(m.id)}>{m.name}</option>)}
        </select>
        <select className="song-pick__sel" value={subSel} onChange={(e) => setSubSel(e.target.value)}>
          <option value="">subgenre…</option>
          {subsForMajor.map((s) => <option key={s.id} value={String(s.id)}>{s.name}</option>)}
          <option value="__new__">➕ new…</option>
        </select>
        <button className="btn btn--xs btn--accent" disabled={assigning || checked.size === 0} onClick={() => void assign()}>
          {assigning ? "…" : `Assign ${checked.size || ""}`}
        </button>
      </div>
      {subSel === "__new__" && (
        <input
          className="song-label__input"
          placeholder="new subgenre name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") void assign(); }}
        />
      )}

      {loading ? (
        <div className="hint">listening…</div>
      ) : matches.length === 0 ? (
        <div className="hint">No matches. Embed + index frequencies first.</div>
      ) : (
        <div className="sim-rows">
          {matches.map((m) => {
            const pct = Math.round(Math.max(0, Math.min(1, m.score)) * 100);
            const on = checked.has(m.track_id);
            return (
              <div className={on ? "simrow simrow--on" : "simrow"} key={m.track_id}>
                <input type="checkbox" checked={on} onChange={() => toggle(m.track_id)} />
                <button className="simrow__play" onClick={() => onPlay(m.track_id)} title="Play" aria-label="Play">▶</button>
                <span className="simrow__name" title={m.name} onClick={() => toggle(m.track_id)}>{m.name}</span>
                <BandStrip bands={m.bands} />
                <span className="simrow__score mono" title="overall match">{pct}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
