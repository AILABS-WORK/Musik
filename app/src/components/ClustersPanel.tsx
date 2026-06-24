import { useCallback, useMemo, useState } from "react";
import type { Genre, Track } from "../types";
import { api } from "../api";

interface ClustersPanelProps {
  tracks: Track[];
  genres: Genre[];
  report: (msg: string, isError?: boolean) => void;
  /** Refresh tracks + genres after a mutation. */
  refresh: () => Promise<void>;
  /** Audition a track (so a cluster can be verified before bulk-labelling it). */
  onPlay: (id: number) => void;
  /** Select a track in the table (to label its parts / inspect). */
  onSelect?: (id: number) => void;
}

interface Cluster {
  id: number;
  members: number[];
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Pull a numeric track id out of a member entry (number or object). */
function memberId(raw: unknown): number | null {
  if (typeof raw === "number") return raw;
  if (typeof raw === "object" && raw !== null) {
    const r = raw as Record<string, unknown>;
    const v = r.track_id ?? r.id;
    if (typeof v === "number") return v;
  }
  return null;
}

/** Coerce one raw API cluster into a typed Cluster, or null if unusable. */
function toCluster(raw: unknown, index: number): Cluster | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const rawMembers = Array.isArray(r.members)
    ? r.members
    : Array.isArray(r.track_ids)
      ? r.track_ids
      : [];
  const members: number[] = [];
  for (const m of rawMembers) {
    const id = memberId(m);
    if (id !== null) members.push(id);
  }
  if (members.length === 0) return null;
  const idVal = r.id ?? r.cluster ?? r.label;
  const id = typeof idVal === "number" ? idVal : index;
  return { id, members };
}

export function ClustersPanel({
  tracks,
  genres,
  report,
  refresh,
  onPlay,
  onSelect,
}: ClustersPanelProps) {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [finding, setFinding] = useState(false);
  // target number of groups (KMeans); helps a homogeneous library split sensibly
  const [nGroups, setNGroups] = useState(12);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  // Per-cluster local form state keyed by cluster id.
  const [names, setNames] = useState<Record<number, string>>({});
  const [assignTo, setAssignTo] = useState<Record<number, string>>({});

  const nameById = useMemo(() => {
    const m = new Map<number, string>();
    for (const t of tracks) m.set(t.id, t.name);
    return m;
  }, [tracks]);

  const find = useCallback(async () => {
    setFinding(true);
    report("finding clusters…");
    try {
      const raw = await api.cluster(2, nGroups > 1 ? nGroups : undefined);
      const list = Array.isArray(raw) ? raw : [];
      const parsed: Cluster[] = [];
      list.forEach((c, i) => {
        const cl = toCluster(c, i);
        if (cl) parsed.push(cl);
      });
      setClusters(parsed);
      report(`found ${parsed.length} cluster(s)`);
    } catch (e) {
      report(`cluster failed: ${errMsg(e)}`, true);
    } finally {
      setFinding(false);
    }
  }, [report, nGroups]);

  const makeGenre = useCallback(
    async (cluster: Cluster) => {
      const name = (names[cluster.id] ?? "").trim();
      if (!name) {
        report("enter a name for the new genre", true);
        return;
      }
      const key = `make-${cluster.id}`;
      setBusyKey(key);
      try {
        await api.byExample({
          name,
          track_ids: cluster.members,
          level: "subgenre",
        });
        report(`created genre “${name}” from ${cluster.members.length} tracks`);
        setNames((prev) => {
          const next = { ...prev };
          delete next[cluster.id];
          return next;
        });
        await refresh();
      } catch (e) {
        report(`make genre failed: ${errMsg(e)}`, true);
      } finally {
        setBusyKey(null);
      }
    },
    [names, report, refresh],
  );

  const assign = useCallback(
    async (cluster: Cluster) => {
      const sel = assignTo[cluster.id] ?? "";
      const genreId = sel ? Number(sel) : NaN;
      if (!Number.isFinite(genreId)) {
        report("pick a genre to assign", true);
        return;
      }
      const key = `assign-${cluster.id}`;
      setBusyKey(key);
      try {
        for (const trackId of cluster.members) {
          await api.confirm({ track_id: trackId, genre_id: genreId });
        }
        const gName =
          genres.find((g) => g.id === genreId)?.name ?? `#${genreId}`;
        report(
          `assigned ${cluster.members.length} tracks to “${gName}”`,
        );
        await refresh();
      } catch (e) {
        report(`assign failed: ${errMsg(e)}`, true);
      } finally {
        setBusyKey(null);
      }
    },
    [assignTo, genres, report, refresh],
  );

  return (
    <>
      <div className="panel-section">
        <div className="apply-group__head">
          <span className="apply-group__title">Clusters</span>
          <label className="cluster-groups" title="How many groups to split the library into">
            groups
            <input
              type="number"
              min={2}
              max={60}
              value={nGroups}
              onChange={(e) => setNGroups(Math.max(2, Number(e.target.value) || 12))}
            />
          </label>
          <button
            className="btn btn--accent btn--xs"
            onClick={() => void find()}
            disabled={finding || busyKey !== null}
          >
            {finding ? "Finding…" : "Find clusters"}
          </button>
        </div>

        {clusters.length === 0 ? (
          <div className="hint">
            {finding
              ? "Clustering…"
              : "No clusters yet. Embed tracks, then find clusters."}
          </div>
        ) : (
          <div className="cluster-list">
            {clusters.map((cluster, i) => {
              const makeKey = `make-${cluster.id}`;
              const assignKey = `assign-${cluster.id}`;
              const rowBusy = busyKey !== null;
              return (
                <div className="cluster-card" key={`${cluster.id}-${i}`}>
                  <div className="cluster-card__head">
                    <span className="cluster-card__title">
                      Cluster {i + 1}
                    </span>
                    <span className="cluster-card__size">
                      {cluster.members.length} tracks
                    </span>
                  </div>

                  <div className="cluster-card__members">
                    {cluster.members.map((id) => (
                      <div className="cluster-card__member" key={id}>
                        <button
                          className="cluster-card__play"
                          onClick={() => onPlay(id)}
                          title="Audition this track"
                          aria-label="Play"
                        >
                          ▶
                        </button>
                        <span
                          className="cluster-card__mname"
                          title={nameById.get(id) ?? `#${id}`}
                          onClick={() => onSelect?.(id)}
                        >
                          {nameById.get(id) ?? `#${id}`}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="cluster-card__action">
                    <input
                      type="text"
                      placeholder="New genre name…"
                      value={names[cluster.id] ?? ""}
                      onChange={(e) =>
                        setNames((prev) => ({
                          ...prev,
                          [cluster.id]: e.target.value,
                        }))
                      }
                    />
                    <button
                      className="btn btn--go btn--xs"
                      onClick={() => void makeGenre(cluster)}
                      disabled={rowBusy}
                    >
                      {busyKey === makeKey ? "…" : "Make genre"}
                    </button>
                  </div>

                  <div className="cluster-card__action">
                    <select
                      value={assignTo[cluster.id] ?? ""}
                      onChange={(e) =>
                        setAssignTo((prev) => ({
                          ...prev,
                          [cluster.id]: e.target.value,
                        }))
                      }
                    >
                      <option value="">— assign to genre —</option>
                      {genres.map((g) => (
                        <option key={g.id} value={String(g.id)}>
                          {g.name}
                        </option>
                      ))}
                    </select>
                    <button
                      className="btn btn--xs"
                      onClick={() => void assign(cluster)}
                      disabled={rowBusy}
                    >
                      {busyKey === assignKey ? "…" : "Assign"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
