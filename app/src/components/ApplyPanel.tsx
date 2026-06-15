import { useState } from "react";
import { api } from "../api";

interface ApplyPanelProps {
  report: (msg: string, isError?: boolean) => void;
  /** Called after a real (applied) write/organize/undo so the app can refresh. */
  onMutated: () => void;
}

interface TagResult {
  count: number;
  applied: boolean;
  rows: string[];
}

interface OrgResult {
  count: number;
  applied: boolean;
  rows: string[];
}

const MAX_ROWS = 15;

function asStr(v: unknown): string {
  return v === null || v === undefined ? "" : String(v);
}

/** Last path segment, handling both / and \ separators. */
function baseName(p: string): string {
  const s = p.replace(/\\/g, "/").replace(/\/+$/, "");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

/** Last N path segments, for compact destinations. */
function tail(p: string, n: number): string {
  const parts = p.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.slice(-n).join("/");
}

/** A tag plan row -> "name: from → to". The API sends {path, from, to}. */
function tagRow(p: Record<string, unknown>): string {
  const path = asStr(p.path ?? p.name ?? p.track ?? p.title);
  const name = path ? baseName(path) : asStr(p.track_id ?? p.id);
  const from = asStr(p.from ?? p.old ?? p.current ?? "");
  const to = asStr(p.to ?? p.new ?? p.genre ?? p.value ?? "");
  return `${name}: ${from || "∅"} → ${to || "∅"}`;
}

/** An organize plan row -> "name → Genre/Subgenre/file". API sends {src, dest}. */
function orgRow(p: Record<string, unknown>): string {
  const src = asStr(p.src ?? p.from ?? p.source ?? p.path ?? "");
  const dest = asStr(p.dest ?? p.to ?? p.target ?? p.destination ?? "");
  return `${baseName(src)} → ${tail(dest, 3) || dest}`;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function ApplyPanel({ report, onMutated }: ApplyPanelProps) {
  const [busy, setBusy] = useState<string | null>(null);
  const [tag, setTag] = useState<TagResult | null>(null);
  const [org, setOrg] = useState<OrgResult | null>(null);

  const runTags = async (dryRun: boolean) => {
    setBusy(dryRun ? "tags-preview" : "tags-write");
    try {
      const r = await api.writeTags(dryRun);
      const plans = Array.isArray(r.plans) ? r.plans : [];
      setTag({
        count: r.count,
        applied: r.applied,
        rows: plans
          .slice(0, MAX_ROWS)
          .map((p) => tagRow(p as Record<string, unknown>)),
      });
      report(
        `${dryRun ? "previewed" : "wrote"} tags · ${r.count} ${
          dryRun ? "(dry-run)" : "applied"
        }`,
      );
      if (!dryRun) onMutated();
    } catch (e) {
      report(`write-tags failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(null);
    }
  };

  const runOrganize = async (dryRun: boolean) => {
    setBusy(dryRun ? "org-preview" : "org-run");
    try {
      const r = await api.organize(dryRun);
      const plan = Array.isArray(r.plan) ? r.plan : [];
      setOrg({
        count: r.count,
        applied: r.applied,
        rows: plan
          .slice(0, MAX_ROWS)
          .map((p) => orgRow(p as Record<string, unknown>)),
      });
      report(
        `${dryRun ? "previewed" : "ran"} organize · ${r.count} ${
          dryRun ? "(dry-run)" : "applied"
        }`,
      );
      if (!dryRun) onMutated();
    } catch (e) {
      report(`organize failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(null);
    }
  };

  const runUndo = async () => {
    setBusy("undo");
    try {
      const r = await api.undo();
      report(`undid ${r.tags} tag writes, ${r.organize} file ops`);
      setTag(null);
      setOrg(null);
      onMutated();
    } catch (e) {
      report(`undo failed: ${errMsg(e)}`, true);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <div className="apply-group">
        <div className="apply-group__head">
          <span className="apply-group__title">Tags</span>
          <button
            className="btn btn--xs"
            onClick={() => runTags(true)}
            disabled={busy !== null}
          >
            {busy === "tags-preview" ? "…" : "Preview"}
          </button>
          <button
            className="btn btn--go btn--xs"
            onClick={() => runTags(false)}
            disabled={busy !== null}
          >
            {busy === "tags-write" ? "…" : "Write tags"}
          </button>
        </div>
        {tag && (
          <Result
            count={tag.count}
            applied={tag.applied}
            rows={tag.rows}
            kind="tag"
          />
        )}
      </div>

      <div className="apply-group">
        <div className="apply-group__head">
          <span className="apply-group__title">Organize</span>
          <button
            className="btn btn--xs"
            onClick={() => runOrganize(true)}
            disabled={busy !== null}
          >
            {busy === "org-preview" ? "…" : "Preview"}
          </button>
          <button
            className="btn btn--go btn--xs"
            onClick={() => runOrganize(false)}
            disabled={busy !== null}
          >
            {busy === "org-run" ? "…" : "Organize"}
          </button>
        </div>
        {org && (
          <Result
            count={org.count}
            applied={org.applied}
            rows={org.rows}
            kind="org"
          />
        )}
      </div>

      <div className="apply-group">
        <div className="apply-group__head">
          <span className="apply-group__title">Undo</span>
          <button
            className="btn btn--warn btn--xs"
            onClick={runUndo}
            disabled={busy !== null}
          >
            {busy === "undo" ? "…" : "Undo last apply"}
          </button>
        </div>
      </div>
    </>
  );
}

function Result({
  count,
  applied,
  rows,
  kind,
}: {
  count: number;
  applied: boolean;
  rows: string[];
  kind: "tag" | "org";
}) {
  return (
    <div className="apply-result">
      <div className="apply-result__meta">
        {count} {kind === "tag" ? "tag change(s)" : "file op(s)"} ·{" "}
        {applied ? (
          <span className="applied">applied</span>
        ) : (
          <span className="dry">dry-run</span>
        )}
      </div>
      {rows.length > 0 && (
        <div className="apply-rows">
          {rows.map((r, i) => (
            <div className="apply-rows__item" key={i} title={r}>
              {r}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
