import { useMemo, useState } from "react";
import type { Genre } from "../types";

interface GenrePanelProps {
  genres: Genre[];
  /** ids of checked tracks, used as the "by example" set. */
  checkedIds: number[];
  /** Create a genre by example; resolves when done so the form can reset. */
  onByExample: (args: {
    name: string;
    parentId: number | null;
    level: string;
  }) => Promise<void>;
}

interface TreeNode extends Genre {
  children: TreeNode[];
}

/** Build a parent_id -> children tree, preserving input order. */
function buildTree(genres: Genre[]): TreeNode[] {
  const byId = new Map<number, TreeNode>();
  for (const g of genres) byId.set(g.id, { ...g, children: [] });
  const roots: TreeNode[] = [];
  for (const g of genres) {
    const node = byId.get(g.id);
    if (!node) continue;
    const parent =
      g.parent_id !== null && g.parent_id !== undefined
        ? byId.get(g.parent_id)
        : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

function flatten(nodes: TreeNode[], depth: number): { node: TreeNode; depth: number }[] {
  const out: { node: TreeNode; depth: number }[] = [];
  for (const n of nodes) {
    out.push({ node: n, depth });
    if (n.children.length) out.push(...flatten(n.children, depth + 1));
  }
  return out;
}

export function GenrePanel({ genres, checkedIds, onByExample }: GenrePanelProps) {
  const [name, setName] = useState("");
  const [parentId, setParentId] = useState<string>("");
  const [level, setLevel] = useState("subgenre");
  const [submitting, setSubmitting] = useState(false);

  const rows = useMemo(() => flatten(buildTree(genres), 0), [genres]);

  const canCreate = name.trim().length > 0 && checkedIds.length > 0 && !submitting;

  const submit = async () => {
    if (!canCreate) return;
    setSubmitting(true);
    try {
      await onByExample({
        name: name.trim(),
        parentId: parentId ? Number(parentId) : null,
        level,
      });
      setName("");
      setParentId("");
      setLevel("subgenre");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="panel-section">
        <h3>Taxonomy ({genres.length})</h3>
        {rows.length === 0 ? (
          <div className="hint">No genres yet.</div>
        ) : (
          <div className="genre-tree">
            {rows.map(({ node, depth }) => (
              <div
                key={node.id}
                className="genre-node"
                style={{ paddingLeft: 6 + depth * 16 }}
              >
                <span
                  className={
                    node.has_centroid
                      ? "genre-node__dot has-centroid"
                      : "genre-node__dot"
                  }
                  title={node.has_centroid ? "has centroid" : "no centroid"}
                />
                <span className="genre-node__name" title={node.name}>
                  {node.name}
                </span>
                <span className="genre-node__level">{node.level}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel-section">
        <h3>Create genre by example</h3>
        <div className="form">
          <div className="form__row">
            <label>Name</label>
            <input
              type="text"
              placeholder="e.g. Deep Dub Techno"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="form__row">
            <label>Parent (optional)</label>
            <select
              value={parentId}
              onChange={(e) => setParentId(e.target.value)}
            >
              <option value="">— none (top level) —</option>
              {rows.map(({ node, depth }) => (
                <option key={node.id} value={String(node.id)}>
                  {" ".repeat(depth * 2)}
                  {node.name}
                </option>
              ))}
            </select>
          </div>
          <div className="form__row">
            <label>Level</label>
            <select value={level} onChange={(e) => setLevel(e.target.value)}>
              <option value="subgenre">subgenre</option>
              <option value="genre">genre</option>
            </select>
          </div>
          <button
            className="btn btn--go"
            onClick={submit}
            disabled={!canCreate}
          >
            {submitting
              ? "Creating…"
              : `Create from ${checkedIds.length} selected`}
          </button>
        </div>
      </div>
    </>
  );
}
