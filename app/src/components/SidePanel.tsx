import type { ReactNode } from "react";

export type SideTab = "genres" | "similar" | "song" | "clusters" | "set" | "id" | "mix" | "apply";

interface SidePanelProps {
  active: SideTab;
  onTab: (t: SideTab) => void;
  children: ReactNode;
}

interface TabDef {
  id: SideTab;
  label: string;
  /** One-line description — shown as a tooltip and as the active subheading. */
  blurb: string;
}

const TABS: TabDef[] = [
  { id: "genres", label: "Genres", blurb: "auto-classify by example" },
  { id: "similar", label: "Similar", blurb: "tracks that sound alike" },
  { id: "song", label: "Song", blurb: "this track's sound profile" },
  { id: "clusters", label: "Clusters", blurb: "group tracks by shared sound" },
  { id: "set", label: "Set", blurb: "build a DJ set from a vibe" },
  { id: "id", label: "ID", blurb: "identify a track by sound" },
  { id: "mix", label: "Mix", blurb: "tracklist a whole mix" },
  { id: "apply", label: "Apply", blurb: "write tags / organize files" },
];

export function SidePanel({ active, onTab, children }: SidePanelProps) {
  const activeDef = TABS.find((t) => t.id === active);
  return (
    <aside className="app__side">
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={t.id === active}
            title={`${t.label} — ${t.blurb}`}
            className={t.id === active ? "tab active" : "tab"}
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {activeDef && (
        <div className="panel-subhead">
          <span className="panel-subhead__label">{activeDef.label}</span>
          <span className="panel-subhead__blurb">{activeDef.blurb}</span>
        </div>
      )}
      <div className="panel-body">{children}</div>
    </aside>
  );
}
