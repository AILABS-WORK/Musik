import type { ReactNode } from "react";

export type SideTab = "genres" | "similar" | "clusters" | "apply";

interface SidePanelProps {
  active: SideTab;
  onTab: (t: SideTab) => void;
  children: ReactNode;
}

const TABS: { id: SideTab; label: string }[] = [
  { id: "genres", label: "Genres" },
  { id: "similar", label: "Similar" },
  { id: "clusters", label: "Clusters" },
  { id: "apply", label: "Apply" },
];

export function SidePanel({ active, onTab, children }: SidePanelProps) {
  return (
    <aside className="app__side">
      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={t.id === active ? "tab active" : "tab"}
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel-body">{children}</div>
    </aside>
  );
}
