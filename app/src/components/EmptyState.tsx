interface EmptyStateProps {
  /** Open the hidden file picker (same upload path as drag-drop). */
  onBrowse: () => void;
}

const STEPS: { n: string; title: string; body: string }[] = [
  {
    n: "1",
    title: "Load your music",
    body: "Drop files anywhere, hit Browse, or paste a folder path.",
  },
  {
    n: "2",
    title: "Tag & Analyze",
    body: "Sound profile, BPM, key and mood — automatically.",
  },
  {
    n: "3",
    title: "Search, organize & build sets",
    body: "Find tracks by sound, auto-classify, and assemble DJ sets.",
  },
];

/** Centered hero shown in the main area when the library is empty. */
export function EmptyState({ onBrowse }: EmptyStateProps) {
  return (
    <div className="empty">
      <div className="empty__inner">
        <div className="empty__glyph" aria-hidden="true">◈</div>
        <h1 className="empty__title">Build your sound library</h1>
        <p className="empty__lede">
          Musik understands your tracks by how they sound — then helps you
          search, classify and mix them.
        </p>

        <ol className="empty__steps">
          {STEPS.map((s) => (
            <li className="empty__step" key={s.n}>
              <span className="empty__step-n mono">{s.n}</span>
              <div className="empty__step-body">
                <span className="empty__step-title">{s.title}</span>
                <span className="empty__step-text">{s.body}</span>
              </div>
            </li>
          ))}
        </ol>

        <button className="btn btn--go btn--lg empty__cta" onClick={onBrowse}>
          <span className="addmusic__browse-icon" aria-hidden="true">⤓</span>
          Browse files to start
        </button>
        <span className="empty__sub">…or just drop audio onto the window.</span>
      </div>
    </div>
  );
}
