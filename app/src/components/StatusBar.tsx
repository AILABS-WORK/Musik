interface StatusBarProps {
  message: string;
  isError: boolean;
  busy: boolean;
}

export function StatusBar({ message, isError, busy }: StatusBarProps) {
  return (
    <footer className={isError ? "statusbar statusbar--error" : "statusbar"}>
      <span className="statusbar__tag">{isError ? "error" : "status"}</span>
      {busy && <span className="busy-dot">●</span>}
      <span>{message || "ready"}</span>
    </footer>
  );
}
