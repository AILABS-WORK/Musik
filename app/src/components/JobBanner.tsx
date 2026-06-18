import type { Progress } from "../types";

/** The background jobs the banner can speak about, in plain language. */
export type JobKind = "embed" | "analyze" | "tag" | "identify" | "deep" | "fuse" | "mbseed" | "suggest" | "auto";

interface JobNote {
  kind: JobKind;
  /** Non-null => the job ended with this error. */
  error: string | null;
}

interface JobBannerProps {
  /** The job that is currently running, or null when idle. */
  kind: JobKind | null;
  /** Live progress for the running job; may be null on the very first tick. */
  progress: Progress | null;
  /** A short-lived "just finished" / "failed" note; null when nothing to show. */
  note: JobNote | null;
}

/** Friendly present-tense label for each job, e.g. "Tagging sounds". */
const RUNNING_LABEL: Record<JobKind, string> = {
  embed: "Listening to your tracks",
  analyze: "Analyzing BPM, key & energy",
  tag: "Tagging sounds",
  identify: "Identifying tracks by sound (AcoustID)",
  deep: "Deep analysis — separating stems & detecting language",
  fuse: "Fusing signals for sharper grouping",
  mbseed: "Building genres from MusicBrainz",
  suggest: "Classifying genres",
  auto: "Auto-sorting your library",
};

/** Friendly past-tense label for a completed job. */
const DONE_LABEL: Record<JobKind, string> = {
  embed: "Tracks listened to",
  analyze: "Analysis complete",
  tag: "Sounds tagged",
  identify: "Tracks identified",
  deep: "Deep analysis complete",
  fuse: "Signals fused",
  mbseed: "Genres seeded from MusicBrainz",
  suggest: "Genres classified",
  auto: "Auto-sort complete",
};

/**
 * A slim, friendly progress banner shown at the top of the main area while a
 * background job runs — and briefly after, to make completion and errors
 * impossible to miss. Plain-language labels (no jargon), a live count, and an
 * animated bar that switches to an indeterminate shimmer before totals arrive.
 */
export function JobBanner({ kind, progress, note }: JobBannerProps) {
  // Running takes priority over the finished note.
  if (kind !== null) {
    const total = progress?.total ?? 0;
    const done = progress?.done ?? 0;
    const known = total > 0;
    const pct = known ? Math.min(100, (done / total) * 100) : 0;
    const last = progress?.last?.trim() ?? "";

    return (
      <div
        className="jobbanner jobbanner--running"
        role="status"
        aria-live="polite"
      >
        <span className="jobbanner__spinner" aria-hidden="true" />
        <div className="jobbanner__body">
          <div className="jobbanner__head">
            <span className="jobbanner__label">
              {RUNNING_LABEL[kind]}
              <span className="jobbanner__ellipsis" aria-hidden="true">
                …
              </span>
            </span>
            <span className="jobbanner__count mono">
              {known ? `${done}/${total}` : "starting…"}
            </span>
          </div>
          <div className="jobbanner__track">
            <div
              className={
                known
                  ? "jobbanner__fill"
                  : "jobbanner__fill jobbanner__fill--indeterminate"
              }
              style={known ? { width: `${pct}%` } : undefined}
            />
          </div>
          {last !== "" && (
            <div className="jobbanner__last mono" title={last}>
              {last}
            </div>
          )}
        </div>
      </div>
    );
  }

  // No job running — maybe a freshly-finished note.
  if (note !== null) {
    const failed = note.error !== null;
    return (
      <div
        className={
          failed
            ? "jobbanner jobbanner--error"
            : "jobbanner jobbanner--done"
        }
        role="status"
        aria-live="polite"
      >
        <span className="jobbanner__glyph" aria-hidden="true">
          {failed ? "!" : "✓"}
        </span>
        <div className="jobbanner__body">
          <span className="jobbanner__label">
            {failed
              ? `${RUNNING_LABEL[note.kind]} failed`
              : DONE_LABEL[note.kind]}
          </span>
          {failed && (
            <div className="jobbanner__last" title={note.error ?? ""}>
              {note.error}
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}
