import { useCallback, useEffect, useRef, useState } from "react";
import "./mobile.css";

/* ============================================================================
 * Musik — Record & Identify (phone-first companion view)
 *
 * Records ~10s from the mic, encodes a real 16-bit mono PCM WAV at 16 kHz
 * (libsndfile-friendly — never webm/opus), base64-encodes it, and POSTs to the
 * engine's /api/identify-upload. Matches the user's own library, Shazam-style.
 *
 * The engine base URL is user-editable + persisted, because a phone can't reach
 * the desktop's 127.0.0.1 — it needs the desktop's LAN IP.
 * ========================================================================== */

interface Match {
  track_id: number;
  name: string;
  score: number;
}

type Phase =
  | "idle"
  | "requesting" // asking for mic permission
  | "recording"
  | "identifying"
  | "results"
  | "error";

const RECORD_SECONDS = 10;
const TARGET_RATE = 16000; // engine-friendly sample rate
const ENGINE_KEY = "musik.engineBase";

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Default engine URL: same protocol/host as the page, on the engine port. */
function defaultEngineBase(): string {
  const proto =
    location.protocol === "https:" ? "https:" : "http:";
  const host = location.hostname || "127.0.0.1";
  return `${proto}//${host}:8000`;
}

function loadEngineBase(): string {
  try {
    const saved = localStorage.getItem(ENGINE_KEY);
    if (saved && saved.trim()) return saved.trim();
  } catch {
    /* private mode / storage disabled — fall through to default */
  }
  return defaultEngineBase();
}

/* ---- WAV encoding -------------------------------------------------------- */

/**
 * Encode mono Float32 PCM (in [-1, 1]) as a 16-bit little-endian WAV.
 * Prepends the standard 44-byte RIFF/WAVE header so libsndfile decodes it.
 */
function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples = samples.length;
  const bytesPerSample = 2; // 16-bit
  const blockAlign = bytesPerSample; // mono
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  // RIFF chunk descriptor
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  // fmt sub-chunk
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // audio format = 1 (PCM)
  view.setUint16(22, 1, true); // channels = 1 (mono)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  // data sub-chunk
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  // Float32 [-1,1] -> Int16
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    let s = samples[i] ?? 0;
    s = s < -1 ? -1 : s > 1 ? 1 : s;
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

/** ArrayBuffer -> base64 (chunked to avoid call-stack limits on big clips). */
function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/* ---- component ----------------------------------------------------------- */

export function MobileRecord() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [remaining, setRemaining] = useState(RECORD_SECONDS);
  const [level, setLevel] = useState(0); // 0..1 mic level for the meter
  const [engineBase, setEngineBase] = useState<string>(loadEngineBase);
  const [editingEngine, setEditingEngine] = useState(false);
  const [engineDraft, setEngineDraft] = useState(engineBase);

  // Audio graph + capture state. Kept in refs so the unmount cleanup and the
  // stop handler can always reach the live objects.
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const recordedRateRef = useRef<number>(TARGET_RATE);
  const stopTimerRef = useRef<number | null>(null);
  const tickTimerRef = useRef<number | null>(null);
  // Guards so a tap-to-stop and the auto-stop timer can't both finalize.
  const finishingRef = useRef(false);

  const persistEngine = useCallback((value: string) => {
    const v = value.trim() || defaultEngineBase();
    setEngineBase(v);
    try {
      localStorage.setItem(ENGINE_KEY, v);
    } catch {
      /* ignore storage failures */
    }
  }, []);

  // Tear down the audio graph + timers. Safe to call repeatedly.
  const teardown = useCallback(() => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (tickTimerRef.current !== null) {
      window.clearInterval(tickTimerRef.current);
      tickTimerRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      try {
        processorRef.current.disconnect();
      } catch {
        /* already disconnected */
      }
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        /* already disconnected */
      }
      sourceRef.current = null;
    }
    if (streamRef.current) {
      for (const t of streamRef.current.getTracks()) t.stop();
      streamRef.current = null;
    }
    if (ctxRef.current) {
      void ctxRef.current.close().catch(() => undefined);
      ctxRef.current = null;
    }
  }, []);

  useEffect(() => teardown, [teardown]);

  // Downsample captured PCM (recorded at the AudioContext rate) to TARGET_RATE
  // with simple averaging, so the WAV we send is always 16 kHz mono.
  const downsample = useCallback(
    (input: Float32Array, fromRate: number): Float32Array => {
      if (fromRate === TARGET_RATE) return input;
      const ratio = fromRate / TARGET_RATE;
      const outLen = Math.floor(input.length / ratio);
      const out = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const start = Math.floor(i * ratio);
        const end = Math.min(Math.floor((i + 1) * ratio), input.length);
        let sum = 0;
        let n = 0;
        for (let j = start; j < end; j++) {
          sum += input[j] ?? 0;
          n++;
        }
        out[i] = n > 0 ? sum / n : 0;
      }
      return out;
    },
    [],
  );

  const identify = useCallback(
    async (wavBase64: string) => {
      setPhase("identifying");
      try {
        const base = engineBase.replace(/\/+$/, "");
        const res = await fetch(`${base}/api/identify-upload`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `mic-${Date.now()}.wav`,
            data_base64: wavBase64,
            n: 5,
          }),
        });
        if (!res.ok) {
          throw new Error(`engine ${res.status}: ${await res.text()}`);
        }
        const data = (await res.json()) as {
          matches?: Match[];
          error?: string;
        };
        if (data.error) throw new Error(data.error);
        setMatches(data.matches ?? []);
        setPhase("results");
      } catch (e) {
        setError(
          `Couldn't reach the engine. Check the address below — on a phone it must be your desktop's LAN IP, not 127.0.0.1. (${errMsg(e)})`,
        );
        setPhase("error");
      }
    },
    [engineBase],
  );

  // Finalize a recording: assemble chunks -> downsample -> WAV -> base64 -> POST.
  const finish = useCallback(() => {
    if (finishingRef.current) return;
    finishingRef.current = true;

    const fromRate = recordedRateRef.current;
    const chunks = chunksRef.current;
    chunksRef.current = [];
    teardown();

    const total = chunks.reduce((n, c) => n + c.length, 0);
    if (total === 0) {
      setError("No audio was captured. Try again and allow the mic.");
      setPhase("error");
      return;
    }
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    const pcm = downsample(merged, fromRate);
    const wav = encodeWav(pcm, TARGET_RATE);
    const b64 = arrayBufferToBase64(wav);
    void identify(b64);
  }, [downsample, identify, teardown]);

  const startRecording = useCallback(async () => {
    setError(null);
    setMatches(null);
    finishingRef.current = false;
    chunksRef.current = [];
    setRemaining(RECORD_SECONDS);
    setLevel(0);
    setPhase("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      type ACtor = typeof AudioContext;
      const Ctor: ACtor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: ACtor })
          .webkitAudioContext!;
      const ctx = new Ctor();
      ctxRef.current = ctx;
      recordedRateRef.current = ctx.sampleRate;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      // ScriptProcessor is deprecated but universally available and dependency-
      // free; perfect for a short capture. 4096-frame buffer, mono in/out.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (ev: AudioProcessingEvent) => {
        const input = ev.inputBuffer.getChannelData(0);
        // copy — the underlying buffer is reused by the browser
        chunksRef.current.push(new Float32Array(input));
        // RMS level for the meter
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
          const s = input[i] ?? 0;
          sum += s * s;
        }
        const rms = Math.sqrt(sum / input.length);
        setLevel(Math.min(1, rms * 4));
      };

      source.connect(processor);
      // Route through a muted gain node so onaudioprocess fires without
      // echoing the mic back to the speakers.
      const sink = ctx.createGain();
      sink.gain.value = 0;
      processor.connect(sink);
      sink.connect(ctx.destination);

      setPhase("recording");

      // Auto-stop after RECORD_SECONDS.
      stopTimerRef.current = window.setTimeout(finish, RECORD_SECONDS * 1000);
      // Countdown tick.
      tickTimerRef.current = window.setInterval(() => {
        setRemaining((r) => (r > 0 ? r - 1 : 0));
      }, 1000);
    } catch (e) {
      teardown();
      const msg = errMsg(e);
      const denied = /denied|notallowed|permission/i.test(msg);
      setError(
        denied
          ? "Microphone access was blocked. Allow the mic in your browser settings and try again."
          : `Couldn't start the mic: ${msg}`,
      );
      setPhase("error");
    }
  }, [finish, teardown]);

  const stopNow = useCallback(() => {
    if (phase === "recording") finish();
  }, [phase, finish]);

  const reset = useCallback(() => {
    teardown();
    finishingRef.current = false;
    chunksRef.current = [];
    setMatches(null);
    setError(null);
    setRemaining(RECORD_SECONDS);
    setLevel(0);
    setPhase("idle");
  }, [teardown]);

  /* ---- render ----------------------------------------------------------- */

  const onButtonTap = () => {
    if (phase === "idle" || phase === "results" || phase === "error") {
      void startRecording();
    } else if (phase === "recording") {
      stopNow();
    }
  };

  const buttonLabel =
    phase === "recording"
      ? "Stop"
      : phase === "requesting"
        ? "…"
        : phase === "identifying"
          ? "…"
          : "Record";

  const buttonDisabled = phase === "requesting" || phase === "identifying";

  return (
    <div className="mrec">
      <header className="mrec__header">
        <div className="mrec__brand">
          <span className="mrec__logo">Musik</span>
          <span className="mrec__tag">Record &amp; Identify</span>
        </div>
      </header>

      <main className="mrec__main">
        <button
          type="button"
          className={`mrec__btn mrec__btn--${phase}`}
          onClick={onButtonTap}
          disabled={buttonDisabled}
          aria-label={buttonLabel}
          style={
            phase === "recording"
              ? ({ "--lvl": String(level) } as React.CSSProperties)
              : undefined
          }
        >
          {phase === "recording" && (
            <span className="mrec__ring" aria-hidden="true" />
          )}
          <span className="mrec__btn-inner">
            {phase === "recording" ? (
              <span className="mrec__count">{remaining}</span>
            ) : phase === "identifying" || phase === "requesting" ? (
              <span className="mrec__spinner" aria-hidden="true" />
            ) : (
              <span className="mrec__mic" aria-hidden="true">
                ●
              </span>
            )}
          </span>
        </button>

        <p className="mrec__status">
          {phase === "idle" && "Tap to record ~10s of what's playing."}
          {phase === "requesting" && "Allow microphone access…"}
          {phase === "recording" && "Listening… tap to stop early."}
          {phase === "identifying" && "Matching against your library…"}
          {phase === "results" &&
            (matches && matches.length > 0
              ? "Best matches from your crate:"
              : "No match found in your library.")}
          {phase === "error" && "Something went wrong."}
        </p>

        {phase === "recording" && (
          <div className="mrec__meter" aria-hidden="true">
            <div
              className="mrec__meter-fill"
              style={{ width: `${Math.round(level * 100)}%` }}
            />
          </div>
        )}

        {phase === "error" && error && (
          <div className="mrec__error" role="alert">
            {error}
          </div>
        )}

        {phase === "results" && matches && matches.length > 0 && (
          <ul className="mrec__results">
            {matches.map((m) => {
              const pct = Math.round(Math.max(0, Math.min(1, m.score)) * 100);
              return (
                <li className="mrec__result" key={m.track_id}>
                  <div className="mrec__result-top">
                    <span className="mrec__result-name" title={m.name}>
                      {m.name}
                    </span>
                    <span className="mrec__result-score">{pct}%</span>
                  </div>
                  <div className="mrec__score-track">
                    <div
                      className="mrec__score-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        {(phase === "results" || phase === "error") && (
          <button
            type="button"
            className="mrec__again"
            onClick={reset}
            disabled={false}
          >
            Record again
          </button>
        )}
      </main>

      <footer className="mrec__footer">
        {editingEngine ? (
          <div className="mrec__engine-edit">
            <label className="mrec__engine-label" htmlFor="mrec-engine">
              Engine address (your desktop's LAN IP)
            </label>
            <input
              id="mrec-engine"
              className="mrec__engine-input"
              type="text"
              inputMode="url"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              placeholder="http://192.168.1.42:8000"
              value={engineDraft}
              onChange={(e) => setEngineDraft(e.target.value)}
            />
            <div className="mrec__engine-actions">
              <button
                type="button"
                className="mrec__engine-save"
                onClick={() => {
                  persistEngine(engineDraft);
                  setEditingEngine(false);
                }}
              >
                Save
              </button>
              <button
                type="button"
                className="mrec__engine-cancel"
                onClick={() => {
                  setEngineDraft(engineBase);
                  setEditingEngine(false);
                }}
              >
                Cancel
              </button>
            </div>
            <p className="mrec__engine-hint">
              On a phone, <code>127.0.0.1</code> points at the phone itself.
              Enter the desktop's LAN IP (e.g. <code>192.168.x.x</code>) so it
              can reach the engine.
            </p>
          </div>
        ) : (
          <button
            type="button"
            className="mrec__engine-pill"
            onClick={() => {
              setEngineDraft(engineBase);
              setEditingEngine(true);
            }}
          >
            <span className="mrec__engine-dot" aria-hidden="true" />
            <span className="mrec__engine-url">{engineBase}</span>
            <span className="mrec__engine-edit-glyph" aria-hidden="true">
              edit
            </span>
          </button>
        )}
      </footer>
    </div>
  );
}
