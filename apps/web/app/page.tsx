"use client";

import { useRef, useState } from "react";

import { AgentGraph } from "@/components/graph/AgentGraph";
import type { WaitPhase } from "@/components/graph/TerminalPanel";
import { Landing, QUERY_PLACEHOLDER } from "@/components/landing/Landing";
import { ScopeNote } from "@/components/landing/ScopeNote";
import { cachedFixtureFor, replayCachedQuery } from "@/lib/replay-client";
import { streamQuery } from "@/lib/sse-client";
import type { GraphEvent } from "@/lib/types/graph-events.generated";

/** Railway's free tier cold-starts, and a cold start can take minutes. These stage the
 * waiting copy so the screen states what it's doing the whole time, and the hard cap
 * turns a genuinely hung request into a visible timeout rather than an indefinite spin.
 * Generous on purpose: aborting a request that was merely waking up would report a
 * failure for something that would have succeeded. */
const WAKING_AFTER_MS = 4_000;
const STILL_WAKING_AFTER_MS = 20_000;
const HARD_TIMEOUT_MS = 90_000;

const TERMINAL_EVENT_TYPES = new Set(["graph_completed", "graph_interrupted", "error"]);

function clientErrorEvent(message: string, retryable: boolean): GraphEvent {
  return {
    type: "error",
    thread_id: "client",
    emitted_at: Date.now() / 1000,
    message,
    retryable,
  };
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [log, setLog] = useState<GraphEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [waitPhase, setWaitPhase] = useState<WaitPhase>("thinking");
  const [hasStarted, setHasStarted] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  /** Synchronous single-flight latch. `isStreaming` is React state, so two submits in
   * the same tick (a double-click, or Enter landing on an already-submitting form) both
   * read the pre-render value and fire duplicate streams. A ref is readable back
   * immediately, so it actually holds. */
  const inFlightRef = useRef(false);
  /** Distinguishes our own timeout abort from a user-initiated one, so a timeout still
   * renders a message while a superseded query stays quiet. */
  const timedOutRef = useRef(false);

  /** Shared by the manual form's submit and every example-question click — the seam a
   * later demo-mode cache can wrap or branch inside without touching either call site. */
  async function runQuery(text: string) {
    if (!text.trim() || inFlightRef.current) return;
    inFlightRef.current = true;
    timedOutRef.current = false;

    setHasStarted(true);
    setQuery(text);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLog([]);
    setWaitPhase("thinking");
    setIsStreaming(true);

    const timers = [
      setTimeout(() => setWaitPhase("waking"), WAKING_AFTER_MS),
      setTimeout(() => setWaitPhase("still_waking"), STILL_WAKING_AFTER_MS),
      setTimeout(() => {
        timedOutRef.current = true;
        controller.abort();
      }, HARD_TIMEOUT_MS),
    ];

    let sawTerminal = false;

    try {
      // Curated example questions replay from a recorded fixture — zero live gateway
      // cost. A free-typed question only takes this path if it happens to match one
      // verbatim, which is correct: same question, same honest answer either way.
      const fixture = cachedFixtureFor(text);
      const source = fixture
        ? replayCachedQuery(fixture, controller.signal)
        : streamQuery({ query: text }, controller.signal);
      for await (const event of source) {
        if (TERMINAL_EVENT_TYPES.has(event.type)) sawTerminal = true;
        setLog((prev) => [...prev, event]);
      }

      // A stream that closed cleanly without ever resolving is a bug — a truncated
      // response, a dropped connection, a 200 with no frames. Say so visibly rather than
      // leaving the screen on the last node that happened to animate.
      if (!sawTerminal && !controller.signal.aborted) {
        setLog((prev) => [
          ...prev,
          clientErrorEvent(
            "The response ended before a final answer arrived. Nothing was lost on your side — try again, or use one of the example questions.",
            true,
          ),
        ]);
      }
    } catch (err) {
      if (timedOutRef.current) {
        setLog((prev) => [
          ...prev,
          clientErrorEvent(
            "The backend didn't respond in time. It may still be waking up from a cold start — try again in a moment, or use one of the example questions, which never need the backend.",
            true,
          ),
        ]);
      } else if (!controller.signal.aborted) {
        setLog((prev) => [
          ...prev,
          clientErrorEvent(err instanceof Error ? err.message : String(err), false),
        ]);
      }
    } finally {
      for (const timer of timers) clearTimeout(timer);
      setIsStreaming(false);
      inFlightRef.current = false;
    }
  }

  function handleSubmit(formEvent: React.FormEvent) {
    formEvent.preventDefault();
    runQuery(query);
  }

  if (!hasStarted) {
    return (
      <main className="mx-auto flex min-h-screen max-w-4xl flex-col p-6">
        <Landing onAsk={runQuery} />
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-4 p-6">
      <h1 className="font-serif text-2xl">AI Lab — streaming shell</h1>

      <div>
        <form onSubmit={handleSubmit} className="flex gap-2">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={QUERY_PLACEHOLDER}
            rows={3}
            className="border-rule flex-1 border p-2 font-serif"
          />
          <button
            type="submit"
            disabled={isStreaming || !query.trim()}
            className="border-rule bg-active shrink-0 self-start border px-4 py-2 font-mono text-sm text-paper disabled:opacity-40"
          >
            {isStreaming ? "Streaming…" : "Ask"}
          </button>
        </form>
        <ScopeNote />
      </div>

      <AgentGraph
        events={log}
        isStreaming={isStreaming}
        waitPhase={waitPhase}
        onBackToExamples={() => setHasStarted(false)}
      />
    </main>
  );
}
