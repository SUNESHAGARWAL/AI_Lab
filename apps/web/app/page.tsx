"use client";

import { useRef, useState } from "react";

import { AgentGraph } from "@/components/graph/AgentGraph";
import { Landing } from "@/components/landing/Landing";
import { cachedFixtureFor, replayCachedQuery } from "@/lib/replay-client";
import { streamQuery } from "@/lib/sse-client";
import type { GraphEvent } from "@/lib/types/graph-events.generated";

export default function Home() {
  const [query, setQuery] = useState("");
  const [log, setLog] = useState<GraphEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  /** Shared by the manual form's submit and every example-question click — the seam a
   * later demo-mode cache can wrap or branch inside without touching either call site. */
  async function runQuery(text: string) {
    if (!text.trim() || isStreaming) return;
    setHasStarted(true);
    setQuery(text);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLog([]);
    setIsStreaming(true);

    try {
      // Curated example questions replay from a recorded fixture — zero live gateway
      // cost. A free-typed question only takes this path if it happens to match one
      // verbatim, which is correct: same question, same honest answer either way.
      const fixture = cachedFixtureFor(text);
      const source = fixture
        ? replayCachedQuery(fixture, controller.signal)
        : streamQuery({ query: text }, controller.signal);
      for await (const event of source) {
        setLog((prev) => [...prev, event]);
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setLog((prev) => [
          ...prev,
          {
            type: "error",
            thread_id: "client",
            emitted_at: Date.now() / 1000,
            message: err instanceof Error ? err.message : String(err),
            retryable: false,
          },
        ]);
      }
    } finally {
      setIsStreaming(false);
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

      <form onSubmit={handleSubmit} className="flex gap-2">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about the EU AI Act or GDPR…"
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

      <AgentGraph events={log} onBackToExamples={() => setHasStarted(false)} />
    </main>
  );
}
