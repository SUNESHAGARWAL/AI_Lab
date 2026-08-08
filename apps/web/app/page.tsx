"use client";

import { useRef, useState } from "react";

import { AgentGraph } from "@/components/graph/AgentGraph";
import { streamQuery } from "@/lib/sse-client";
import type { GraphEvent } from "@/lib/types/graph-events.generated";

export default function Home() {
  const [query, setQuery] = useState("");
  const [log, setLog] = useState<GraphEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function handleSubmit(formEvent: React.FormEvent) {
    formEvent.preventDefault();
    if (!query.trim() || isStreaming) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLog([]);
    setIsStreaming(true);

    try {
      for await (const event of streamQuery({ query }, controller.signal)) {
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

      <AgentGraph events={log} />
    </main>
  );
}
