import type { GraphEvent } from "./types/graph-events.generated";

/** Mirrors api.routes.stream.StreamQueryRequest — see apps/api/src/api/routes/stream.py. */
export interface StreamQueryRequest {
  query: string;
  thread_id?: string | null;
  max_retries?: number | null;
}

/**
 * POSTs to /query/stream and yields typed GraphEvents as they arrive. Hand-rolled
 * because EventSource can't send a POST body and the AI SDK's stream reader expects its
 * own data-stream protocol, not this backend's SSE contract (id:/event:/data: frames of
 * JSON-encoded Pydantic events — see api.graph.streaming.stream_graph_events).
 */
export async function* streamQuery(
  request: StreamQueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<GraphEvent> {
  const response = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream request failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseFrame(frame);
        if (event) yield event;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): GraphEvent | null {
  let data: string | null = null;
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) {
      data = line.slice(5).trim();
    }
  }
  if (data === null) return null;
  return JSON.parse(data) as GraphEvent;
}
