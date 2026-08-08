import type { GraphEvent } from "@/lib/types/graph-events.generated";

/** The raw SSE event log — unchanged from the original streaming-shell view, just
 * extracted so AgentGraph can show it behind a dev-view toggle instead of by default. */
export function EventLog({ events }: { events: GraphEvent[] }) {
  return (
    <div className="border-rule flex-1 overflow-y-auto border p-2 font-mono text-xs">
      {events.length === 0 && <p className="text-abstain">No events yet.</p>}
      {events.map((event, i) => (
        <pre key={i} className="border-rule whitespace-pre-wrap border-b py-2 last:border-b-0">
          {JSON.stringify(event, null, 2)}
        </pre>
      ))}
    </div>
  );
}
