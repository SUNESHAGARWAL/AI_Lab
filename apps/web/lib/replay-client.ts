import exampleFixtures from "./example-fixtures.json";
import type { GraphEvent } from "./types/graph-events.generated";

/** Recorded from real live runs of the exact questions in
 * components/landing/ExampleQuestions.tsx's EXAMPLES array (see the plan this was
 * built from) — real citations, real abstain reasons, nothing synthesized. Keyed by
 * the exact question string so a free-typed question that happens to match an example
 * verbatim also gets the free cached path, correctly (same question, same honest
 * answer, no reason to spend money twice). */
const FIXTURES = exampleFixtures as Record<string, GraphEvent[]>;

export function cachedFixtureFor(question: string): GraphEvent[] | undefined {
  return FIXTURES[question];
}

const MAX_REPLAY_DELAY_MS = 2500;

/** Same `AsyncGenerator<GraphEvent>` shape as sse-client.ts's streamQuery — a drop-in
 * swap wherever that's consumed. Replays a recorded run's real inter-event gaps
 * (`emitted_at` deltas, not fabricated timing) so the animation reads identically to a
 * live run, capped so a fixture's real gaps never make a visitor wait longer than a
 * live run reasonably would. */
export async function* replayCachedQuery(
  events: GraphEvent[],
  signal?: AbortSignal,
): AsyncGenerator<GraphEvent> {
  let prevEmittedAt = events[0]?.emitted_at ?? 0;
  for (const event of events) {
    const deltaMs = (event.emitted_at - prevEmittedAt) * 1000;
    const delayMs = Math.min(Math.max(deltaMs, 0), MAX_REPLAY_DELAY_MS);
    prevEmittedAt = event.emitted_at;
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    if (signal?.aborted) return;
    yield event;
  }
}
