import { describe, expect, it } from "vitest";

import { NODE_ORDER } from "./graph-layout";
import { graphReducer, initialGraphVizState, selectTerminal, type Terminal } from "./graph-state";
import { cachedFixtureFor } from "./replay-client";
import type { GraphEvent } from "./types/graph-events.generated";

function replay(events: GraphEvent[]) {
  return events.reduce(graphReducer, initialGraphVizState());
}

const started: GraphEvent = {
  type: "graph_started",
  thread_id: "t",
  emitted_at: 1,
  query: "what does the EU AI Act mention on personal information handling",
};

/** Every node firing start->complete, i.e. the full six-node animation the live site
 * shows before rendering nothing. */
function fullNodeRun(): GraphEvent[] {
  return NODE_ORDER.flatMap((node, i) => [
    { type: "node_started", thread_id: "t", emitted_at: 2 + i, node, run_id: `r${i}` } as GraphEvent,
    {
      type: "node_completed",
      thread_id: "t",
      emitted_at: 2.5 + i,
      node,
      run_id: `r${i}`,
      latency_ms: 100,
      payload: {},
    } as GraphEvent,
  ]);
}

describe("selectTerminal", () => {
  it("renders the draft answer carried by a review interrupt", () => {
    // The exact production sequence behind the reported blank screen: hitl_gate
    // interrupts with a below-threshold draft and no graph_completed ever follows, so
    // this payload is the only place the answer exists.
    const events: GraphEvent[] = [
      started,
      ...fullNodeRun(),
      {
        type: "graph_interrupted",
        thread_id: "t",
        emitted_at: 9,
        interrupt: {
          type: "review",
          answer: "The EU AI Act addresses personal data handling through Article 10.",
          citations: ["aia-art-10"],
          confidence: 0.58,
          abstained: false,
          abstain_reason: null,
        },
      },
    ];

    const terminal = selectTerminal(replay(events), false);

    expect(terminal.kind).toBe("review");
    expect(terminal).toMatchObject({
      answer: "The EU AI Act addresses personal data handling through Article 10.",
      citations: ["aia-art-10"],
      confidence: 0.58,
    });
  });

  it("degrades a review interrupt with no draft text to an abstention", () => {
    const state = replay([
      started,
      {
        type: "graph_interrupted",
        thread_id: "t",
        emitted_at: 9,
        interrupt: { type: "review", answer: "", citations: [], confidence: 0.1, abstained: false },
      },
    ]);

    expect(selectTerminal(state, false).kind).toBe("abstain");
  });

  it("reads an explicitly abstained review interrupt as an abstention with its reason", () => {
    const state = replay([
      started,
      {
        type: "graph_interrupted",
        thread_id: "t",
        emitted_at: 9,
        interrupt: {
          type: "review",
          answer: "",
          citations: [],
          confidence: 0,
          abstained: true,
          abstain_reason: "no chunks were retrieved to ground an answer in",
        },
      },
    ]);

    expect(selectTerminal(state, false)).toEqual({
      kind: "abstain",
      reason: "no chunks were retrieved to ground an answer in",
    });
  });

  it("reads an out_of_scope interrupt as out_of_scope, carrying its reason", () => {
    const state = replay([
      started,
      {
        type: "graph_interrupted",
        thread_id: "t",
        emitted_at: 9,
        interrupt: { type: "out_of_scope", reason: "that looks like a greeting", message: "…" },
      },
    ]);

    expect(selectTerminal(state, false)).toEqual({
      kind: "out_of_scope",
      reason: "that looks like a greeting",
    });
  });

  it("reads a confident graph_completed as an answer", () => {
    const state = replay([
      started,
      {
        type: "graph_completed",
        thread_id: "t",
        emitted_at: 9,
        answer: "A DPIA is required when processing is likely to result in high risk.",
        citations: ["gdpr-art-35"],
        confidence: 0.91,
        abstained: false,
        human_approved: true,
      },
    ]);

    expect(selectTerminal(state, false)).toMatchObject({
      kind: "answer",
      citations: ["gdpr-art-35"],
    });
  });

  it("reads an abstained graph_completed as an abstention", () => {
    const state = replay([
      started,
      {
        type: "graph_completed",
        thread_id: "t",
        emitted_at: 9,
        answer: "",
        citations: [],
        confidence: 0,
        abstained: true,
        human_approved: null,
      },
    ]);

    expect(selectTerminal(state, false).kind).toBe("abstain");
  });

  it("marks only rate_limited and budget_exhausted errors as friendly", () => {
    const errorAt = (reason: string | null): Terminal =>
      selectTerminal(
        replay([
          started,
          {
            type: "error",
            thread_id: "t",
            emitted_at: 9,
            message: "m",
            retryable: true,
            ...(reason === null ? {} : { reason }),
          } as GraphEvent,
        ]),
        false,
      );

    expect(errorAt("rate_limited")).toMatchObject({ kind: "error", friendly: true });
    expect(errorAt("budget_exhausted")).toMatchObject({ kind: "error", friendly: true });
    expect(errorAt("provider_exhausted")).toMatchObject({ kind: "error", friendly: false });
    expect(errorAt(null)).toMatchObject({ kind: "error", friendly: false });
  });

  it("an error wins over a result, so a mid-stream failure is never hidden", () => {
    const state = replay([
      started,
      {
        type: "graph_completed",
        thread_id: "t",
        emitted_at: 8,
        answer: "a",
        citations: [],
        confidence: 0.9,
        abstained: false,
        human_approved: true,
      },
      { type: "error", thread_id: "t", emitted_at: 9, message: "boom", retryable: false },
    ]);

    expect(selectTerminal(state, false).kind).toBe("error");
  });

  it("distinguishes an open stream from one that ended with nothing", () => {
    const midRun = replay([started, ...fullNodeRun()]);
    expect(selectTerminal(midRun, true).kind).toBe("streaming");
    expect(selectTerminal(midRun, false).kind).toBe("no_terminal");
  });
});

describe("the never-blank guarantee", () => {
  /** The kinds TerminalPanel renders as a visible panel. `selectTerminal` is total over
   * GraphVizState, so if every kind it can return is in this set, no state can leave the
   * screen empty. */
  const RENDERABLE = new Set<Terminal["kind"]>([
    "answer",
    "review",
    "abstain",
    "out_of_scope",
    "error",
    "streaming",
    "no_terminal",
  ]);

  /** Every recorded live run behind the curated examples — the real event shapes off the
   * wire, not synthesized ones. */
  const FIXTURES: GraphEvent[][] = [
    "What are the requirements for high-risk AI systems under the EU AI Act?",
    "What is a data protection impact assessment under GDPR?",
    "How does the GDPR define personal data?",
    "How do the fines for AI Act deployer violations compare to GDPR fines?",
  ]
    .map((q) => cachedFixtureFor(q))
    .filter((f): f is GraphEvent[] => f !== undefined);

  const SEQUENCES: GraphEvent[][] = [[], [started], [started, ...fullNodeRun()], ...FIXTURES];

  it("resolves every prefix of every known event sequence to something renderable", () => {
    for (const sequence of SEQUENCES) {
      for (let i = 0; i <= sequence.length; i++) {
        const state = replay(sequence.slice(0, i));
        for (const isStreaming of [true, false]) {
          const terminal = selectTerminal(state, isStreaming);
          expect(RENDERABLE.has(terminal.kind), `${terminal.kind} at prefix ${i}`).toBe(true);
        }
      }
    }
  });

  it("never yields an answer or review kind with empty text", () => {
    for (const sequence of SEQUENCES) {
      for (let i = 0; i <= sequence.length; i++) {
        const terminal = selectTerminal(replay(sequence.slice(0, i)), false);
        if (terminal.kind === "answer" || terminal.kind === "review") {
          expect(terminal.answer.trim()).not.toBe("");
        }
      }
    }
  });

  it("resolves every fully-replayed curated example to a real result, never no_terminal", () => {
    expect(FIXTURES.length).toBeGreaterThan(0);
    for (const fixture of FIXTURES) {
      const terminal = selectTerminal(replay(fixture), false);
      expect(["answer", "review", "abstain", "out_of_scope"]).toContain(terminal.kind);
    }
  });
});
