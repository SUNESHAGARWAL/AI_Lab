import { NODE_ORDER } from "./graph-layout";
import type { GraphCompletedEvent, GraphErrorEvent, GraphEvent, NodeName } from "./types/graph-events.generated";

export type NodeStatus = "idle" | "active" | "completed" | "pending_review" | "abstain" | "error";

export interface NodeVizState {
  status: NodeStatus;
  runId: string | null;
  latencyMs: number | null;
  /** Sum of llm_calls[].estimated_cost_usd. Null (not zero) when the node made no
   * gateway call at all — retriever/reranker/hitl_gate on the happy path. */
  estimatedCostUsd: number | null;
  payload: Record<string, unknown> | null;
}

/** Retriever's payload.candidates / reranker's payload.reranked — see
 * api.graph.events.retriever_payload / reranker_payload. Only the length is used here;
 * the chunk shape itself belongs to RetrievalFanOut, not this reducer. */
interface RetrieverPayload {
  candidates?: unknown[];
}
interface RerankerPayload {
  reranked?: unknown[];
}

export interface FanOutState {
  phase: "idle" | "fanned" | "collapsing";
  candidateCount: number;
  rerankedCount: number;
}

export interface RetryState {
  active: boolean;
  count: number;
  reason: string | null;
}

export interface GraphVizState {
  nodes: Record<NodeName, NodeVizState>;
  /** The edge currently drawing — control just passed from `from` (completed) to `to`
   * (just started). Cleared once GraphEdge finishes its draw animation. */
  activeEdge: { from: NodeName; to: NodeName } | null;
  drawnEdges: Set<string>;
  retry: RetryState | null;
  fanOut: FanOutState;
  interrupt: Record<string, unknown> | null;
  completed: GraphCompletedEvent | null;
  error: GraphErrorEvent | null;
}

function initialNodeState(): NodeVizState {
  return { status: "idle", runId: null, latencyMs: null, estimatedCostUsd: null, payload: null };
}

export function initialGraphVizState(): GraphVizState {
  const nodes = {} as Record<NodeName, NodeVizState>;
  for (const name of NODE_ORDER) nodes[name] = initialNodeState();
  return {
    nodes,
    activeEdge: null,
    drawnEdges: new Set(),
    retry: null,
    fanOut: { phase: "idle", candidateCount: 0, rerankedCount: 0 },
    interrupt: null,
    completed: null,
    error: null,
  };
}

function lastCompletedNode(nodes: Record<NodeName, NodeVizState>, before: NodeName): NodeName | null {
  const idx = NODE_ORDER.indexOf(before);
  for (let i = idx - 1; i >= 0; i--) {
    const candidate = NODE_ORDER[i]!;
    if (nodes[candidate].status === "completed") return candidate;
  }
  return null;
}

function currentlyActiveNode(nodes: Record<NodeName, NodeVizState>): NodeName | null {
  return NODE_ORDER.find((n) => nodes[n].status === "active") ?? null;
}

/** Pure reducer: GraphEvent -> GraphVizState. No timers, no component lifecycle — the
 * fan-out collapse-to-idle and edge-draw-to-settled transitions are driven by the
 * components themselves (via onAnimationComplete), not by this function, so this stays
 * a straightforward snapshot-of-last-known-event. */
export function graphReducer(state: GraphVizState, event: GraphEvent): GraphVizState {
  switch (event.type) {
    case "graph_started":
      return initialGraphVizState();

    case "node_started": {
      const nodes = { ...state.nodes, [event.node]: { ...initialNodeState(), status: "active" as const, runId: event.run_id } };
      const from = lastCompletedNode(nodes, event.node);
      return {
        ...state,
        nodes,
        activeEdge: from ? { from, to: event.node } : state.activeEdge,
      };
    }

    case "node_completed": {
      const llmCalls = event.llm_calls ?? [];
      const estimatedCostUsd = llmCalls.length > 0 ? llmCalls.reduce((sum, c) => sum + c.estimated_cost_usd, 0) : null;
      const payload = (event.payload ?? {}) as Record<string, unknown>;
      const nodes: Record<NodeName, NodeVizState> = {
        ...state.nodes,
        [event.node]: {
          status: "completed",
          runId: event.run_id,
          latencyMs: event.latency_ms,
          estimatedCostUsd,
          payload,
        },
      };

      let fanOut = state.fanOut;
      if (event.node === "retriever") {
        const { candidates } = payload as RetrieverPayload;
        fanOut = { phase: "fanned", candidateCount: candidates?.length ?? 0, rerankedCount: 0 };
      } else if (event.node === "reranker") {
        const { reranked } = payload as RerankerPayload;
        fanOut = { ...state.fanOut, phase: "collapsing", rerankedCount: reranked?.length ?? 0 };
      }

      const drawnEdges = state.activeEdge ? new Set(state.drawnEdges).add(`${state.activeEdge.from}->${state.activeEdge.to}`) : state.drawnEdges;

      return { ...state, nodes, fanOut, drawnEdges };
    }

    case "retry_loop":
      return { ...state, retry: { active: true, count: event.retry_count, reason: event.reason ?? null } };

    case "graph_interrupted": {
      const interrupt = event.interrupt;
      const type = interrupt.type;
      let hitlStatus: NodeStatus = "pending_review";
      if (type === "out_of_scope") {
        hitlStatus = "abstain";
      } else if (type === "review" && interrupt.abstained === true) {
        hitlStatus = "abstain";
      } else if (type === "review") {
        hitlStatus = "pending_review";
      }
      return {
        ...state,
        interrupt,
        nodes: {
          ...state.nodes,
          hitl_gate: { ...state.nodes.hitl_gate, status: hitlStatus },
        },
      };
    }

    case "graph_completed":
      return {
        ...state,
        completed: event,
        nodes: { ...state.nodes, hitl_gate: { ...state.nodes.hitl_gate, status: "completed" } },
      };

    case "error": {
      const active = currentlyActiveNode(state.nodes);
      if (!active) return { ...state, error: event };
      return {
        ...state,
        error: event,
        nodes: { ...state.nodes, [active]: { ...state.nodes[active], status: "error" } },
      };
    }

    default:
      return state;
  }
}

/** The one thing a finished (or still-running) query resolves to on screen. Exhaustive
 * by construction: `selectTerminal` returns exactly one of these for *any* GraphVizState,
 * including states no backend should ever produce. That total-ness is the never-blank
 * guarantee — the bug this replaced was three components each independently deciding not
 * to render, with a real cited answer sitting unread in `interrupt`. */
export type Terminal =
  | { kind: "answer"; answer: string; citations: string[]; confidence: number | null }
  | { kind: "review"; answer: string; citations: string[]; confidence: number | null }
  | { kind: "abstain"; reason: string }
  | { kind: "out_of_scope"; reason: string | null }
  | { kind: "error"; message: string; retryable: boolean; friendly: boolean }
  | { kind: "streaming" }
  | { kind: "no_terminal" };

/** `reason` on GraphErrorEvent, for the two cases that are a budget fact about the demo
 * rather than a fault — they get visitor-facing framing, not a raw error box. */
const FRIENDLY_ERROR_REASONS = new Set(["rate_limited", "budget_exhausted"]);

function str(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function strList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

function num(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

const NO_SUPPORT = "The available sources didn't support a confident answer.";

/** GraphInterruptedEvent.interrupt -> Terminal. The `review` framing is the one the
 * production stream actually sends for a below-threshold answer (see
 * api.graph.nodes.make_hitl_gate_node): it carries the full draft and its citations, and
 * no `graph_completed` ever follows it, so this is the *only* place that answer exists. */
function terminalFromInterrupt(interrupt: Record<string, unknown>): Terminal {
  if (interrupt.type === "out_of_scope") {
    return { kind: "out_of_scope", reason: str(interrupt.reason) };
  }

  if (interrupt.abstained === true) {
    return { kind: "abstain", reason: str(interrupt.abstain_reason) ?? NO_SUPPORT };
  }

  const answer = str(interrupt.answer);
  // A draft with no text is not an answer, whatever the abstained flag says — degrade
  // rather than render an empty box.
  if (!answer) return { kind: "abstain", reason: str(interrupt.abstain_reason) ?? NO_SUPPORT };

  return {
    kind: "review",
    answer,
    citations: strList(interrupt.citations),
    confidence: num(interrupt.confidence),
  };
}

/** Collapses a whole run into the single panel to show. Total function: every branch
 * returns, and the fallthrough is `no_terminal` (a visible "the run ended without a final
 * response" state), never null. `isStreaming` only decides between "still working" and
 * "ended with nothing" — it can never suppress a real result. */
export function selectTerminal(state: GraphVizState, isStreaming: boolean): Terminal {
  if (state.error) {
    return {
      kind: "error",
      message: state.error.message,
      retryable: state.error.retryable,
      friendly: FRIENDLY_ERROR_REASONS.has(state.error.reason ?? ""),
    };
  }

  if (state.interrupt) return terminalFromInterrupt(state.interrupt);

  if (state.completed) {
    const answer = str(state.completed.answer);
    if (state.completed.abstained || !answer) {
      return { kind: "abstain", reason: NO_SUPPORT };
    }
    return {
      kind: "answer",
      answer,
      citations: state.completed.citations,
      confidence: state.completed.confidence,
    };
  }

  return isStreaming ? { kind: "streaming" } : { kind: "no_terminal" };
}

export function resolveFanOutIdle(state: GraphVizState): GraphVizState {
  if (state.fanOut.phase === "idle") return state;
  return { ...state, fanOut: { ...state.fanOut, phase: "idle" } };
}

export function clearActiveEdge(state: GraphVizState): GraphVizState {
  if (!state.activeEdge) return state;
  return { ...state, activeEdge: null };
}

export function clearRetry(state: GraphVizState): GraphVizState {
  if (!state.retry?.active) return state;
  return { ...state, retry: { ...state.retry, active: false } };
}
