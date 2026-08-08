"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useReducedMotion } from "motion/react";

import { AnswerPanel } from "@/components/citations/AnswerPanel";
import { buildChunkIndex } from "@/lib/citations";
import { FORWARD_EDGES, NODE_ORDER, RETRY_EDGE, VIEWBOX } from "@/lib/graph-layout";
import { graphReducer, initialGraphVizState, resolveFanOutIdle, type GraphVizState } from "@/lib/graph-state";
import type { GraphEvent } from "@/lib/types/graph-events.generated";

import { AbstainPanel } from "./AbstainPanel";
import { EventLog } from "./EventLog";
import { ForwardGraphEdge, RetryGraphEdge } from "./GraphEdge";
import { GraphNode } from "./GraphNode";
import { RetrievalFanOut } from "./RetrievalFanOut";

type VizAction = { type: "event"; event: GraphEvent } | { type: "fanout-settled" };

function vizReducer(state: GraphVizState, action: VizAction): GraphVizState {
  return action.type === "event" ? graphReducer(state, action.event) : resolveFanOutIdle(state);
}

const RETRY_LABEL_MS = 2600;

export function AgentGraph({ events }: { events: GraphEvent[] }) {
  const [state, dispatch] = useReducer(vizReducer, undefined, initialGraphVizState);
  const appliedCount = useRef(0);
  const reduceMotion = useReducedMotion() ?? false;
  const [showLog, setShowLog] = useState(false);
  const [retryVisible, setRetryVisible] = useState(false);

  // `events` grows one item at a time (or resets to []) in the parent; replay only
  // what's new into the reducer, exactly once each. graph_started itself resets the
  // rest of GraphVizState (see graphReducer), so a fresh query self-cleans on replay.
  useEffect(() => {
    if (events.length < appliedCount.current) appliedCount.current = 0;
    for (let i = appliedCount.current; i < events.length; i++) {
      dispatch({ type: "event", event: events[i]! });
    }
    appliedCount.current = events.length;
  }, [events]);

  useEffect(() => {
    if (!state.retry?.active) return;
    setRetryVisible(true);
    const timer = setTimeout(() => setRetryVisible(false), RETRY_LABEL_MS);
    return () => clearTimeout(timer);
    // Retry is understated and rare — key off retry_count so a second retry in the same
    // run retriggers the timer even if the first one hasn't finished fading.
  }, [state.retry?.count, state.retry?.active]);

  const handleFanOutCollapsed = useCallback(() => dispatch({ type: "fanout-settled" }), []);

  const abstained = state.nodes.hitl_gate.status === "abstain";
  const rerankerPayload = state.nodes.reranker.payload;
  const chunksById = useMemo(
    () => buildChunkIndex(rerankerPayload?.reranked as unknown[] | undefined),
    [rerankerPayload],
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setShowLog((v) => !v)}
          className="font-mono text-[11px] text-ink/60 underline underline-offset-2"
        >
          {showLog ? "hide raw events" : "show raw events"}
        </button>
      </div>

      <svg
        viewBox={`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`}
        className="hidden h-auto w-full md:block"
        role="img"
        aria-label="Agent graph: planner, retriever, reranker, generator, critic, human review gate"
      >
        {FORWARD_EDGES.map((edge) => {
          const revealed =
            state.drawnEdges.has(edge.key) ||
            (state.activeEdge?.from === edge.from && state.activeEdge?.to === edge.to);
          return <ForwardGraphEdge key={edge.key} spec={edge} revealed={revealed} reduceMotion={reduceMotion} />;
        })}
        {retryVisible && <RetryGraphEdge spec={RETRY_EDGE} reduceMotion={reduceMotion} />}
        <RetrievalFanOut fanOut={state.fanOut} reduceMotion={reduceMotion} onCollapsed={handleFanOutCollapsed} />
        {NODE_ORDER.map((name) => (
          <GraphNode key={name} name={name} state={state.nodes[name]} reduceMotion={reduceMotion} variant="graph" />
        ))}
      </svg>

      <div className="border-rule divide-rule flex flex-col divide-y border md:hidden">
        {NODE_ORDER.map((name) => (
          <div key={name} className="px-3">
            <GraphNode name={name} state={state.nodes[name]} reduceMotion={reduceMotion} variant="row" />
            {name === "critic" && retryVisible && (
              <p className="text-active pb-2 font-mono text-[10px]">
                ↩ retrying retriever…{state.retry?.reason ? ` — ${state.retry.reason}` : ""}
              </p>
            )}
          </div>
        ))}
      </div>

      <AbstainPanel interrupt={abstained ? state.interrupt : null} reduceMotion={reduceMotion} />

      <AnswerPanel completed={state.completed} chunksById={chunksById} reduceMotion={reduceMotion} />

      {state.error && (
        <div className="border-ink text-ink border border-dashed px-4 py-3 font-mono text-xs">
          graph error{state.error.retryable ? " (retryable)" : ""}: {state.error.message}
        </div>
      )}

      {showLog && <EventLog events={events} />}
    </div>
  );
}
