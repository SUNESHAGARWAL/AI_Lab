"use client";

import { motion } from "motion/react";

import type { EdgeSpec } from "@/lib/graph-layout";

interface ForwardEdgeProps {
  spec: EdgeSpec;
  revealed: boolean;
  reduceMotion: boolean;
}

/** A main-flow edge. The faint `rule`-colored line is always visible (it's what makes
 * this read as a graph rather than a blank canvas); the `citation`-colored overlay
 * draws in (pathLength 0->1) the moment control actually passes along it. */
export function ForwardGraphEdge({ spec, revealed, reduceMotion }: ForwardEdgeProps) {
  return (
    <g>
      <path d={spec.path} stroke="var(--color-rule)" strokeWidth={1.5} fill="none" />
      <motion.path
        d={spec.path}
        stroke="var(--color-citation)"
        strokeWidth={2}
        fill="none"
        initial={false}
        animate={{ pathLength: revealed ? 1 : 0, opacity: revealed ? 1 : 0 }}
        transition={{ duration: reduceMotion ? 0 : 0.6, ease: "easeInOut" }}
      />
    </g>
  );
}

interface RetryEdgeProps {
  spec: EdgeSpec;
  reduceMotion: boolean;
}

/** The critic -> retriever retry loop. Only ever mounted (via AnimatePresence in
 * AgentGraph) while a retry is actually in flight — it's rare in practice, so the
 * animation stays understated: dashed, `active`-colored, drawn once. */
export function RetryGraphEdge({ spec, reduceMotion }: RetryEdgeProps) {
  return (
    <motion.path
      d={spec.path}
      stroke="var(--color-active)"
      strokeWidth={2}
      strokeDasharray="6 4"
      fill="none"
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.9, ease: "easeInOut" }}
    />
  );
}
