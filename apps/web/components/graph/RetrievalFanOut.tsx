"use client";

import { useEffect } from "react";
import { motion } from "motion/react";

import { NODE_POSITIONS, NODE_RADIUS } from "@/lib/graph-layout";
import type { FanOutState } from "@/lib/graph-state";

const MAX_CHIPS = 12;

interface RetrievalFanOutProps {
  fanOut: FanOutState;
  reduceMotion: boolean;
  /** Called once the collapse has visually finished, so the parent can move
   * fanOut.phase back to "idle" — a component-driven transition, not part of the
   * event-sourced reducer. */
  onCollapsed: () => void;
}

/** Makes retrieval visible: on the retriever's node_completed, its candidate chunks
 * briefly fan out around the node; on the reranker's node_completed, they collapse
 * toward it, leaving a "N kept" count. Every number shown (candidate count, kept
 * count, the +N overflow label) comes straight off the real event payloads. */
export function RetrievalFanOut({ fanOut, reduceMotion, onCollapsed }: RetrievalFanOutProps) {
  const origin = NODE_POSITIONS.retriever;
  const target = NODE_POSITIONS.reranker;
  const shown = Math.min(fanOut.candidateCount, MAX_CHIPS);
  const overflow = fanOut.candidateCount - shown;

  useEffect(() => {
    // Chips stay mounted across fanned -> collapsing (only their target cx/cy/opacity
    // change), so there's no unmount to hook an exit callback onto — and with reduced
    // motion or zero candidates there's no animation to wait for at all. Settle
    // immediately in either case; otherwise the trailing chip's onAnimationComplete
    // (below) is what actually resolves the collapse.
    if (fanOut.phase === "collapsing" && (reduceMotion || shown === 0)) onCollapsed();
  }, [reduceMotion, fanOut.phase, shown, onCollapsed]);

  if (reduceMotion) {
    if (fanOut.phase === "idle") return null;
    return (
      <text x={target.x} y={target.y - NODE_RADIUS - 10} textAnchor="middle" className="fill-ink/60 font-mono text-[9px]">
        {fanOut.candidateCount} candidates → {fanOut.rerankedCount} kept
      </text>
    );
  }

  const chips = Array.from({ length: shown }, (_, i) => {
    const angle = (Math.PI * 2 * i) / Math.max(shown, 1) - Math.PI / 2;
    const spread = NODE_RADIUS + 22;
    return { cx: origin.x + Math.cos(angle) * spread, cy: origin.y + Math.sin(angle) * spread };
  });

  return (
    <g aria-hidden="true">
      {fanOut.phase !== "idle" &&
        chips.map((pos, i) => (
          <motion.circle
            key={i}
            r={3}
            fill="var(--color-citation)"
            initial={{ cx: origin.x, cy: origin.y, opacity: 0 }}
            animate={
              fanOut.phase === "fanned"
                ? { cx: pos.cx, cy: pos.cy, opacity: 0.8 }
                : { cx: target.x, cy: target.y, opacity: 0 }
            }
            transition={{ duration: 0.5, delay: (fanOut.phase === "fanned" ? i : shown - i) * 0.02, ease: "easeInOut" }}
            onAnimationComplete={() => {
              if (fanOut.phase === "collapsing" && i === 0) onCollapsed();
            }}
          />
        ))}
      {overflow > 0 && fanOut.phase === "fanned" && (
        <text x={origin.x} y={origin.y - NODE_RADIUS - 10} textAnchor="middle" className="fill-ink/50 font-mono text-[9px]">
          +{overflow} more
        </text>
      )}
      {fanOut.phase === "collapsing" && fanOut.rerankedCount > 0 && (
        <text x={target.x} y={target.y - NODE_RADIUS - 10} textAnchor="middle" className="fill-ink/60 font-mono text-[9px]">
          {fanOut.rerankedCount} kept
        </text>
      )}
    </g>
  );
}
