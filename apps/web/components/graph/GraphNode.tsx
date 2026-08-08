"use client";

import { motion } from "motion/react";

import { NODE_LABELS, NODE_POSITIONS, NODE_RADIUS } from "@/lib/graph-layout";
import type { NodeStatus, NodeVizState } from "@/lib/graph-state";
import type { NodeName } from "@/lib/types/graph-events.generated";

interface StatusStyle {
  stroke: string;
  fillOpacity: number;
  dash?: string;
  label: string;
}

const STATUS_STYLE: Record<NodeStatus, StatusStyle> = {
  idle: { stroke: "var(--color-rule)", fillOpacity: 0, label: "idle" },
  active: { stroke: "var(--color-active)", fillOpacity: 0.85, label: "working" },
  completed: { stroke: "var(--color-citation)", fillOpacity: 0, label: "done" },
  pending_review: { stroke: "var(--color-citation)", fillOpacity: 0.12, label: "awaiting human review" },
  abstain: { stroke: "var(--color-abstain)", fillOpacity: 0.18, dash: "4 3", label: "abstained" },
  error: { stroke: "var(--color-ink)", fillOpacity: 0, dash: "3 3", label: "error" },
};

function formatCost(usd: number): string {
  return `$${usd.toFixed(6)}`;
}

function badgeText(state: NodeVizState): string | null {
  if (state.status !== "completed" && state.status !== "pending_review") return null;
  if (state.latencyMs === null) return null;
  const latency = `${Math.round(state.latencyMs)}ms`;
  return state.estimatedCostUsd !== null ? `${latency} · ${formatCost(state.estimatedCostUsd)}` : latency;
}

function NodeCircle({ status, radius, reduceMotion }: { status: NodeStatus; radius: number; reduceMotion: boolean }) {
  const style = STATUS_STYLE[status];
  const isActive = status === "active";
  return (
    <motion.circle
      r={radius}
      stroke={style.stroke}
      strokeWidth={2}
      strokeDasharray={style.dash}
      fill={isActive ? "var(--color-active)" : status === "abstain" ? "var(--color-abstain)" : status === "pending_review" ? "var(--color-citation)" : "transparent"}
      fillOpacity={style.fillOpacity}
      animate={
        isActive && !reduceMotion
          ? { scale: [1, 1.06, 1], opacity: [0.75, 1, 0.75] }
          : { scale: 1, opacity: 1 }
      }
      transition={
        isActive && !reduceMotion
          ? { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
          : { duration: reduceMotion ? 0 : 0.3 }
      }
      style={{ transformOrigin: "center", transformBox: "fill-box" }}
      className="transition-colors duration-300"
    />
  );
}

interface GraphNodeProps {
  name: NodeName;
  state: NodeVizState;
  reduceMotion: boolean;
  variant?: "graph" | "row";
}

export function GraphNode({ name, state, reduceMotion, variant = "graph" }: GraphNodeProps) {
  const style = STATUS_STYLE[state.status];
  const badge = badgeText(state);
  const ariaLabel = `${NODE_LABELS[name]}: ${style.label}${badge ? `, ${badge}` : ""}`;

  if (variant === "row") {
    return (
      <div className="flex items-center gap-3 py-2" role="img" aria-label={ariaLabel} tabIndex={0}>
        <svg width={36} height={36} viewBox="-18 -18 36 36" aria-hidden="true">
          <NodeCircle status={state.status} radius={14} reduceMotion={reduceMotion} />
        </svg>
        <div className="flex flex-col">
          <span className="font-mono text-xs text-ink">{NODE_LABELS[name]}</span>
          {badge && <span className="font-mono text-[10px] text-ink/60">{badge}</span>}
          {state.status === "abstain" && <span className="font-mono text-[10px] text-abstain">abstained</span>}
        </div>
      </div>
    );
  }

  const { x, y } = NODE_POSITIONS[name];
  return (
    <g transform={`translate(${x}, ${y})`} role="img" aria-label={ariaLabel} tabIndex={0} focusable="true">
      <NodeCircle status={state.status} radius={NODE_RADIUS} reduceMotion={reduceMotion} />
      <text textAnchor="middle" dy={NODE_RADIUS + 18} className="fill-ink font-mono text-[11px]">
        {NODE_LABELS[name]}
      </text>
      {badge && (
        <text textAnchor="middle" dy={NODE_RADIUS + 32} className="fill-ink/60 font-mono text-[9px]">
          {badge}
        </text>
      )}
      {state.status === "abstain" && (
        <text textAnchor="middle" dy={NODE_RADIUS + 32} className="fill-abstain font-mono text-[9px]">
          abstained
        </text>
      )}
      {state.status === "error" && (
        <text textAnchor="middle" dy={4} className="fill-ink font-mono text-sm">
          ×
        </text>
      )}
    </g>
  );
}
