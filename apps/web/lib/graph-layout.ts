import type { NodeName } from "./types/graph-events.generated";

/** Fixed node order — matches api.graph.events.NodeName and drives both the desktop
 * SVG layout and the mobile vertical-timeline fallback. */
export const NODE_ORDER: NodeName[] = [
  "planner",
  "retriever",
  "reranker",
  "generator",
  "critic",
  "hitl_gate",
];

export const NODE_LABELS: Record<NodeName, string> = {
  planner: "planner",
  retriever: "retriever",
  reranker: "reranker",
  generator: "generator",
  critic: "critic",
  hitl_gate: "hitl_gate",
};

export const VIEWBOX = { width: 960, height: 420 };

export const NODE_RADIUS = 34;

/** Desktop node centers. Main flow arcs left-to-right across the top; hitl_gate hangs
 * below critic so its edge is short and vertical — geometrically distinct from the
 * long horizontal forward edges with no extra styling needed. */
export const NODE_POSITIONS: Record<NodeName, { x: number; y: number }> = {
  planner: { x: 100, y: 120 },
  retriever: { x: 280, y: 180 },
  reranker: { x: 460, y: 220 },
  generator: { x: 640, y: 180 },
  critic: { x: 820, y: 120 },
  hitl_gate: { x: 820, y: 300 },
};

export interface EdgeSpec {
  key: string;
  from: NodeName;
  to: NodeName;
  path: string;
  kind: "forward" | "retry";
}

function edgeKey(from: NodeName, to: NodeName): string {
  return `${from}->${to}`;
}

function straightPath(from: NodeName, to: NodeName): string {
  const a = NODE_POSITIONS[from];
  const b = NODE_POSITIONS[to];
  return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
}

/** The five main-flow edges — drawn in `rule` while idle, settle to `citation` once the
 * downstream node has started (see components/graph/GraphEdge.tsx). */
export const FORWARD_EDGES: EdgeSpec[] = [
  { key: edgeKey("planner", "retriever"), from: "planner", to: "retriever", path: straightPath("planner", "retriever"), kind: "forward" },
  { key: edgeKey("retriever", "reranker"), from: "retriever", to: "reranker", path: straightPath("retriever", "reranker"), kind: "forward" },
  { key: edgeKey("reranker", "generator"), from: "reranker", to: "generator", path: straightPath("reranker", "generator"), kind: "forward" },
  { key: edgeKey("generator", "critic"), from: "generator", to: "critic", path: straightPath("generator", "critic"), kind: "forward" },
  { key: edgeKey("critic", "hitl_gate"), from: "critic", to: "hitl_gate", path: straightPath("critic", "hitl_gate"), kind: "forward" },
];

/** The rare critic -> retriever retry loop: a large backward arc dipping well below the
 * main flow so it never crosses another node. Dashed + `active`-colored, only rendered
 * while a retry is in flight (see GraphVizState.retry). */
export const RETRY_EDGE: EdgeSpec = {
  key: edgeKey("critic", "retriever"),
  from: "critic",
  to: "retriever",
  path: (() => {
    const a = NODE_POSITIONS.critic;
    const b = NODE_POSITIONS.retriever;
    const dipY = 390;
    return `M ${a.x} ${a.y} C ${a.x} ${dipY}, ${b.x} ${dipY}, ${b.x} ${b.y}`;
  })(),
  kind: "retry",
};

export const ALL_EDGES: EdgeSpec[] = [...FORWARD_EDGES, RETRY_EDGE];

export function edgeFor(from: NodeName, to: NodeName): EdgeSpec | undefined {
  return ALL_EDGES.find((e) => e.from === from && e.to === to);
}
