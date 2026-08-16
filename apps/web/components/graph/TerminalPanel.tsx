"use client";

import { AnswerPanel } from "@/components/citations/AnswerPanel";
import type { CitedChunk } from "@/lib/citations";
import type { Terminal } from "@/lib/graph-state";

import { AbstainPanel } from "./AbstainPanel";

/** How long the query has been waiting with nothing back yet. Railway's free tier
 * cold-starts, and a cold start can run into the minutes — the visitor has to be able to
 * tell "waking up" from "dead", or they re-click and fire duplicate streams. */
export type WaitPhase = "thinking" | "waking" | "still_waking";

const WAIT_COPY: Record<WaitPhase, string> = {
  thinking: "Working through the graph…",
  waking: "Waking the backend up — free-tier cold starts take a moment.",
  still_waking: "Still waking up — this can take up to a minute on the free tier.",
};

interface TerminalPanelProps {
  terminal: Terminal;
  chunksById: Map<string, CitedChunk>;
  reduceMotion: boolean;
  waitPhase: WaitPhase;
  onBackToExamples?: () => void;
  /** Flips the "show raw events" toggle in the parent — the no_terminal fallback's one
   * useful action, since the raw log is where an unexpected event shape is visible. */
  onShowRawEvents?: () => void;
}

function ReviewNotice({ confidence }: { confidence: number | null }) {
  return (
    <div className="border-citation bg-citation/5 mb-4 border-l-4 px-4 py-3">
      <p className="text-citation font-mono text-[11px] tracking-wide uppercase">
        flagged for review
        {confidence !== null && ` · confidence ${confidence.toFixed(2).replace(/^0/, "")}`}
      </p>
      <p className="mt-1 max-w-prose font-serif text-sm leading-relaxed text-ink/80">
        Below this system&apos;s confidence threshold — verify against the cited text before
        relying on it.
      </p>
    </div>
  );
}

/**
 * The single renderer for however a query ends. Exhaustive over `Terminal.kind`, and the
 * `default` branch renders a visible fallback rather than returning null — together with
 * `selectTerminal` being a total function, that is the guarantee that a query can never
 * leave the screen blank.
 *
 * Every terminal state used to be an independent conditional in AgentGraph, which meant
 * they could all decline at once: that is exactly what happened for a below-threshold
 * answer, whose text arrived in the `graph_interrupted` payload and was never rendered.
 * Do not reintroduce a branch here that can return null.
 */
export function TerminalPanel({
  terminal,
  chunksById,
  reduceMotion,
  waitPhase,
  onBackToExamples,
  onShowRawEvents,
}: TerminalPanelProps) {
  switch (terminal.kind) {
    case "answer":
      return (
        <AnswerPanel
          answer={terminal.answer}
          citations={terminal.citations}
          chunksById={chunksById}
          reduceMotion={reduceMotion}
        />
      );

    case "review":
      return (
        <AnswerPanel
          answer={terminal.answer}
          citations={terminal.citations}
          chunksById={chunksById}
          reduceMotion={reduceMotion}
          notice={<ReviewNotice confidence={terminal.confidence} />}
        />
      );

    case "abstain":
      return (
        <AbstainPanel
          variant="abstain"
          reason={terminal.reason}
          reduceMotion={reduceMotion}
          onBackToExamples={onBackToExamples}
        />
      );

    case "out_of_scope":
      return (
        <AbstainPanel
          variant="out_of_scope"
          reason={terminal.reason}
          reduceMotion={reduceMotion}
          onBackToExamples={onBackToExamples}
        />
      );

    case "error":
      return terminal.friendly ? (
        <div className="border-citation bg-citation/5 border-l-4 px-5 py-4" role="status">
          <p className="font-serif text-sm leading-relaxed text-ink">{terminal.message}</p>
          {onBackToExamples && (
            <button
              type="button"
              onClick={onBackToExamples}
              className="text-citation mt-2 font-mono text-xs underline underline-offset-2"
            >
              back to the example questions
            </button>
          )}
        </div>
      ) : (
        <div className="border-active bg-active/5 border-l-4 px-5 py-4" role="alert">
          <p className="text-active font-mono text-[11px] tracking-wide uppercase">
            error{terminal.retryable ? " · retryable" : ""}
          </p>
          <p className="mt-1 max-w-prose font-serif text-sm leading-relaxed text-ink">
            {terminal.message}
          </p>
          {onBackToExamples && (
            <button
              type="button"
              onClick={onBackToExamples}
              className="text-citation mt-3 font-mono text-xs underline underline-offset-2"
            >
              back to the example questions
            </button>
          )}
        </div>
      );

    case "streaming":
      return (
        <div className="border-rule border border-dashed px-5 py-4" role="status" aria-live="polite">
          <p className="font-mono text-[11px] tracking-wide text-ink/50 uppercase">working</p>
          <p className="mt-1 font-serif text-sm leading-relaxed text-ink/70">{WAIT_COPY[waitPhase]}</p>
        </div>
      );

    // no_terminal, plus anything a future backend event shape could produce. A blank
    // screen is never an acceptable outcome, so this says so out loud instead.
    default:
      return (
        <div className="border-rule border border-dashed px-5 py-4" role="status">
          <p className="font-mono text-[11px] tracking-wide text-ink/50 uppercase">no response</p>
          <p className="mt-1 max-w-prose font-serif text-sm leading-relaxed text-ink/70">
            The run ended without a final response. That&apos;s a bug, not an abstention —
            the raw event log below shows what actually arrived.
          </p>
          <div className="mt-3 flex flex-wrap gap-4">
            {onShowRawEvents && (
              <button
                type="button"
                onClick={onShowRawEvents}
                className="text-citation font-mono text-xs underline underline-offset-2"
              >
                show raw events
              </button>
            )}
            {onBackToExamples && (
              <button
                type="button"
                onClick={onBackToExamples}
                className="text-citation font-mono text-xs underline underline-offset-2"
              >
                back to the example questions
              </button>
            )}
          </div>
        </div>
      );
  }
}
