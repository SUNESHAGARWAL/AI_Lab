"use client";

import { AnimatePresence, motion } from "motion/react";

interface AbstainPanelProps {
  /** GraphInterruptedEvent.interrupt, already narrowed by the caller to a case where
   * the graph actually abstained — either hitl_gate's "review" framing with
   * abstained=true, or its "out_of_scope" framing (planner routed straight here). */
  interrupt: Record<string, unknown> | null;
  reduceMotion: boolean;
}

function reasonFor(interrupt: Record<string, unknown>): string {
  if (interrupt.type === "out_of_scope") {
    return typeof interrupt.reason === "string" ? interrupt.reason : "This question is outside what this system can help with.";
  }
  return typeof interrupt.abstain_reason === "string"
    ? interrupt.abstain_reason
    : "The available sources didn't support a confident answer.";
}

/** The centerpiece: a deliberate, calm "I can't answer this faithfully" state — styled
 * to read as honesty, not failure. No error iconography, generous whitespace, the
 * `abstain` token doing the only color work. */
export function AbstainPanel({ interrupt, reduceMotion }: AbstainPanelProps) {
  const show = interrupt !== null;
  return (
    <AnimatePresence>
      {show && interrupt && (
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
          transition={{ duration: reduceMotion ? 0 : 0.4, ease: "easeOut" }}
          className="border-l-4 border-abstain bg-abstain/5 px-5 py-4"
          role="status"
        >
          <p className="font-mono text-[11px] uppercase tracking-wide text-abstain">abstained</p>
          <h2 className="mt-1 font-serif text-lg text-ink">This system can&apos;t answer that faithfully.</h2>
          <p className="mt-2 max-w-prose font-serif text-sm leading-relaxed text-ink/80">{reasonFor(interrupt)}</p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
