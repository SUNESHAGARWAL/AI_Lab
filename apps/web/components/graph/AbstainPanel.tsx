"use client";

import { motion } from "motion/react";

/** The one-line restatement of what this system is for. Shown on every declined query so
 * a visitor who tested it like a general chatbot learns the scope from the response
 * itself, not just from the note above the input. Deliberately says nothing factual — it
 * describes the tool, it never answers outside the corpus. */
const CAPABILITY_LINE =
  "This assistant answers EU AI Act and GDPR compliance questions, with citations to the regulation text.";

interface AbstainPanelProps {
  /** Why the system declined. `variant` distinguishes a genuine abstention (retrieval
   * ran, sources didn't support an answer) from an out-of-scope query (never worth
   * retrieving for) — same calm panel, different framing. */
  variant: "abstain" | "out_of_scope";
  reason: string | null;
  reduceMotion: boolean;
  onBackToExamples?: () => void;
}

/** The centerpiece: a deliberate, calm "I can't answer this faithfully" state — styled
 * to read as honesty, not failure. No error iconography, generous whitespace, the
 * `abstain` token doing the only color work. Out-of-scope adds the capability line and a
 * route forward so a dead end always points somewhere. */
export function AbstainPanel({ variant, reason, reduceMotion, onBackToExamples }: AbstainPanelProps) {
  const outOfScope = variant === "out_of_scope";

  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.4, ease: "easeOut" }}
      className="border-abstain bg-abstain/5 border-l-4 px-5 py-4"
      role="status"
    >
      <p className="text-abstain font-mono text-[11px] tracking-wide uppercase">
        {outOfScope ? "outside scope" : "abstained"}
      </p>
      <h2 className="mt-1 font-serif text-lg text-ink">
        {outOfScope
          ? "That's outside what this system can answer."
          : "This system can't answer that faithfully."}
      </h2>
      <p className="mt-2 max-w-prose font-serif text-sm leading-relaxed text-ink/80">
        {reason ?? (outOfScope ? CAPABILITY_LINE : "The available sources didn't support a confident answer.")}
      </p>
      {outOfScope && reason && (
        <p className="mt-2 max-w-prose font-serif text-sm leading-relaxed text-ink/80">{CAPABILITY_LINE}</p>
      )}
      {onBackToExamples && (
        <button
          type="button"
          onClick={onBackToExamples}
          className="text-citation mt-3 font-mono text-xs underline underline-offset-2"
        >
          try one of the example questions
        </button>
      )}
    </motion.div>
  );
}
