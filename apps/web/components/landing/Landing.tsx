"use client";

import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";

import { ExampleQuestions } from "./ExampleQuestions";
import { Footer } from "./Footer";
import { Hero } from "./Hero";
import { ScopeNote } from "./ScopeNote";

export const QUERY_PLACEHOLDER =
  "e.g. What are the transparency obligations for a limited-risk AI system?";

/** The first-impression surface: hero, curated example questions, a manual fallback
 * input, footer. One entrance sequence, gated by reduced-motion the same way every
 * other animated component in this codebase is (see AgentGraph.tsx/AbstainPanel.tsx) —
 * duration collapses to 0 rather than skipping the variant, so the final layout is
 * identical either way, just instant. */
export function Landing({ onAsk }: { onAsk: (question: string) => void }) {
  const [query, setQuery] = useState("");
  const reduceMotion = useReducedMotion() ?? false;

  const container = {
    hidden: {},
    show: { transition: { staggerChildren: reduceMotion ? 0 : 0.08 } },
  };
  const item = {
    hidden: { opacity: 0, y: 8 },
    show: { opacity: 1, y: 0, transition: { duration: reduceMotion ? 0 : 0.4, ease: "easeOut" as const } },
  };

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onAsk(query);
  }

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-10">
      <motion.div variants={item}>
        <Hero />
      </motion.div>

      <motion.div variants={item}>
        <ExampleQuestions onAsk={onAsk} />
      </motion.div>

      <motion.div variants={item}>
        <p className="font-mono text-[11px] tracking-wide text-ink/50 uppercase">or ask your own question</p>
        <form onSubmit={handleSubmit} className="mt-3 flex gap-2">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={QUERY_PLACEHOLDER}
            rows={3}
            className="border-rule flex-1 border p-2 font-serif"
          />
          <button
            type="submit"
            disabled={!query.trim()}
            className="border-rule bg-active shrink-0 self-start border px-4 py-2 font-mono text-sm text-paper disabled:opacity-40"
          >
            Ask
          </button>
        </form>
        <ScopeNote />
      </motion.div>

      <motion.div variants={item}>
        <Footer />
      </motion.div>
    </motion.div>
  );
}
