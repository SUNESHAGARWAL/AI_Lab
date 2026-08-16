"use client";

import { useState } from "react";
import { AnimatePresence } from "motion/react";

import type { CitedChunk } from "@/lib/citations";

import { CitationChip } from "./CitationChip";
import { SourcePanel } from "./SourcePanel";

interface AnswerPanelProps {
  /** Already narrowed to a non-empty answer by lib/graph-state.ts's selectTerminal —
   * this component no longer decides whether there is anything worth rendering, which is
   * exactly how the blank-screen bug got in. It renders what it's handed. */
  answer: string;
  citations: string[];
  chunksById: Map<string, CitedChunk>;
  reduceMotion: boolean;
  /** Rendered above the answer for the flagged-for-review case; absent on the confident
   * path. Keeps one citation/SourcePanel implementation serving both terminals. */
  notice?: React.ReactNode;
}

interface Selection {
  chunkId: string;
  trigger: HTMLElement | null;
}

/** The confident-answer counterpart to AbstainPanel: the answer in the legal-instrument
 * reading style, citations as a trailing inline chip run (see lib/citations.ts's module
 * doc for why they can't be placed mid-sentence), each opening SourcePanel on click. */
export function AnswerPanel({ answer, citations, chunksById, reduceMotion, notice }: AnswerPanelProps) {
  const [selected, setSelected] = useState<Selection | null>(null);

  function handleOpen(chunkId: string, trigger: HTMLElement) {
    setSelected({ chunkId, trigger });
  }

  function handleClose() {
    selected?.trigger?.focus();
    setSelected(null);
  }

  const selectedChunk = selected ? chunksById.get(selected.chunkId) : undefined;

  return (
    <div className="border-rule border px-5 py-4">
      {notice}
      <p className="text-citation font-mono text-[11px] tracking-wide uppercase">answer</p>
      <p className="mt-2 max-w-prose font-serif text-base leading-relaxed text-ink">
        {answer}
        {citations.length > 0 && (
          <span className="ml-2 inline-flex flex-wrap items-center gap-1 align-middle">
            {citations.map((chunkId) => (
              <CitationChip
                key={chunkId}
                chunkId={chunkId}
                chunk={chunksById.get(chunkId)}
                isOpen={selected?.chunkId === chunkId}
                onOpen={(id, trigger) => handleOpen(id, trigger)}
              />
            ))}
          </span>
        )}
      </p>

      <AnimatePresence>
        {selected && selectedChunk && (
          <SourcePanel chunkId={selected.chunkId} chunk={selectedChunk} onClose={handleClose} reduceMotion={reduceMotion} />
        )}
      </AnimatePresence>
    </div>
  );
}
