"use client";

import { motion } from "motion/react";

import { citationLabel, type CitedChunk } from "@/lib/citations";

interface CitationChipProps {
  chunkId: string;
  chunk: CitedChunk | undefined;
  isOpen: boolean;
  onOpen: (chunkId: string, trigger: HTMLElement) => void;
}

const CHIP_CLASS =
  "border-citation/40 text-citation inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-xs leading-none";

/** One citation, rendered inline after the answer text it supports. Shares `layoutId`
 * with SourcePanel.tsx so opening it reads as the chip expanding into the panel, not a
 * new element appearing — see motion's shared layout animations. A chunk id the
 * reranker payload doesn't have (shouldn't happen — see lib/citations.ts) degrades to
 * plain, non-interactive text rather than a dead button. */
export function CitationChip({ chunkId, chunk, isOpen, onOpen }: CitationChipProps) {
  if (!chunk) {
    return <span className={`${CHIP_CLASS} opacity-50`}>{chunkId}</span>;
  }

  const label = citationLabel(chunk.metadata);

  return (
    <motion.button
      type="button"
      layoutId={`citation-${chunkId}`}
      onClick={(e) => onOpen(chunkId, e.currentTarget)}
      aria-label={`Open source: ${label}`}
      aria-expanded={isOpen}
      // Stays mounted (so the text flow doesn't reflow) but visually yields to
      // SourcePanel while it's open — both share layoutId, and motion's projection
      // animates the FLIP between wherever this chip sits and wherever the panel ends
      // up, per motion.dev's shared-layout pattern.
      className={`${CHIP_CLASS} hover:border-citation cursor-pointer transition-colors ${isOpen ? "pointer-events-none opacity-0" : ""}`}
    >
      {label}
    </motion.button>
  );
}
