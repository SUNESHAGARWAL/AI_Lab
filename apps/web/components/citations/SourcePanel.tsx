"use client";

import { useEffect, useRef } from "react";
import { motion } from "motion/react";

import { citationLabel, splitChunkText, type CitedChunk } from "@/lib/citations";

interface SourcePanelProps {
  chunkId: string;
  chunk: CitedChunk;
  onClose: () => void;
  reduceMotion: boolean;
}

/** The chip's counterpart — shares `layoutId={citation-${chunkId}}` with CitationChip so
 * opening it reads as the chip expanding into this panel. Desktop: fixed slide-in from
 * the right. Mobile (`<md`): bottom sheet, same layoutId — motion re-targets the FLIP to
 * wherever the element ends up in the new layout, no separate JS branch needed. */
export function SourcePanel({ chunkId, chunk, onClose, reduceMotion }: SourcePanelProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const label = citationLabel(chunk.metadata);
  const { header, body } = splitChunkText(chunk.text, label);

  useEffect(() => {
    closeButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <>
      <motion.button
        type="button"
        aria-label="Close source panel"
        onClick={onClose}
        className="bg-ink/30 fixed inset-0 z-40"
        initial={reduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: reduceMotion ? 0 : 0.25 }}
      />
      <motion.div
        layoutId={`citation-${chunkId}`}
        role="dialog"
        aria-modal="true"
        aria-label={`Source: ${label}`}
        transition={{ duration: reduceMotion ? 0 : 0.4, ease: "easeOut" }}
        className="bg-paper border-rule fixed inset-x-0 bottom-0 z-50 flex max-h-[75vh] flex-col rounded-t-lg border shadow-lg md:inset-x-auto md:inset-y-0 md:right-0 md:bottom-auto md:h-full md:max-h-none md:w-[min(420px,90vw)] md:rounded-t-none md:rounded-l-lg"
      >
        <div className="border-rule flex shrink-0 items-start justify-between gap-3 border-b px-5 py-4">
          <div>
            <p className="text-citation font-mono text-sm">{label}</p>
            {header && header !== label && <p className="text-ink/60 mt-0.5 font-mono text-[11px]">{header}</p>}
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-ink/60 hover:text-ink shrink-0 font-mono text-sm"
          >
            close
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">
          <p className="font-serif text-sm leading-relaxed whitespace-pre-wrap text-ink">{body}</p>
        </div>
      </motion.div>
    </>
  );
}
