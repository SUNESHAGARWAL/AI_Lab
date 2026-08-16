/** Always-visible statement of what this tool is and isn't, sat directly under the input
 * on both the landing surface and the post-query view. It exists so a visitor learns the
 * boundary *before* spending a query testing it like a general chatbot — the same message
 * AbstainPanel repeats when one gets declined anyway. Marginal-note styling, inside the
 * locked token set: no new colors, no new fonts. */
export function ScopeNote() {
  return (
    <p className="border-rule mt-3 max-w-prose border-l pl-3 font-mono text-[11px] leading-relaxed text-ink/55">
      <span className="text-citation tracking-wide uppercase">Scope</span> — EU AI Act and
      GDPR compliance questions, answered with citations to the regulation text. Anything
      outside that, it declines rather than guesses.
    </p>
  );
}
