"""Pre-graph short-circuit for greetings and meta questions.

A visitor's first input is often "hi" or "what can you do". Those are out of scope, and
the planner already classifies them correctly — but only after a gateway call. On a demo
with a per-day token budget and a 5/hour per-visitor live-query limit, spending either on
"hi" is waste.

This module answers that class deterministically: no LLM, no retrieval, no graph build,
no rate-limit consumption. It is *not* a fallback answerer and must never grow into one —
it emits the same out-of-scope framing the graph would, and nothing factual. Anything it
is not certain about goes to the planner, which is the component qualified to classify.

Matching is whole-string on a normalized form, never substring: "hi" must not swallow
"high-risk AI systems", and "who are you" must not swallow "who are the data controllers
under GDPR". The length ceiling is a second, independent guard on the same property.
"""

import re

# A real question is longer than any greeting. Anything above this goes to the planner
# regardless of what it starts with — a cheap, phrasing-independent backstop on top of
# the whole-string match.
MAX_GUARDED_LENGTH = 40

_NORMALIZE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")

_GREETINGS = frozenset(
    {
        "hi",
        "hi there",
        "hello",
        "hello there",
        "hey",
        "hey there",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "greetings",
        "thanks",
        "thank you",
        "ty",
        "ok",
        "okay",
        "test",
        "testing",
        "ping",
    }
)

_META = frozenset(
    {
        "who are you",
        "what are you",
        "what is this",
        "what is this thing",
        "whats this",
        "what do you do",
        "what can you do",
        "what can i ask",
        "what can i ask you",
        "what can you help with",
        "help",
        "help me",
        "how does this work",
    }
)

GREETING_REASON = (
    "That looks like a greeting rather than a compliance question — there's no "
    "regulation text to cite for it."
)

META_REASON = (
    "That's a question about this system rather than about the regulations, so there's "
    "no source text to ground an answer in."
)


def _normalize(query: str) -> str:
    return _WHITESPACE.sub(" ", _NORMALIZE.sub(" ", query.strip().lower())).strip()


def scope_guard_reason(query: str) -> str | None:
    """The out-of-scope reason for a greeting or meta question, else ``None``.

    ``None`` means "not sure" as well as "definitely a real question" — the caller must
    treat it as "run the graph". Failing open costs one cheap planner call; failing
    closed would silently refuse a legitimate compliance question, which is far worse.
    """
    if len(query) > MAX_GUARDED_LENGTH:
        return None

    normalized = _normalize(query)
    if not normalized:
        return None
    if normalized in _GREETINGS:
        return GREETING_REASON
    if normalized in _META:
        return META_REASON
    return None


def out_of_scope_interrupt(reason: str) -> dict[str, str]:
    """The same ``out_of_scope`` interrupt shape hitl_gate emits (see
    api.graph.nodes.make_hitl_gate_node), so the frontend needs no new event type and the
    generated TypeScript stays valid."""
    return {
        "type": "out_of_scope",
        "message": "This question is outside what this system can help with.",
        "reason": reason,
        "suggestion": (
            "This assistant answers EU AI Act and GDPR compliance questions, with "
            "citations to the regulation text. Try one of the example questions."
        ),
    }
