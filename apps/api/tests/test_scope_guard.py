"""The scope guard's whole value is that it never swallows a real compliance question.
The negative cases below are the important half of this file — each one is a real
question that a naive substring match would have eaten."""

import pytest
from api.graph.scope_guard import (
    GREETING_REASON,
    META_REASON,
    out_of_scope_interrupt,
    scope_guard_reason,
)


@pytest.mark.parametrize(
    "query",
    ["hi", "Hi!", "  HELLO  ", "hey there", "Good morning", "thanks", "Thank you!", "test", "ping"],
)
def test_greetings_are_guarded(query: str) -> None:
    assert scope_guard_reason(query) == GREETING_REASON


@pytest.mark.parametrize(
    "query",
    ["who are you", "What can you do?", "what is this", "Help", "how does this work?"],
)
def test_meta_questions_are_guarded(query: str) -> None:
    assert scope_guard_reason(query) == META_REASON


@pytest.mark.parametrize(
    "query",
    [
        # The question that motivated this whole change.
        "what does the EU AI Act mention on personal information handling",
        # Starts with the guarded token "hi" — must not match.
        "high-risk AI systems obligations",
        "hi-risk classification under Annex III",
        # Starts with the guarded phrase "who are you" territory — must not match.
        "who are the data controllers under GDPR",
        "what are the transparency obligations for limited-risk AI systems",
        "what is this article's scope under GDPR Art. 35",
        "help me understand DPIA thresholds",
        # Every curated example question (apps/web/components/landing/ExampleQuestions.tsx).
        "What are the requirements for high-risk AI systems under the EU AI Act?",
        "What is a data protection impact assessment under GDPR?",
        "How does the GDPR define personal data?",
        "How do the fines for AI Act deployer violations compare to GDPR fines?",
    ],
)
def test_real_questions_pass_through(query: str) -> None:
    assert scope_guard_reason(query) is None


def test_long_input_always_passes_through() -> None:
    """The length ceiling is an independent backstop on the whole-string match."""
    assert scope_guard_reason("hi " * 30) is None


def test_empty_input_passes_through() -> None:
    assert scope_guard_reason("   ") is None


def test_interrupt_matches_the_hitl_gate_out_of_scope_shape() -> None:
    """The frontend branches on interrupt["type"] — this must stay identical to what
    api.graph.nodes.make_hitl_gate_node emits, or the guard renders as nothing."""
    payload = out_of_scope_interrupt(GREETING_REASON)
    assert payload["type"] == "out_of_scope"
    assert set(payload) == {"type", "message", "reason", "suggestion"}
    assert payload["reason"] == GREETING_REASON
