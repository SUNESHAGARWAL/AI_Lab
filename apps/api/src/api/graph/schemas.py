from pydantic import BaseModel, ConfigDict, Field


class GeneratedAnswer(BaseModel):
    """Structured output contract for the generator node — see the system prompt in
    api.graph.nodes for how the model is instructed to fill this in. `citations` must
    reference only chunk ids present in the chunks given to the model; the generator
    node defensively re-validates this after parsing (a model can still hallucinate an
    id despite instructions) and forces `abstained=True` if it doesn't hold."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    abstain_reason: str | None = None


class CriticVerdict(BaseModel):
    """Structured output contract for the critic's lightweight faithfulness check —
    see the critic system prompt in api.graph.nodes. A cheap fast-tier self-check, not
    a second full generation pass: only the cited chunks (not the whole reranked set)
    are shown to the judge."""

    model_config = ConfigDict(frozen=True)

    faithful: bool
    reason: str | None = None
