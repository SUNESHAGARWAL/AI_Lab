import asyncio

from core.models import Query, ScoredChunk
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_MAX_LENGTH = 512  # BGE reranker family's standard max sequence length


class CrossEncoderReranker:
    """CPU-only local cross-encoder reranker — see ADR 0002 (models run locally,
    never a hosted API). BAAI/bge-reranker-base: ~0.3B params, the smaller of the
    two current bge-reranker options and sufficient for this project's English-only
    corpus — bge-reranker-v2-m3's extra multilingual capacity (100+ languages) goes
    unused here, for roughly double the CPU cost."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self._model = CrossEncoder(model_name, device="cpu", max_length=max_length)

    async def rerank(self, query: Query, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query.text, sc.chunk.text) for sc in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=float(score))
            for sc, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored
