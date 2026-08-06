import asyncio

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Per the model card: queries get this instruction prefix, passages/documents don't —
# the asymmetric usage core.ports.Embedder's docstring already anticipated ("adapters
# may use different code paths for queries vs. documents... asymmetric models like
# BGE/e5 apply different instructions/prefixes to each").
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    """CPU-only local embedder — see ADR 0002 (embeddings run locally, never a hosted
    API). BAAI/bge-small-en-v1.5: 33M params, 384-dim, the confirmed right-sized model
    for this project's free-tier/CPU-only constraint."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")

    async def embed_query(self, text: str) -> list[float]:
        vector = await asyncio.to_thread(
            self._model.encode, _QUERY_INSTRUCTION + text, normalize_embeddings=True
        )
        return vector.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return vectors.tolist()
