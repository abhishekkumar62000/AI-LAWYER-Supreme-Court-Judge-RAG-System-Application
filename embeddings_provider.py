import os
from functools import lru_cache
from typing import Iterable, List, Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Legacy Ollama model names are mapped to the new default so older manifests can rebuild.
LEGACY_OLLAMA_MODELS = {
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
}


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def resolve_embedding_model_name(model_name: Optional[str] = None) -> str:
    candidate = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    if not candidate or candidate in LEGACY_OLLAMA_MODELS:
        return DEFAULT_EMBEDDING_MODEL
    return candidate


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = resolve_embedding_model_name(model_name)

    @property
    def client(self) -> SentenceTransformer:
        return _load_sentence_transformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = [text if isinstance(text, str) else str(text) for text in texts]
        vectors = self.client.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.client.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()


def get_embedding_model(model_name: Optional[str] = None) -> Embeddings:
    return SentenceTransformerEmbeddings(model_name)
