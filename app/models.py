"""Lazy, cached loading of the embedding and reranker models."""
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def embed_query(query: str) -> list[float]:
    vec = get_embedder().encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
    return vec.tolist()
