"""Retrieval: vector search + BM25 keyword search, fused with RRF, then reranked.

    query ──┬─> vector search (ChromaDB) ──┐
            └─> BM25 keyword search ───────┴─> RRF fusion ─> cross-encoder rerank ─> top-k
"""
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.models import embed_query, get_reranker
from app.utils import rrf, tokenize


class Retriever:
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ):
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            self.collection = client.get_collection(config.collection_name(chunk_size))
        except Exception as exc:
            raise RuntimeError(
                f"Index for chunk size {chunk_size} not found. "
                "Run `python -m app.ingest` first."
            ) from exc

        data = self.collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.id_to_idx = {cid: n for n, cid in enumerate(self.ids)}

        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if use_hybrid else None
        self.reranker = get_reranker() if use_rerank else None

    def __len__(self) -> int:
        return len(self.ids)

    def _vector_search(self, query: str, n: int) -> list[int]:
        result = self.collection.query(
            query_embeddings=[embed_query(query)],
            n_results=min(n, len(self.ids)),
        )
        return [self.id_to_idx[cid] for cid in result["ids"][0]]

    def _bm25_search(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:n]
        return [int(i) for i in order if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        candidates: int = config.CANDIDATES,
    ) -> list[dict]:
        ranked = self._vector_search(query, candidates)
        if self.use_hybrid:
            ranked = rrf([ranked, self._bm25_search(query, candidates)])[:candidates]

        scored: list[tuple[int, float | None]] = [(i, None) for i in ranked]
        if self.use_rerank and ranked:
            pairs = [(query, self.docs[i]) for i in ranked]
            scores = self.reranker.predict(pairs)
            order = np.argsort(scores)[::-1]
            scored = [(ranked[o], float(scores[o])) for o in order]

        results = []
        for idx, score in scored[:top_k]:
            results.append(
                {
                    "text": self.docs[idx],
                    "source": self.metas[idx]["source"],
                    "page": int(self.metas[idx]["page"]),
                    "score": score,
                }
            )
        return results
