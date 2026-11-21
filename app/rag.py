"""The RAG pipeline: rewrite follow-ups -> retrieve -> generate a grounded, cited answer."""
from app import config
from app.llm import generate
from app.retriever import Retriever

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You are DocuMind, an assistant that answers questions strictly from the provided document excerpts.

Rules:
1. Use ONLY the numbered context excerpts. Do not use outside knowledge.
2. Cite the excerpts you used with their numbers in square brackets, e.g. [1] or [2][3], right after the claim.
3. If the context does not contain the answer, reply exactly: {NOT_FOUND}
4. Be concise and precise. Copy numbers, dates and names exactly as written."""


def format_context(chunks: list[dict]) -> str:
    blocks = [
        f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


class DocuMind:
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ):
        self.retriever = Retriever(chunk_size, use_hybrid, use_rerank)

    def rewrite_question(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up like 'and its deadline?' into a standalone question."""
        if not history:
            return question
        convo = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
        )
        prompt = (
            "Given the conversation and a follow-up question, rewrite the follow-up "
            "as a standalone question that contains all the context needed to "
            "understand it. Reply with the question only.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
        )
        rewritten = generate(prompt).strip()
        return rewritten or question

    def ask(
        self, question: str, history: list[dict] | None = None, top_k: int = config.TOP_K
    ) -> dict:
        standalone = self.rewrite_question(question, history or [])
        chunks = self.retriever.retrieve(standalone, top_k=top_k)
        prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {standalone}\n\nAnswer:"
        answer = generate(prompt, system=SYSTEM_PROMPT).strip()
        return {"answer": answer, "sources": chunks, "standalone_question": standalone}
