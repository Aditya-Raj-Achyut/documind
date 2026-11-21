"""FastAPI service.  Run:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm import LLMError
from app.rag import DocuMind

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = DocuMind()  # loads index + models once at startup
    yield
    state.clear()


app = FastAPI(title="DocuMind API", version="1.0.0", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = []


class Source(BaseModel):
    source: str
    page: int
    text: str
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    standalone_question: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(state["rag"].retriever)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # plain `def` (not async): FastAPI runs it in a worker thread, so the
    # blocking model + HTTP calls do not freeze the server.
    try:
        return state["rag"].ask(req.question, [m.model_dump() for m in req.history])
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
