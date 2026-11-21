"""Central configuration. Override anything with environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.getenv("PDF_DIR", BASE_DIR / "data" / "pdfs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))

# --- Retrieval models -------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models work best when queries (not documents) get this instruction prefix.
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chunking / retrieval ---------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # characters (~200 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # characters
TOP_K = int(os.getenv("TOP_K", "5"))  # chunks sent to the LLM
CANDIDATES = int(os.getenv("CANDIDATES", "20"))  # chunks fetched before reranking

# --- LLM --------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def collection_name(chunk_size: int) -> str:
    """One Chroma collection per chunk size, so we can compare them in eval."""
    return f"documind_c{chunk_size}"
