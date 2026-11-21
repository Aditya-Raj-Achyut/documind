"""Small, dependency-free helpers (easy to unit test)."""
import json
import re


def clean_text(text: str) -> str:
    """Normalise raw PDF text: fix hyphenated line breaks, collapse whitespace."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)  # "inter-\nnational" -> "international"
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Sentence-aware chunking.

    Sentences are packed into chunks of at most `size` characters. Each new chunk
    starts with the last ~`overlap` characters of the previous one so that an
    answer sitting on a chunk boundary is not lost.
    """
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")

    # 1. split into sentences; hard-split any sentence that is longer than `size`
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > size:
            pieces.append(sentence[:size])
            sentence = sentence[size - overlap :]
        if sentence:
            pieces.append(sentence)

    # 2. pack sentences into chunks
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= size:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:  # start the overlap on a word boundary
                tail = tail[tail.find(" ") + 1 :]
            if len(tail) + 1 + len(piece) > size:
                tail = ""
            current = f"{tail} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25."""
    return re.findall(r"\w+", text.lower())


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merge several ranked lists of ids into one.

    An item scores 1/(k + rank) in every list it appears in; items found by
    several retrievers float to the top. No score normalisation needed.
    """
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def extract_json(text: str):
    """Pull the first JSON object out of an LLM reply (or return None)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
