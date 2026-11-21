"""Draft an evaluation set from your own PDFs using the LLM.

Usage:
    python -m eval.make_testset --n 50

IMPORTANT: this produces a DRAFT. Open eval/testset.jsonl and fix or delete
questions that are vague, wrong, or answerable from several places. A reviewed
test set is what makes your reported numbers trustworthy.
"""
import argparse
import json
import random
import time
from pathlib import Path

import chromadb

from app import config
from app.llm import LLMError, generate
from app.utils import extract_json

PROMPT = """Below is a passage from a document. Write ONE specific, factual question that can be answered using ONLY this passage, plus the short correct answer.
- The question must make sense on its own: never say "the passage", "this text" or "the document".
- It must not be a yes/no question.
Reply with JSON only: {"question": "...", "answer": "..."}

Passage:
<<PASSAGE>>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/testset.jsonl")
    parser.add_argument(
        "--min-chars", type=int, default=300, help="skip chunks shorter than this"
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.collection_name(config.CHUNK_SIZE))
    data = collection.get(include=["documents", "metadatas"])

    pool = [
        (doc, meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
        if len(doc) >= args.min_chars  # skip tiny / low-information chunks
    ]
    random.Random(args.seed).shuffle(pool)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pages: set[tuple[str, int]] = set()
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for doc, meta in pool:
            if written >= args.n:
                break
            key = (meta["source"], int(meta["page"]))
            if key in seen_pages:  # at most one question per page
                continue
            try:
                reply = generate(PROMPT.replace("<<PASSAGE>>", doc))
            except LLMError as exc:
                print(f"Stopping: {exc}")
                break
            item = extract_json(reply)
            if not item or not item.get("question") or not item.get("answer"):
                continue
            seen_pages.add(key)
            record = {
                "question": item["question"].strip(),
                "gold_answer": str(item["answer"]).strip(),
                "source": meta["source"],
                "page": int(meta["page"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{written}/{args.n}] {record['question']}")
            time.sleep(1.5)  # be gentle with free-tier rate limits

    print(f"\nWrote {written} questions to {out_path}. Now REVIEW them by hand.")


if __name__ == "__main__":
    main()
