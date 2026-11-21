"""Measure retrieval quality (and optionally answer quality) and write eval/results.md.

Usage:
    python -m eval.evaluate                              # retrieval only (free, fast)
    python -m eval.evaluate --chunk-sizes 500 800 1200   # also compare chunk sizes
    python -m eval.evaluate --judge --judge-n 20         # + LLM-judged answer quality

Metrics
    Hit@k : fraction of questions where a retrieved chunk comes from the correct
            source AND page (page-level, so different chunk sizes are comparable).
    MRR   : mean of 1/rank of the first correct chunk (rewards ranking it higher).
"""
import argparse
import json
from datetime import date
from pathlib import Path

from app import config
from app.ingest import build_index, collection_exists
from app.llm import generate
from app.rag import DocuMind, format_context
from app.retriever import Retriever
from app.utils import extract_json

CONFIGS = [
    ("Vector only", dict(use_hybrid=False, use_rerank=False)),
    ("Hybrid (BM25 + vector)", dict(use_hybrid=True, use_rerank=False)),
    ("Hybrid + reranker", dict(use_hybrid=True, use_rerank=True)),
]

JUDGE_PROMPT = """Question: <<Q>>
Reference answer: <<GOLD>>

Context given to the system:
<<CONTEXT>>

System answer: <<ANSWER>>

Return JSON only: {"faithful": true/false, "correct": true/false}
- faithful: every claim in the system answer is supported by the context above.
- correct: the system answer agrees in meaning with the reference answer."""


def load_testset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(retriever: Retriever, testset: list[dict], k: int):
    hits, reciprocal = 0, 0.0
    for item in testset:
        results = retriever.retrieve(item["question"], top_k=k)
        rank = next(
            (
                i
                for i, r in enumerate(results, start=1)
                if r["source"] == item["source"] and r["page"] == item["page"]
            ),
            None,
        )
        if rank:
            hits += 1
            reciprocal += 1 / rank
    n = len(testset)
    return hits / n, reciprocal / n


def judge_answers(testset: list[dict], n: int) -> tuple[float, float, int]:
    rag = DocuMind()  # hybrid + reranker, default chunk size
    faithful = correct = scored = 0
    for item in testset[:n]:
        out = rag.ask(item["question"])
        prompt = (
            JUDGE_PROMPT.replace("<<Q>>", item["question"])
            .replace("<<GOLD>>", item["gold_answer"])
            .replace("<<CONTEXT>>", format_context(out["sources"]))
            .replace("<<ANSWER>>", out["answer"])
        )
        verdict = extract_json(generate(prompt, system="You are a strict evaluator."))
        if not verdict:
            continue
        scored += 1
        faithful += bool(verdict.get("faithful"))
        correct += bool(verdict.get("correct"))
        print(f"  judged {scored}/{n}")
    return faithful / max(scored, 1), correct / max(scored, 1), scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/testset.jsonl")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[config.CHUNK_SIZE])
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-n", type=int, default=20)
    args = parser.parse_args()

    testset = load_testset(args.testset)
    print(f"Loaded {len(testset)} questions.")

    rows = []
    for size in args.chunk_sizes:
        if not collection_exists(size):
            print(f"Building index for chunk size {size}...")
            build_index(size, overlap=int(size * 0.15))
        for name, kwargs in CONFIGS:
            retriever = Retriever(chunk_size=size, **kwargs)
            hit, mrr = retrieval_metrics(retriever, testset, args.k)
            rows.append((size, name, hit, mrr))
            print(f"chunk={size:<5} {name:<24} Hit@{args.k}={hit:.1%}  MRR={mrr:.3f}")

    lines = [
        f"# Evaluation results ({date.today()})",
        "",
        f"Test set: {len(testset)} questions. Hit@{args.k} is page-level.",
        "",
        f"| Chunk size | Retrieval method | Hit@{args.k} | MRR |",
        "|---|---|---|---|",
    ]
    lines += [f"| {s} | {n} | {h:.1%} | {m:.3f} |" for s, n, h, m in rows]

    if args.judge:
        print("Judging generated answers (uses LLM calls)...")
        faithful, correct, scored = judge_answers(testset, args.judge_n)
        lines += [
            "",
            f"## Answer quality (LLM-as-judge, {scored} questions, hybrid + reranker)",
            "",
            f"- Faithfulness (no unsupported claims): {faithful:.1%}",
            f"- Correctness vs reference answer: {correct:.1%}",
            "",
            "_Judge and generator use the same LLM family, so treat these as indicative._",
        ]

    out = Path("eval/results.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
