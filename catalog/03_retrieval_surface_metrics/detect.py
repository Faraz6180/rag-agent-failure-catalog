"""Detect intent-level retrieval failures with a small answerability eval set.

Run:  python catalog/03_retrieval_surface_metrics/detect.py

Surface metrics cannot see this failure, so the detector is an eval set: real user phrasings
paired with a fact that must be present in the retrieved context. Run it in CI on every change
to chunking, embeddings, query rewriting or the corpus, and fail the build on regression.

`gap_report` prints the per-query diagnosis you need to fix a miss: what was retrieved, what
was missing, and where the missing chunk ranked (if anywhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import ACME_CHUNKS, KeywordRetriever
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")

MIN_ANSWER_RECALL = 0.90  # CI gate; choose from your own baseline


def rank_of_answer(retriever: KeywordRetriever, query: str, must_contain: str) -> int | None:
    """1-based rank of the first chunk containing the needed fact over the whole corpus, or None."""
    ranked = retriever.retrieve(query, k=len(retriever.chunks))
    for i, (chunk, _) in enumerate(ranked, start=1):
        if must_contain.lower() in chunk.text.lower():
            return i
    return None


def gap_report(retriever: KeywordRetriever, k: int = 3) -> list[dict]:
    report = []
    rows = rep.evaluate(retriever, k=k)
    for row, case in zip(rows, rep.EVAL_SET):
        if not row.answer_recall:
            report.append(
                {
                    "query": row.query,
                    "retrieved": row.top_ids,
                    "needed": case["must_contain"],
                    "answer_chunk_rank": rank_of_answer(retriever, case["query"], case["must_contain"]),
                }
            )
    return report


def passes_gate(retriever: KeywordRetriever, k: int = 3, rewrite=None) -> bool:
    return rep.summarize(rep.evaluate(retriever, k=k, rewrite=rewrite))["answer_recall@k"] >= MIN_ANSWER_RECALL


def main() -> None:
    banner("03 Detect: answerability eval set")
    retriever = KeywordRetriever(ACME_CHUNKS)
    for item in gap_report(retriever):
        status(False, item["query"], f"needed {item['needed']!r}; answer chunk ranked #{item['answer_chunk_rank']}; got {item['retrieved']}")
    ok = passes_gate(retriever)
    status(ok, f"CI gate: answer_recall@3 >= {MIN_ANSWER_RECALL:.0%}")


if __name__ == "__main__":
    main()
