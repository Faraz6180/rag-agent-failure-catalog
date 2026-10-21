"""Failure 03 - Retrieval that looks fine on surface metrics but misses the user's intent.

Run:  python catalog/03_retrieval_surface_metrics/reproduce.py

Scenario
--------
Users phrase things in their own words ("get my money back", "log in with Okta", "weekends").
The documentation uses its own ("refund", "SAML SSO", "support hours"). The retriever returns
k chunks every time, and the top chunk almost always shares *some* keyword with the question,
so surface checks stay green while the chunk that answers the question is not retrieved.

Surface metrics (what dashboards usually track):
    returned_k      retriever returned k results
    keyword_hit@1   top-1 chunk shares at least one query keyword
Intent metric (what users need):
    answer_recall@k at least one retrieved chunk contains the fact needed to answer

No model is called; this entry is about retrieval only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import KeywordRetriever
from common.utils import banner, content_tokens, verdict

# Each case: the user's wording, and a phrase that must appear in a retrieved chunk for the
# question to be answerable. These are the cases you should be collecting from real traffic.
EVAL_SET = [
    {"query": "can I get my money back if I cancel my annual plan?", "must_contain": "refunded in full"},
    {"query": "what happens to my project and data if I stop paying?", "must_contain": "deletion"},
    {"query": "who do I contact on weekends when my deployment fails?", "must_contain": "24/7"},
    {"query": "how many calls per minute can I make on the pro tier?", "must_contain": "1,000 requests per minute"},
    {"query": "can my team log in with Okta?", "must_contain": "SAML SSO"},
    {"query": "is my data encrypted and who audited you?", "must_contain": "SOC 2"},
]


@dataclass
class Row:
    query: str
    top_ids: list[str]
    top1_score: float
    returned_k: bool
    keyword_hit_at_1: bool
    answer_recall: bool


def evaluate(
    retriever: KeywordRetriever,
    k: int = 3,
    rewrite: Optional[Callable[[str], str]] = None,
    eval_set: list[dict] = EVAL_SET,
) -> list[Row]:
    rows = []
    for case in eval_set:
        query = rewrite(case["query"]) if rewrite else case["query"]
        hits = retriever.retrieve(query, k=k)
        top_chunk, top_score = hits[0]
        qtoks = set(content_tokens(case["query"]))
        rows.append(
            Row(
                query=case["query"],
                top_ids=[c.id for c, _ in hits],
                top1_score=top_score,
                returned_k=len(hits) == k,
                keyword_hit_at_1=bool(qtoks & set(content_tokens(top_chunk.text))),
                answer_recall=any(case["must_contain"].lower() in c.text.lower() for c, _ in hits),
            )
        )
    return rows


def summarize(rows: list[Row]) -> dict[str, float]:
    n = len(rows)
    return {
        "returned_k": sum(r.returned_k for r in rows) / n,
        "keyword_hit@1": sum(r.keyword_hit_at_1 for r in rows) / n,
        "answer_recall@k": sum(r.answer_recall for r in rows) / n,
    }


def print_table(rows: list[Row]) -> None:
    print(f"{'query':<58} {'k?':>3} {'kw@1':>5} {'answer':>7}  top-3")
    for r in rows:
        print(f"{r.query[:57]:<58} {'y' if r.returned_k else 'n':>3} {'y' if r.keyword_hit_at_1 else 'n':>5} "
              f"{'HIT' if r.answer_recall else 'MISS':>7}  {', '.join(r.top_ids)}")


def main() -> None:
    banner("03 Retrieval that looks good but misses intent")
    rows = evaluate(KeywordRetriever(), k=3)
    print_table(rows)
    m = summarize(rows)
    print("\nSurface metrics:  " + ", ".join(f"{k}={v:.0%}" for k, v in m.items() if k != "answer_recall@k"))
    print(f"Intent metric:    answer_recall@3={m['answer_recall@k']:.0%}")
    verdict(
        m["keyword_hit@1"] == 1.0 and m["answer_recall@k"] < 0.75,
        "surface metrics are at 100% while the chunk that answers the question is missing for several queries.",
    )


if __name__ == "__main__":
    main()
