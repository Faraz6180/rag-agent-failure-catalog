"""Detect confident answers that the evidence does not justify.

Run:  python catalog/06_overconfident_oos/detect.py

Two cheap signals, combined:

  evidence   top-1 retrieval score. Low means the knowledge base has little on this topic.
  ambiguity  the question names no plan, but the best chunk gives different values per plan.

A response is flagged `overconfident` when evidence is weak or the question is ambiguous AND the
answer contains no hedging. The evaluation reports how many out-of-scope/ambiguous questions
are caught and how many in-scope ones are wrongly flagged.

The threshold below is tuned on six examples to keep the demo readable. Calibrate on at least
a few hundred labeled production questions before trusting a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import KeywordRetriever
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")

EVIDENCE_THRESHOLD = 0.5
PLAN_NAMES = ("starter", "pro", "enterprise")


def mentions_plans(text: str) -> set[str]:
    lowered = text.lower()
    return {p for p in PLAN_NAMES if p in lowered.split() or f"{p} " in lowered or f"{p}." in lowered}


def is_ambiguous(question: str, retriever: KeywordRetriever) -> bool:
    top_chunk, _ = retriever.retrieve(question, k=1)[0]
    return not mentions_plans(question) and len(mentions_plans(top_chunk.text)) >= 2


def weak_evidence(question: str, retriever: KeywordRetriever, threshold: float = EVIDENCE_THRESHOLD) -> bool:
    return retriever.retrieve(question, k=1)[0][1] < threshold


def flag_overconfident(question: str, answer: str, retriever: KeywordRetriever) -> list[str]:
    if rep.is_hedged(answer):
        return []
    reasons = []
    if weak_evidence(question, retriever):
        reasons.append("weak_evidence")
    elif is_ambiguous(question, retriever):  # only meaningful when the topic is covered at all
        reasons.append("ambiguous")
    return reasons


def evaluate(client=None, retriever: KeywordRetriever | None = None) -> dict[str, float]:
    """Catch rate on out-of-scope/ambiguous questions and false-flag rate on in-scope ones."""
    retriever = retriever or KeywordRetriever()
    client = client or rep.get_client(rep.simulated_model)
    caught = missed = false_flags = in_scope = 0
    for item in rep.QUESTIONS:
        answer = client.complete(item["q"], system=rep.SYSTEM)
        flagged = bool(flag_overconfident(item["q"], answer, retriever))
        if item["kind"] == "in_scope":
            in_scope += 1
            false_flags += flagged
        else:
            caught += flagged
            missed += not flagged
    return {
        "catch_rate": caught / (caught + missed),
        "false_flag_rate": false_flags / in_scope,
        "n": len(rep.QUESTIONS),
    }


def main() -> None:
    banner("06 Detect: evidence + ambiguity vs. confidence")
    retriever = KeywordRetriever()
    client = rep.get_client(rep.simulated_model)
    for item in rep.QUESTIONS:
        answer = client.complete(item["q"], system=rep.SYSTEM)
        reasons = flag_overconfident(item["q"], answer, retriever)
        status(not reasons, f"[{item['kind']}] {item['q']}", f"flags={reasons}")
    print(f"\nEval: {evaluate(client, retriever)}")


if __name__ == "__main__":
    main()
