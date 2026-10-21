"""Mitigation: decide whether to answer *before* generating.

Run:  python catalog/06_overconfident_oos/mitigate.py

A scope gate sits between retrieval and generation and picks one of three actions:

    answer    evidence is strong and the question is specific
    clarify   the question is ambiguous (ask which plan)
    abstain   the knowledge base has little on this topic

Routing before generation is cheaper and more reliable than asking the model to express
calibrated uncertainty after the fact. Log every abstain/clarify decision: they are your
best signal for documentation gaps and missing features.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import KeywordRetriever, format_context
from common.utils import banner, base_parser, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")

ABSTAIN_MSG = "I don't have documentation covering that, so I can't answer reliably. A support engineer can help."
CLARIFY_MSG = "Which plan are you asking about: Starter, Pro or Enterprise? The answer differs by plan."
EXPECTED = {"in_scope": "answer", "oos": "abstain", "ambiguous": "clarify"}


@dataclass
class Decision:
    action: str  # answer | clarify | abstain
    reason: str


class ScopeGate:
    def __init__(self, retriever: KeywordRetriever, evidence_threshold: float = det.EVIDENCE_THRESHOLD) -> None:
        self.retriever = retriever
        self.threshold = evidence_threshold

    def decide(self, question: str) -> Decision:
        if det.weak_evidence(question, self.retriever, self.threshold):
            return Decision("abstain", "top-1 retrieval score below threshold")
        if det.is_ambiguous(question, self.retriever):
            return Decision("clarify", "question names no plan but the best chunk differs by plan")
        return Decision("answer", "strong evidence, specific question")


def guarded_answer(client: LLMClient, question: str, gate: ScopeGate) -> tuple[Decision, str]:
    decision = gate.decide(question)
    if decision.action == "abstain":
        return decision, ABSTAIN_MSG
    if decision.action == "clarify":
        return decision, CLARIFY_MSG
    context = format_context([c for c, _ in gate.retriever.retrieve(question, k=3)])
    return decision, client.complete(f"Context:\n{context}\n\nQuestion: {question}", system=rep.SYSTEM)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(rep.simulated_model, live=args.live)
    gate = ScopeGate(KeywordRetriever())

    banner("06 Mitigate: scope gate (answer / clarify / abstain)")
    correct = 0
    for item in rep.QUESTIONS:
        decision, text = guarded_answer(client, item["q"], gate)
        ok = decision.action == EXPECTED[item["kind"]]
        correct += ok
        status(ok, f"[{item['kind']}] {item['q']}", f"action={decision.action}")
        print(f"      -> {text}")
    print(f"\nRouting matched expectation on {correct}/{len(rep.QUESTIONS)} questions (tiny set; calibrate on real traffic).")


if __name__ == "__main__":
    main()
