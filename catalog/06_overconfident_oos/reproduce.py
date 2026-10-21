"""Failure 06 - Over-confident answers on out-of-scope or ambiguous questions.

Run:  python catalog/06_overconfident_oos/reproduce.py [--live]

Scenario
--------
A support bot for Acme Cloud receives four kinds of question: answerable, out-of-scope (the
docs do not cover it), opinion, and ambiguous ("my plan" when plans differ). The failure is
that all four get the same confident, unhedged tone. The out-of-scope answers are invented, and
the ambiguous one silently picks a plan.

The offline stub always answers confidently, which is the behaviour under study.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import get_client
from common.retrieval_simulator import KeywordRetriever, format_context
from common.utils import banner, base_parser, verdict

QUESTIONS = [
    {"q": "What is the rate limit on the Pro plan?", "kind": "in_scope"},
    {"q": "What is the refund policy for annual plans?", "kind": "in_scope"},
    {"q": "Does Acme Cloud offer a free trial?", "kind": "oos"},
    {"q": "Which database should I use for my startup?", "kind": "oos"},
    {"q": "Is Acme Cloud better than AWS?", "kind": "oos"},
    {"q": "What's the rate limit for my plan?", "kind": "ambiguous"},
]

SYSTEM = "You are a support assistant for Acme Cloud. Answer the user's question using the context."

HEDGES = (
    "i'm not sure", "i am not sure", "i don't know", "i do not know", "not certain", "it depends",
    "couldn't find", "could not find", "can't find", "cannot find", "unclear", "i don't have",
    "which plan", "could you clarify", "not covered", "may vary", "might",
)


def is_hedged(answer: str) -> bool:
    lowered = answer.lower()
    return any(h in lowered for h in HEDGES)


def simulated_model(system: str, user: str, temperature: float) -> str:
    """Always confident. For in-scope questions the content happens to be right."""
    u = user.lower()
    if "free trial" in u:
        return "Yes, there is a 14-day free trial with no credit card required."
    if "database" in u:
        return "For most startups, PostgreSQL is the best choice because it is reliable, flexible and well supported."
    if "better than aws" in u:
        return "Yes, Acme Cloud is generally better than AWS for small teams thanks to simpler pricing and faster setup."
    if "my plan" in u:
        return "Your rate limit is 1,000 requests per minute."  # silently assumes Pro
    if "refund" in u:
        return "Annual plans can be refunded in full within 30 days of purchase."
    return "Pro allows 1,000 requests per minute."


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(simulated_model, live=args.live)
    retriever = KeywordRetriever()

    banner(f"06 Over-confident answers   (model: {client.name})")
    bad = 0
    for item in QUESTIONS:
        hits = retriever.retrieve(item["q"], k=3)
        context = format_context([c for c, _ in hits])
        answer = client.complete(f"Context:\n{context}\n\nQuestion: {item['q']}", system=SYSTEM)
        confident = not is_hedged(answer)
        should_hedge = item["kind"] != "in_scope"
        bad += confident and should_hedge
        flag = "OVERCONFIDENT" if confident and should_hedge else "ok"
        print(f"[{item['kind']:<9}] top-1 retrieval={hits[0][1]:.2f}  {flag}\n    Q: {item['q']}\n    A: {answer}\n")
    verdict(bad > 0, f"{bad} of 4 out-of-scope/ambiguous questions got a confident, unhedged answer.")


if __name__ == "__main__":
    main()
