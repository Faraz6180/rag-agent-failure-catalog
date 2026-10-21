"""Failure 01 - Ungrounded answer.

Run:  python catalog/01_ungrounded_answer/reproduce.py [--live]

Scenario
--------
The user asks about a free trial. The documentation says nothing about one. The retriever
still returns the "closest" chunks (retrievers always return *something*), and a model that
is trying to be helpful fills the gap with a fluent, specific, unsupported answer.

Offline mode uses a stub that models this behaviour so the run is deterministic.
--live sends the same prompt to a real model; many modern models will abstain here, which is
fine: try a weaker model, a more leading question, or a noisier corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import KeywordRetriever, format_context
from common.utils import banner, base_parser, support_detail, verdict

QUESTION = "Does Acme Cloud offer a free trial, and how long does it last?"

NAIVE_SYSTEM = (
    "You are a support assistant for Acme Cloud. Answer the user's question using the context."
)


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def simulated_model(system: str, user: str, temperature: float) -> str:
    """Models a helpful-but-ungrounded completion.

    For the refund question the context really does contain the answer, so the stub is
    grounded. For the trial question the context has nothing, so every specific below
    comes from the model's priors rather than the documents.
    """
    if "refund" in user.lower():
        return "Annual plans can be refunded in full within 30 days of purchase."
    return (
        "Yes, Acme Cloud offers a 14-day free trial of the Pro plan. "
        "No credit card is required, and you can upgrade at any time during the trial."
    )


def naive_rag_answer(
    client: LLMClient, question: str, retriever: KeywordRetriever, k: int = 3
) -> tuple[str, str]:
    """The baseline pipeline: retrieve, stuff, generate. Nothing checks the answer."""
    hits = retriever.retrieve(question, k=k)
    context = format_context([chunk for chunk, _ in hits])
    answer = client.complete(build_user_prompt(question, context), system=NAIVE_SYSTEM, temperature=0.0)
    return answer, context


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(simulated_model, live=args.live)
    retriever = KeywordRetriever()

    banner(f"01 Ungrounded answer   (model: {client.name})")
    answer, context = naive_rag_answer(client, QUESTION, retriever)
    print(f"Question:\n  {QUESTION}\n\nRetrieved context:\n{context}\n\nAnswer:\n  {answer}\n")

    score, missing = support_detail(answer, context)
    print(f"Context mentions 'trial': {'trial' in context.lower()}")
    print(f"Lexical support of answer in context: {score:.2f}")
    print(f"Answer terms with no support in context: {missing}")
    verdict(
        score < 0.6,
        "the answer asserts a trial length and billing terms that appear nowhere in the retrieved context.",
    )


if __name__ == "__main__":
    main()
