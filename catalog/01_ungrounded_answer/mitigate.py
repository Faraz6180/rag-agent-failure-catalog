"""Mitigation: instruct the model to abstain, then verify the draft before returning it.

Run:  python catalog/01_ungrounded_answer/mitigate.py

Two layers, because either alone is not enough:
  1. Prompt: tell the model to answer only from context and emit a sentinel if it cannot.
  2. Gate: check every sentence of the draft against the context and replace the answer
     with an explicit "not found" response if anything is unsupported.

The offline stub ignores the prompt instruction on purpose (as real models sometimes do), so
this demo shows why the deterministic gate is the part you can rely on.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import KeywordRetriever, format_context
from common.utils import banner, base_parser, load_sibling

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"
GROUNDED_SYSTEM = (
    "You are a support assistant for Acme Cloud. Answer ONLY with facts stated in the context. "
    f"If the context does not contain the answer, reply exactly: {ABSTAIN_TOKEN}"
)
FALLBACK = (
    "I couldn't find that in the Acme Cloud documentation, so I won't guess. "
    "Please contact support or rephrase the question."
)


@dataclass
class GuardedAnswer:
    text: str
    grounded: bool
    reason: str


def guarded_answer(
    client: LLMClient, question: str, retriever: KeywordRetriever, k: int = 3, threshold: float = det.THRESHOLD
) -> GuardedAnswer:
    context = format_context([c for c, _ in retriever.retrieve(question, k=k)])
    draft = client.complete(rep.build_user_prompt(question, context), system=GROUNDED_SYSTEM, temperature=0.0)

    if ABSTAIN_TOKEN in draft:
        return GuardedAnswer(FALLBACK, False, "model abstained")
    flagged = [f for f in det.find_unsupported(draft, context, threshold) if f.flagged]
    if flagged:
        # Log what was suppressed: these are your best source of new eval cases.
        return GuardedAnswer(FALLBACK, False, f"unsupported: {flagged[0].sentence!r} (missing {flagged[0].missing})")
    return GuardedAnswer(draft, True, "all sentences supported")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(rep.simulated_model, live=args.live)
    retriever = KeywordRetriever()

    banner("01 Mitigate: abstention prompt + post-hoc groundedness gate")
    for question in [rep.QUESTION, "What is the refund policy for annual plans?"]:
        result = guarded_answer(client, question, retriever)
        tag = "[PASSED ]" if result.grounded else "[BLOCKED]"
        print(f"{tag} {question}\n          reason: {result.reason}\n          -> {result.text}\n")


if __name__ == "__main__":
    main()
