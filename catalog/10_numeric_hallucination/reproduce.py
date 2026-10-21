"""Failure 10 - Numeric, date and entity hallucination that looks plausible.

Run:  python catalog/10_numeric_hallucination/reproduce.py [--live]

Scenario
--------
The context describes the Pro plan accurately. The model's summary is fluent and mostly right,
but it states a log retention of 60 days (the docs say 90) and lists a region that does not
exist (us-west-2). Both are plausible neighbours of the true values, which is what makes them
dangerous: a reviewer skimming for "does this look right" will pass them.

Offline mode uses a stub that produces exactly these errors. --live may or may not reproduce them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import get_client
from common.retrieval_simulator import format_context, get_chunk
from common.utils import banner, base_parser, verdict

CHUNKS = [get_chunk(i) for i in ["pricing", "rate-limits", "retention", "regions"]]
QUESTION = "Summarize the Pro plan: price, rate limit, log retention and available regions."
SYSTEM = "Summarize using only the context."

CORRECT = (
    "The Pro plan costs $99 per month and allows 1,000 requests per minute. "
    "Logs are retained for 90 days. It is available in us-east-1 and eu-west-1."
)
FAULTY = (
    "The Pro plan costs $99 per month and allows 1,000 requests per minute. "
    "Logs are retained for 60 days. It is available in us-east-1, eu-west-1 and us-west-2."
)


def simulated_model(system: str, user: str, temperature: float) -> str:
    # With a FACTS TABLE in the prompt the stub copies values correctly (see mitigate.py).
    return CORRECT if "FACTS TABLE" in user else FAULTY


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(simulated_model, live=args.live)
    context = format_context(CHUNKS)

    banner(f"10 Numeric / entity hallucination   (model: {client.name})")
    answer = client.complete(f"Context:\n{context}\n\nTask: {QUESTION}", system=SYSTEM)
    print(f"Answer:\n  {answer}\n")

    # Inline lexical check so this script stands alone; detect.py has the full atom-level check.
    from common.utils import support_detail

    score, missing = support_detail(answer, "\n".join(c.text for c in CHUNKS))
    print(f"Overall lexical support: {score:.2f}  (looks high: nearly every word is in the context)")
    print(f"Terms not in context: {missing}")
    verdict(bool(missing), "overall overlap is high, but one number and one region name are not in the source.")


if __name__ == "__main__":
    main()
