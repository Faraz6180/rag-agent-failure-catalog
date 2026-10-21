"""Failure 02 - Citation drift.

Run:  python catalog/02_citation_drift/reproduce.py [--live]

Scenario
--------
The model answers from three numbered chunks and cites them. The claims look right and every
sentence carries a citation, but some citations point at the wrong chunk, and one claim is not
in any chunk. A reader who clicks [2] to verify lands on the wrong passage.

Offline mode uses a stub with exactly this behaviour. --live asks a real model to cite and then
checks what it did.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import get_client
from common.retrieval_simulator import format_context, get_chunk
from common.utils import banner, base_parser, extract_citations, split_sentences, support_score, verdict

CHUNKS = [get_chunk("sla"), get_chunk("rate-limits"), get_chunk("retention")]
QUESTION = "What uptime SLA and rate limit does the Pro plan have, and how long are logs kept?"
SYSTEM = (
    "Answer using the numbered context. Cite the supporting chunk after every claim, like [1]. "
    "Use only the context."
)


def simulated_model(system: str, user: str, temperature: float) -> str:
    # Sentence 1 and 2 are cited correctly. Sentence 3 is true but cited to the wrong chunk.
    # Sentence 4 is not in any chunk at all, yet carries a citation that makes it look sourced.
    return (
        "The Pro plan includes a 99.9% uptime SLA [1]. "
        "It allows 1,000 requests per minute [2]. "
        "Logs are kept for 90 days [2]. "
        "Violations trigger automatic service credits [1]."
    )


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(simulated_model, live=args.live)
    context = format_context(CHUNKS)

    banner(f"02 Citation drift   (model: {client.name})")
    answer = client.complete(f"Context:\n{context}\n\nQuestion: {QUESTION}", system=SYSTEM)
    print(f"Context:\n{context}\n\nAnswer:\n  {answer}\n")

    bad = 0
    for sentence in split_sentences(answer):
        for idx in extract_citations(sentence) or [0]:
            if not 1 <= idx <= len(CHUNKS):
                print(f"  [BAD ] cites [{idx}] which does not exist: {sentence}")
                bad += 1
                continue
            s = support_score(sentence, CHUNKS[idx - 1].text)
            ok = s >= 0.6
            bad += not ok
            print(f"  [{' ok ' if ok else 'BAD '}] cites [{idx}] support={s:.2f}: {sentence}")
    verdict(bad > 0, f"{bad} citation(s) point at a chunk that does not support the claim.")


if __name__ == "__main__":
    main()
