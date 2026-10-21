"""Detect citation drift by verifying each citation against the chunk it names.

Run:  python catalog/02_citation_drift/detect.py

For every cited sentence we compute support against the cited chunk and against all chunks:

    supported   the cited chunk supports the claim
    drift       the claim is supported, but by a different chunk (citation is wrong)
    unsupported no chunk supports the claim (citation is decoration)

The distinction matters: drift is fixable automatically, unsupported content is not.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import Chunk, get_chunk
from common.utils import (banner, extract_citations, load_sibling, split_sentences, status,
                          support_score)

MIN_SUPPORT = 0.6


@dataclass
class CitationFinding:
    sentence: str
    cited: list[int]
    verdict: str  # supported | drift | unsupported | no_citation
    best_chunk: int | None
    best_support: float


def check_citations(answer: str, chunks: Sequence[Chunk], min_support: float = MIN_SUPPORT) -> list[CitationFinding]:
    findings = []
    for sentence in split_sentences(answer):
        cited = extract_citations(sentence)
        supports = [support_score(sentence, c.text) for c in chunks]
        best = max(range(len(chunks)), key=lambda i: supports[i])
        best_support = supports[best]
        if not cited:
            verdict = "no_citation"
        elif best_support < min_support:
            verdict = "unsupported"
        elif any(1 <= i <= len(chunks) and supports[i - 1] >= min_support for i in cited):
            verdict = "supported"
        else:
            verdict = "drift"
        findings.append(CitationFinding(sentence, cited, verdict, best + 1, best_support))
    return findings


# ---- tiny labeled set: (answer, expected verdict per sentence) ---------------------------------
def _chunks() -> list[Chunk]:
    return [get_chunk("sla"), get_chunk("rate-limits"), get_chunk("retention")]


LABELED = [
    ("The Pro plan includes a 99.9% uptime SLA [1].", ["supported"]),
    ("It allows 1,000 requests per minute [3].", ["drift"]),
    ("Logs are retained for 90 days on Pro [3].", ["supported"]),
    ("Logs are retained for 90 days on Pro [1].", ["drift"]),
    ("Violations trigger automatic service credits [1].", ["unsupported"]),
    ("Pro allows 1,000 requests per minute [2].", ["supported"]),
]


def evaluate() -> dict[str, float]:
    """Accuracy of the verdicts on the labeled set."""
    correct = total = 0
    for answer, expected in LABELED:
        got = [f.verdict for f in check_citations(answer, _chunks())]
        correct += sum(g == e for g, e in zip(got, expected))
        total += len(expected)
    return {"accuracy": correct / total, "n": total}


def main() -> None:
    banner("02 Detect: verify each citation against its chunk")
    rep = load_sibling(__file__, "reproduce")
    answer = rep.simulated_model("", "", 0.0)
    for f in check_citations(answer, rep.CHUNKS):
        status(f.verdict == "supported", f"[{f.verdict}] cited={f.cited} best=[{f.best_chunk}] ({f.best_support:.2f})", f.sentence)
    print(f"\nEval: {evaluate()}")


if __name__ == "__main__":
    main()
