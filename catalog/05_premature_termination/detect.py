"""Detect premature termination by checking the final answer against an explicit checklist.

Run:  python catalog/05_premature_termination/detect.py

The idea is a "definition of done" that exists outside the model. For tasks you can enumerate
(report sections, required fields, questions that must be answered), write the checklist as
patterns and score the final answer against it. Coverage < 100% on a run that stopped by
itself is a premature-termination candidate.

For open-ended tasks you cannot enumerate, use an LLM judge with the same interface: given the
task and the answer, list requested items that the answer does not address. The checklist
version is cheaper, deterministic and good enough where it applies.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")


@dataclass
class Coverage:
    covered: list[str]
    missing: list[str]

    @property
    def ratio(self) -> float:
        total = len(self.covered) + len(self.missing)
        return len(self.covered) / total if total else 1.0


def checklist_coverage(answer: str, checklist: dict[str, list[str]]) -> Coverage:
    """An item is covered only if *all* of its patterns match (e.g. both plans' values)."""
    covered, missing = [], []
    for item, patterns in checklist.items():
        (covered if all(re.search(p, answer) for p in patterns) else missing).append(item)
    return Coverage(covered, missing)


LABELED = [
    (rep.PARTIAL, False),
    (rep.COMPLETE, True),
    ("Starter: $29, 100 requests per minute. Pro: $99, 1,000 requests per minute.", False),  # no retention
    ("Starter: $29, logs 30 days. Pro: $99, logs 90 days.", False),                           # no rate limits
]


def evaluate() -> dict[str, float]:
    """Does coverage == 100% agree with a human 'complete' label?"""
    agree = sum((checklist_coverage(a, rep.CHECKLIST).ratio == 1.0) == label for a, label in LABELED)
    return {"agreement": agree / len(LABELED), "n": len(LABELED)}


def main() -> None:
    banner("05 Detect: checklist coverage of the final answer")
    cov = checklist_coverage(rep.PARTIAL, rep.CHECKLIST)
    status(cov.ratio == 1.0, "agent's final answer", f"coverage={cov.ratio:.0%} covered={cov.covered} missing={cov.missing}")
    cov = checklist_coverage(rep.COMPLETE, rep.CHECKLIST)
    status(cov.ratio == 1.0, "complete reference answer", f"coverage={cov.ratio:.0%}")
    print(f"\nEval: {evaluate()}")


if __name__ == "__main__":
    main()
