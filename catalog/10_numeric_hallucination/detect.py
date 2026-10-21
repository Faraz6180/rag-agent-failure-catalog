"""Detect numeric, date and entity hallucination by extracting atoms and checking each one.

Run:  python catalog/10_numeric_hallucination/detect.py

Sentence-level overlap (entry 01) is the wrong tool for this failure: "Logs are retained for
60 days" shares almost every word with "Logs are retained for 90 days". So instead we extract
*atoms* (money, percentages, quantities with units, dates, region names), normalize them, and
require each one in the answer to appear in the context.

Quantities include their unit, so "90 hours" does not match "90 days".

Extend `ATOM_PATTERNS` for your domain: SKUs, version numbers, plan names, account IDs.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import format_context
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")

_UNITS = r"(?:days?|hours?|minutes?|seconds?|requests?|months?|years?|GB|MB|TB|records?|users?|seats?)"
_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

ATOM_PATTERNS = {
    "money": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    "percent": re.compile(r"\d+(?:\.\d+)?\s?%"),
    "quantity": re.compile(rf"(?<![\w.$])\d[\d,]*(?:\.\d+)?\s+{_UNITS}\b", re.I),
    "date_iso": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "date_text": re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?", re.I),
    "region": re.compile(r"\b[a-z]{2}-[a-z]+-\d\b"),
}


@dataclass(frozen=True)
class Atom:
    kind: str
    value: str  # normalized


def _normalize(kind: str, raw: str) -> str:
    text = raw.lower().replace(",", "").replace(" ", "")
    if kind == "quantity":
        m = re.match(r"([\d.]+)([a-z]+)", raw.lower().replace(",", "").replace(" ", ""))
        if m:
            unit = m.group(2)
            text = f"{m.group(1)}{unit[:-1] if unit.endswith('s') and len(unit) > 2 else unit}"
    return text


def extract_atoms(text: str) -> set[Atom]:
    atoms = set()
    for kind, pattern in ATOM_PATTERNS.items():
        for m in pattern.finditer(text):
            atoms.add(Atom(kind, _normalize(kind, m.group(0))))
    return atoms


def unsupported_atoms(answer: str, context: str) -> list[Atom]:
    return sorted(extract_atoms(answer) - extract_atoms(context), key=lambda a: (a.kind, a.value))


# ---- tiny labeled set: (context, answer, has_error) --------------------------------------------
LABELED = [
    ("Logs are retained for 90 days on Pro.", "Pro retains logs for 90 days.", False),
    ("Logs are retained for 90 days on Pro.", "Pro retains logs for 60 days.", True),
    ("Logs are retained for 90 days on Pro.", "Pro retains logs for 90 hours.", True),
    ("The Pro plan includes a 99.9% uptime SLA.", "Pro includes a 99.99% uptime SLA.", True),
    ("The Pro plan costs $99 per month.", "Pro costs $99 per month.", False),
    ("It runs in us-east-1 and eu-west-1.", "It runs in us-east-1, eu-west-1 and us-west-2.", True),
    ("The migration finished on 2025-03-14 and moved 1,200 records.", "The migration finished on 2025-03-12.", True),
    ("The migration finished on 2025-03-14 and moved 1,200 records.", "It moved 1200 records on 2025-03-14.", False),
]


def evaluate() -> dict[str, float]:
    tp = fp = fn = tn = 0
    for context, answer, label in LABELED:
        flagged = bool(unsupported_atoms(answer, context))
        tp += flagged and label
        fp += flagged and not label
        fn += (not flagged) and label
        tn += (not flagged) and not label
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "n": len(LABELED)}


def main() -> None:
    banner("10 Detect: atom-level verification")
    context = format_context(rep.CHUNKS)
    for atom in sorted(extract_atoms(rep.FAULTY), key=lambda a: (a.kind, a.value)):
        ok = atom in extract_atoms(context)
        status(ok, f"{atom.kind:<9} {atom.value}", "in context" if ok else "NOT in context")
    print(f"\nEval: {evaluate()}")


if __name__ == "__main__":
    main()
