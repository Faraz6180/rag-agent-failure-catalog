"""Mitigation: repair or remove citations after generation, and prefer quote-backed claims.

Run:  python catalog/02_citation_drift/mitigate.py

Two patterns:

1. repair_citations - post-process: rebind each claim to the chunk that actually supports it,
   and strip claims that no chunk supports. Cheap, deterministic, no extra model call.
2. verify_quotes - structured output: ask the model for (claim, chunk, verbatim quote) and
   accept a claim only if the quote is an exact substring of the named chunk. Exact-substring
   checks are much harder to fool than topical similarity.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import Chunk
from common.utils import banner, load_sibling, split_sentences, status, strip_citations

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")


def repair_citations(answer: str, chunks: Sequence[Chunk]) -> tuple[str, list[str]]:
    """Return (repaired answer, dropped claims). Unsupported claims are removed, not guessed at."""
    kept: list[str] = []
    dropped: list[str] = []
    findings = det.check_citations(answer, chunks)
    for f in findings:
        claim = strip_citations(f.sentence).rstrip(".")
        if f.verdict == "unsupported":
            dropped.append(f.sentence)
        elif f.verdict == "no_citation" and f.best_support < det.MIN_SUPPORT:
            dropped.append(f.sentence)
        else:
            kept.append(f"{claim} [{f.best_chunk}].")
    return " ".join(kept), dropped


def verify_quotes(claims: list[dict], chunks: Sequence[Chunk]) -> list[dict]:
    """Keep claims whose `quote` appears verbatim in chunk number `chunk` (1-based).

    Expected claim shape (ask the model for JSON in this form):
        {"claim": "...", "chunk": 2, "quote": "Pro allows 1,000 requests per minute."}
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    verified = []
    for c in claims:
        idx = c.get("chunk", 0)
        if 1 <= idx <= len(chunks) and norm(c.get("quote", "")) and norm(c["quote"]) in norm(chunks[idx - 1].text):
            verified.append(c)
    return verified


def main() -> None:
    banner("02 Mitigate: citation repair + quote verification")
    answer = rep.simulated_model("", "", 0.0)
    print(f"Before:\n  {answer}\n")

    repaired, dropped = repair_citations(answer, rep.CHUNKS)
    print(f"After repair_citations:\n  {repaired}\n")
    for d in dropped:
        status(False, "dropped unsupported claim", d)

    print("\nQuote verification:")
    claims = [
        {"claim": "Pro allows 1,000 requests per minute", "chunk": 2, "quote": "Pro allows 1,000 requests per minute."},
        {"claim": "Logs kept 90 days on Pro", "chunk": 2, "quote": "retained for 90 days on Pro"},      # wrong chunk
        {"claim": "Automatic service credits", "chunk": 1, "quote": "automatic service credits"},      # invented quote
    ]
    ok = verify_quotes(claims, rep.CHUNKS)
    for c in claims:
        status(c in ok, c["claim"], f"chunk [{c['chunk']}]")


if __name__ == "__main__":
    main()
