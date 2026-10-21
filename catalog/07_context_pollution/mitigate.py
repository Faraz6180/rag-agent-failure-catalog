"""Mitigation: send less context, and order what remains for the reader.

Run:  python catalog/07_context_pollution/mitigate.py [--live]

Pipeline: dedupe -> rank -> trim to k and a token budget -> edge-order.

  dedupe      near-duplicate chunks add tokens but no information
  rank        by retrieval score
  trim        fewer chunks means fewer places for the answer to hide, and less noise
  edge-order  best chunk first, second-best last, weakest in the middle

Trimming trades recall for precision: if k is too small the answer chunk is dropped. Pick k by
measuring answer recall (see entry 03), not by feel.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import Chunk, KeywordRetriever
from common.utils import banner, base_parser, content_tokens, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")


def dedupe(chunks: Sequence[Chunk], threshold: float = 0.8) -> list[Chunk]:
    kept: list[Chunk] = []
    kept_sets: list[set[str]] = []
    for c in chunks:
        toks = set(content_tokens(c.text))
        if any(len(toks & other) / max(len(toks | other), 1) >= threshold for other in kept_sets):
            continue
        kept.append(c)
        kept_sets.append(toks)
    return kept


def edge_order(ranked: Sequence[Chunk]) -> list[Chunk]:
    """Place rank 1 first, rank 2 last, rank 3 second, rank 4 second-to-last, and so on."""
    front: list[Chunk] = []
    back: list[Chunk] = []
    for i, chunk in enumerate(ranked):
        (front if i % 2 == 0 else back).append(chunk)
    return front + back[::-1]


def prepare_context(question: str, candidates: Sequence[Chunk], k: int = 5, token_budget: int = 200) -> list[Chunk]:
    unique = dedupe(candidates)
    retriever = KeywordRetriever(unique)
    ranked = [c for c, _ in retriever.retrieve(question, k=len(unique))]
    kept: list[Chunk] = []
    used = 0
    for c in ranked[:k]:
        cost = len(content_tokens(c.text))
        if kept and used + cost > token_budget:
            break
        kept.append(c)
        used += cost
    return edge_order(kept)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = rep.get_client(rep.simulated_model, live=args.live)
    banner(f"07 Mitigate: dedupe, trim, edge-order   (model: {client.name})")

    candidates = rep.build_context(6)  # worst case for the naive prompt: answer in the middle
    naive = rep.ask(client, candidates)
    prepared = prepare_context(rep.QUESTION, candidates)
    guarded = rep.ask(client, prepared)

    status(rep.EXPECTED in naive, f"naive: all {len(candidates)} chunks, answer at position 6", naive)
    status(rep.EXPECTED in guarded, f"prepared: {len(prepared)} chunks, edge-ordered", guarded)
    print(f"\npollution: {det.pollution_ratio(candidates, {rep.GOLD.id}):.0%} -> {det.pollution_ratio(prepared, {rep.GOLD.id}):.0%}")
    print("prepared order:", [c.id for c in prepared])


if __name__ == "__main__":
    main()
