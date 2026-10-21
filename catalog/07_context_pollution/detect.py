"""Detect context pollution and position sensitivity.

Run:  python catalog/07_context_pollution/detect.py [--live]

Two measurements:

  position sweep    accuracy as a function of where the answer sits. A large edge-vs-middle
                    gap means ordering is silently changing your answers.
  pollution ratio   share of context tokens that are not relevant to the question. Track it
                    per request; it is free and predicts risk before you pay for generation.

Run the sweep on your real model and prompt before choosing k. The shape of the curve (and
whether it exists) is model-specific and changes between model versions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import Chunk
from common.utils import banner, base_parser, content_tokens, load_sibling, status

rep = load_sibling(__file__, "reproduce")


def pollution_ratio(chunks: Sequence[Chunk], relevant_ids: set[str]) -> float:
    total = sum(len(content_tokens(c.text)) for c in chunks)
    relevant = sum(len(content_tokens(c.text)) for c in chunks if c.id in relevant_ids)
    return 1 - relevant / total if total else 0.0


def sensitivity(acc: list[float]) -> dict[str, float]:
    """Edge-vs-middle accuracy gap over a position sweep."""
    k = max(len(acc) // 3, 1)
    edges = (sum(acc[:k]) + sum(acc[-k:])) / (2 * k)
    middle = sum(acc[k:-k]) / max(len(acc) - 2 * k, 1)
    return {"edge_accuracy": edges, "middle_accuracy": middle, "gap": edges - middle}


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = rep.get_client(rep.simulated_model, live=args.live)
    banner(f"07 Detect: position sweep + pollution ratio   (model: {client.name})")

    acc = rep.position_sweep(client, repeats=3 if args.live else 1)
    s = sensitivity(acc)
    status(s["gap"] < 0.2, "position sensitivity", f"edge={s['edge_accuracy']:.0%} middle={s['middle_accuracy']:.0%} gap={s['gap']:.0%}")

    chunks = rep.build_context(6)
    ratio = pollution_ratio(chunks, {rep.GOLD.id})
    status(ratio < 0.5, "pollution ratio of the 12-chunk prompt", f"{ratio:.0%} of context tokens are irrelevant")


if __name__ == "__main__":
    main()
