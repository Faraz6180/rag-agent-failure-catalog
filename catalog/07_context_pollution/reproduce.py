"""Failure 07 - Context window pollution / lost-in-the-middle.

Run:  python catalog/07_context_pollution/reproduce.py [--live]

Scenario
--------
The prompt contains 12 chunks: one has the answer, eleven are plausible but irrelevant. We move
the answer-bearing chunk through every position and ask the same question each time. The
information never changes; only its position does.

Offline mode uses a *toy* reader whose attention is strongest at the edges of the context and
weakest in the middle, modelling the position effect reported for long-context models. It is a
stand-in, not a measurement. Run with --live to measure a real model: that is the useful part.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import Chunk, format_context, get_chunk
from common.utils import banner, base_parser, verdict

QUESTION = "How many requests per minute does the Pro plan allow?"
GOLD = get_chunk("rate-limits")
EXPECTED = "1,000"
DISTRACTORS = [get_chunk(i) for i in
               ["pricing", "sla", "retention", "support", "sso", "regions", "security",
                "deployments", "webhooks", "noise-01", "noise-02"]]
SYSTEM = "Answer the question using only the numbered context."


def simulated_model(system: str, user: str, temperature: float) -> str:
    """Toy position-biased reader. Attention is 1.0 at the edges and 0.4 in the middle."""
    lines = [ln for ln in user.splitlines() if ln.startswith("[")]
    n = len(lines)
    for pos, line in enumerate(lines):
        if "1,000 requests" in line:
            edge_distance = abs(2 * pos / max(n - 1, 1) - 1)  # 1 at the edges, 0 in the middle
            attention = 0.4 + 0.6 * edge_distance
            if attention >= 0.6:
                return "The Pro plan allows 1,000 requests per minute."
    return "The provided documentation does not specify a rate limit for the Pro plan."


def build_context(gold_position: int, gold: Chunk = GOLD, distractors: list[Chunk] = DISTRACTORS) -> list[Chunk]:
    chunks = list(distractors)
    chunks.insert(gold_position, gold)
    return chunks


def ask(client: LLMClient, chunks: list[Chunk]) -> str:
    return client.complete(f"Context:\n{format_context(chunks)}\n\nQuestion: {QUESTION}", system=SYSTEM)


def position_sweep(client: LLMClient, repeats: int = 1) -> list[float]:
    """Accuracy at each gold position (fraction of `repeats` runs that contain the expected value)."""
    n = len(DISTRACTORS) + 1
    return [sum(EXPECTED in ask(client, build_context(pos)) for _ in range(repeats)) / repeats for pos in range(n)]


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(simulated_model, live=args.live)

    banner(f"07 Lost in the middle   (model: {client.name})")
    acc = position_sweep(client, repeats=3 if args.live else 1)
    print("position of the answer chunk -> correct?   (same 12 chunks, same question)\n")
    for pos, a in enumerate(acc):
        print(f"  pos {pos:>2}  {'#' * int(a * 20):<20} {a:.0%}")
    edges = (acc[0] + acc[-1]) / 2
    middle = sum(acc[4:8]) / 4
    print(f"\nedge accuracy={edges:.0%}  middle accuracy={middle:.0%}")
    verdict(edges - middle >= 0.3, "the answer is in the context in every run, but the model only finds it near the edges.")


if __name__ == "__main__":
    main()
