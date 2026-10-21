"""Detect inconsistency with a paraphrase-and-resample consistency check.

Run:  python catalog/08_inconsistent_behavior/detect.py [--live]

For each question you care about, collect answers to several paraphrases and several samples,
reduce each answer to the fact a user would act on, and measure agreement. Report two numbers:

    agreement     share of answers matching the most common fact signature
    correctness   share of answers matching the documented fact

They are different. A system can be perfectly consistent and consistently wrong, and a gate
based on agreement alone will approve it.

The fact signature here is "the numbers in the answer". Real checks should extract whatever
your users act on: yes/no, an entity, a date, a plan name. Use an LLM or a rule per question type.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import banner, base_parser, load_sibling, status

rep = load_sibling(__file__, "reproduce")

MIN_AGREEMENT = 0.8


@dataclass
class ConsistencyReport:
    agreement: float
    correctness: float
    signatures: dict[str, int]

    @property
    def inconsistent(self) -> bool:
        return self.agreement < MIN_AGREEMENT


def consistency_report(answers: list[str], expected_fact: str) -> ConsistencyReport:
    keys = [rep.fact_key(a) for a in answers]
    counts = Counter(keys)
    modal_count = counts.most_common(1)[0][1]
    correct = sum(expected_fact in k.split("+") for k in keys)
    return ConsistencyReport(modal_count / len(keys), correct / len(keys), dict(counts))


def collect(answer_fn: Callable[[str, float], str], paraphrases: list[str], samples: int = 3) -> list[str]:
    answers = []
    for q in paraphrases:
        answers.append(answer_fn(q, 0.0))
        answers += [answer_fn(q, 0.7) for _ in range(samples)]
    return answers


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = rep.get_client(rep.make_stub(), live=args.live)
    answers = collect(lambda q, t: rep.ask(client, q, t), rep.PARAPHRASES)
    r = consistency_report(answers, rep.EXPECTED_FACT)

    banner(f"08 Detect: paraphrase + resample consistency   (model: {client.name})")
    status(not r.inconsistent, "agreement", f"{r.agreement:.0%} (min {MIN_AGREEMENT:.0%})")
    status(r.correctness >= MIN_AGREEMENT, "correctness vs documented fact", f"{r.correctness:.0%}")
    print(f"signatures: {r.signatures}")


if __name__ == "__main__":
    main()
