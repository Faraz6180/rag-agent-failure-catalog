"""Mitigation: canonicalize the question, vote across samples, cache the result.

Run:  python catalog/08_inconsistent_behavior/mitigate.py [--live]

  1. Canonicalize: map recognized intents to one canonical question, so paraphrases share a
     prompt. Here it is a rule table; in production use a classifier, an embedding match
     against known intents (a semantic cache), or an LLM rewrite step.
  2. Vote: sample K answers on the canonical prompt and return the most common fact signature.
  3. Cache by intent, so the same intent returns the same answer until you invalidate it.

Warning: this makes the system more *consistent*, not more *correct*. If the model's modal answer
is wrong, voting and caching make it reliably wrong. Always check correctness separately
(see detect.py) and review cached answers for high-stakes intents.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient
from common.utils import banner, base_parser, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")

# (all of these terms must appear, canonical question)
INTENT_RULES = [
    ({"refund", "annual"}, "What is the refund window for annual plans?"),
    ({"refund", "yearly"}, "What is the refund window for annual plans?"),
    ({"money back", "yearly"}, "What is the refund window for annual plans?"),
    ({"money back", "annual"}, "What is the refund window for annual plans?"),
]


def canonical_question(question: str) -> str:
    lowered = question.lower()
    for terms, canonical in INTENT_RULES:
        if all(t in lowered for t in terms):
            return canonical
    return question


class StableAnswerer:
    def __init__(self, client: LLMClient, votes: int = 5, temperature: float = 0.7) -> None:
        self.client = client
        self.votes = votes
        self.temperature = temperature
        self.cache: dict[str, str] = {}

    def answer(self, question: str) -> str:
        key = canonical_question(question)
        if key not in self.cache:
            samples = [rep.ask(self.client, key, self.temperature) for _ in range(self.votes)]
            winner, _ = Counter(rep.fact_key(s) for s in samples).most_common(1)[0]
            self.cache[key] = next(s for s in samples if rep.fact_key(s) == winner)
        return self.cache[key]


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = rep.get_client(rep.make_stub(), live=args.live)
    banner(f"08 Mitigate: canonicalize + vote + cache   (model: {client.name})")

    before = det.consistency_report(det.collect(lambda q, t: rep.ask(client, q, t), rep.PARAPHRASES), rep.EXPECTED_FACT)
    answerer = StableAnswerer(client)
    after_answers = [answerer.answer(q) for q in rep.PARAPHRASES for _ in range(4)]
    after = det.consistency_report(after_answers, rep.EXPECTED_FACT)

    status(not before.inconsistent, "before", f"agreement={before.agreement:.0%} correctness={before.correctness:.0%}")
    status(not after.inconsistent, "after ", f"agreement={after.agreement:.0%} correctness={after.correctness:.0%}")
    print(f"\ncached intents: {list(answerer.cache)}")
    print("Reminder: agreement went up by construction. Correctness is what you should be watching.")

    if not args.live:
        # Consistency is not correctness: repeat the whole experiment with different stub seeds.
        from common.llm_client import OfflineStub

        correct_runs = 0
        for seed in range(12):
            c = OfflineStub(rep.make_stub(seed))
            a = StableAnswerer(c)
            report = det.consistency_report([a.answer(q) for q in rep.PARAPHRASES], rep.EXPECTED_FACT)
            correct_runs += report.correctness == 1.0
        print(f"Across 12 stub seeds, voting gave 100% agreement every time but the voted answer was correct in "
              f"{correct_runs}/12 runs. A wrong majority is made reliably wrong by the cache.")


if __name__ == "__main__":
    main()
