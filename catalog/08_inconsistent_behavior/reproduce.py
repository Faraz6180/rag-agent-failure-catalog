"""Failure 08 - Inconsistent behaviour across similar queries.

Run:  python catalog/08_inconsistent_behavior/reproduce.py [--live]

Scenario
--------
Five phrasings of one question, each asked at temperature 0 and sampled three more times at
temperature 0.7. The documented answer is "30 days". The failure is that the answer a user gets
depends on how they phrased the question and on which sample they happened to draw, and nothing
in a single response reveals that.

Note on temperature 0: it removes sampling noise for a *fixed* prompt. It does not make
different paraphrases give the same answer. The offline stub reflects both effects: at
temperature 0 each paraphrase maps to one fixed answer, and the answers differ across
paraphrases.
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import format_context, get_chunk
from common.utils import banner, base_parser, verdict

PARAPHRASES = [
    "What's the refund window for annual plans?",
    "How long do I have to get my money back on a yearly subscription?",
    "annual plan refund period?",
    "If I buy the annual plan and change my mind, how many days do I have to ask for a refund?",
    "Can I get a refund on a yearly plan? Until when?",
]
EXPECTED_FACT = "30"
SYSTEM = "Answer the question using the context."

# (answer, probability). The first is the documented answer.
VARIANTS = [
    ("Annual plans can be refunded in full within 30 days of purchase.", 0.55),
    ("Annual plans can be refunded within 14 days of purchase.", 0.15),
    ("Refund eligibility depends on your plan and circumstances. Please contact support.", 0.15),
    ("Monthly plans are not refundable, but you can cancel at any time.", 0.15),
]


def _pick(u: float) -> str:
    cumulative = 0.0
    for text, p in VARIANTS:
        cumulative += p
        if u < cumulative:
            return text
    return VARIANTS[-1][0]


def make_stub(seed: int = 1):
    """Handler factory. Temperature 0 -> fixed answer per exact prompt; temperature > 0 -> sampled."""
    state = {"calls": 0}

    def handler(system: str, user: str, temperature: float) -> str:
        question = user.split("Question:")[-1].strip()
        state["calls"] += 1
        if temperature == 0:
            u = int(hashlib.sha1(question.encode()).hexdigest(), 16) % 1000 / 1000
        else:
            u = random.Random(f"{seed}|{question}|{state['calls']}").random()
        return _pick(u)

    return handler


def fact_key(answer: str) -> str:
    """Reduce an answer to the fact a user would act on: the numbers, or 'no-number' if there are none."""
    nums = re.findall(r"\d+", answer)
    return "+".join(nums) if nums else "no-number"


def ask(client: LLMClient, question: str, temperature: float) -> str:
    context = format_context([get_chunk("refund")])
    return client.complete(f"Context:\n{context}\n\nQuestion: {question}", system=SYSTEM, temperature=temperature)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    client = get_client(make_stub(), live=args.live)

    banner(f"08 Inconsistent behaviour   (model: {client.name})")
    keys: list[str] = []
    for q in PARAPHRASES:
        t0 = ask(client, q, 0.0)
        sampled = [ask(client, q, 0.7) for _ in range(3)]
        keys += [fact_key(t0)] + [fact_key(a) for a in sampled]
        print(f"Q: {q}\n   temp 0 : [{fact_key(t0):>9}] {t0}")
        print("   temp .7: " + ", ".join(f"[{fact_key(a)}]" for a in sampled))
    modal = max(set(keys), key=keys.count)
    agreement = keys.count(modal) / len(keys)
    print(f"\n{len(keys)} answers, {len(set(keys))} distinct fact signatures; agreement with the most common = {agreement:.0%}")
    verdict(agreement < 0.8, "the same question gets different facts depending on wording and on the sample drawn.")


if __name__ == "__main__":
    main()
