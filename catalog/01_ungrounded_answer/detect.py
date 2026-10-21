"""Detect ungrounded sentences by checking each one against the retrieved context.

Run:  python catalog/01_ungrounded_answer/detect.py

Approach: split the answer into sentences and measure how much of each sentence's content
(numbers weighted double) is present in the context. Low coverage, or any number that the
context never mentions, means the sentence is probably coming from the model rather than
the documents.

Limits: this is lexical. It misses paraphrased unsupported claims and flags legitimate
paraphrases that share few words with the source. For production, use the same interface
with an NLI model or an LLM judge in place of `support_detail`. The evaluation harness at
the bottom tells you whether the swap helped.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import get_chunk
from common.utils import banner, load_sibling, split_sentences, status, support_detail

THRESHOLD = 0.8


@dataclass
class Finding:
    sentence: str
    score: float
    missing: list[str]
    flagged: bool


def find_unsupported(answer: str, context: str, threshold: float = THRESHOLD) -> list[Finding]:
    findings = []
    for sentence in split_sentences(answer):
        score, missing = support_detail(sentence, context)
        # A number the context never mentions is flagged regardless of overall overlap:
        # "Pro allows 5,000 requests per minute" shares most of its words with the real answer.
        missing_number = any(ch.isdigit() for tok in missing for ch in tok)
        findings.append(Finding(sentence, score, missing, score < threshold or missing_number))
    return findings


def is_ungrounded(answer: str, context: str, threshold: float = THRESHOLD) -> bool:
    return any(f.flagged for f in find_unsupported(answer, context, threshold))


# ---- tiny labeled set: (context chunk ids, answer, is_ungrounded) ------------------------------
LABELED = [
    (["refund"], "Annual plans can be refunded within 30 days of purchase.", False),
    (["refund"], "Annual plans are refundable for 60 days, and monthly plans get a prorated credit.", True),
    (["sla"], "The Pro plan includes a 99.9% uptime SLA.", False),
    (["sla"], "The Pro plan includes a 99.99% uptime SLA with automatic service credits.", True),
    (["rate-limits"], "Starter allows 100 requests per minute and Pro allows 1,000.", False),
    (["rate-limits"], "Pro allows 5,000 requests per minute and bursts are free.", True),
    (["sso"], "SAML SSO is only available on Enterprise plans.", False),
    (["sso"], "SAML SSO is available on Pro plans for an extra fee.", True),
    (["regions"], "Acme Cloud runs in us-east-1 and eu-west-1.", False),
    (["regions"], "Acme Cloud runs in us-east-1, eu-west-1 and ap-south-1.", True),
]


def evaluate(threshold: float = THRESHOLD) -> dict[str, float]:
    """Precision / recall of the detector on the labeled set (positive class = ungrounded)."""
    tp = fp = fn = tn = 0
    for chunk_ids, answer, label in LABELED:
        context = "\n".join(get_chunk(c).text for c in chunk_ids)
        predicted = is_ungrounded(answer, context, threshold)
        tp += predicted and label
        fp += predicted and not label
        fn += (not predicted) and label
        tn += (not predicted) and not label
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": precision, "recall": recall, "n": len(LABELED)}


def main() -> None:
    banner("01 Detect: sentence-level groundedness check")
    rep = load_sibling(__file__, "reproduce")
    from common.retrieval_simulator import KeywordRetriever

    answer, context = rep.naive_rag_answer(
        rep.get_client(rep.simulated_model), rep.QUESTION, KeywordRetriever()
    )
    for f in find_unsupported(answer, context):
        status(not f.flagged, f.sentence, f"support={f.score:.2f}, missing={f.missing}")

    metrics = evaluate()
    print(f"\nEval on {metrics['n']} labeled answers: precision={metrics['precision']:.2f} recall={metrics['recall']:.2f}")


if __name__ == "__main__":
    main()
