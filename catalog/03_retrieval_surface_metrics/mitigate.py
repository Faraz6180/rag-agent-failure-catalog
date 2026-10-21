"""Mitigation: close vocabulary gaps and gate on the intent metric.

Run:  python catalog/03_retrieval_surface_metrics/mitigate.py

Three parts:
  1. A query expander that maps user vocabulary onto the corpus vocabulary. Here it is a
     small hand-written glossary, standing in for an LLM query rewriter, a synonym table
     mined from search logs, or a hybrid (lexical + dense) retriever.
  2. A reranking step is the other usual fix; it only helps if the answer chunk is inside the
     candidate set, so measure answer rank (see detect.py) before choosing.
  3. The eval set from detect.py as a release gate, so the fix cannot silently regress.

The glossary below is intentionally small. Do not read the post-fix recall as a benchmark: the
eval set and the glossary were written by the same person.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.retrieval_simulator import KeywordRetriever
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")

# user phrase -> corpus vocabulary appended to the query
GLOSSARY = {
    "money back": "refund",
    "weekends": "support hours",
    "okta": "single sign-on SAML SSO",
    "log in": "single sign-on login",
    "calls per minute": "rate limits requests per minute",
    "tier": "plan",
    "audited": "SOC 2 certified security",
    "stop paying": "cancel billing",
}


def expand_query(query: str) -> str:
    lowered = query.lower()
    extra = [target for phrase, target in GLOSSARY.items() if phrase in lowered]
    return query if not extra else f"{query} {' '.join(extra)}"


def main() -> None:
    banner("03 Mitigate: query expansion + eval-set gate")
    retriever = KeywordRetriever()

    before = rep.summarize(rep.evaluate(retriever, k=3))
    after_rows = rep.evaluate(retriever, k=3, rewrite=expand_query)
    after = rep.summarize(after_rows)

    print(f"answer_recall@3 before: {before['answer_recall@k']:.0%}")
    print(f"answer_recall@3 after : {after['answer_recall@k']:.0%}\n")
    rep.print_table(after_rows)

    print()
    status(det.passes_gate(retriever), "release gate before fix")
    status(det.passes_gate(retriever, rewrite=expand_query), "release gate after fix")
    print("\nCheck the other direction too: expansion can pull in off-topic chunks for queries that already worked.")


if __name__ == "__main__":
    main()
