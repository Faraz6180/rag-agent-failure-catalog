"""Mitigation: verify atoms, regenerate against a facts table, fall back to extraction.

Run:  python catalog/10_numeric_hallucination/mitigate.py [--live]

Ladder, cheapest first:

  1. Draft the answer normally.
  2. Verify every number, date and entity against the context (detect.py). Pass if clean.
  3. If not, regenerate once with an explicit FACTS TABLE extracted from the context and an
     instruction to copy values verbatim. Verify again.
  4. If it still fails, return verbatim excerpts from the source instead of a generated answer.

Step 3 usually helps and sometimes does not; the verification in step 2 and the fallback in
step 4 are what make the pipeline safe. Never skip the second verification.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm_client import LLMClient, get_client
from common.retrieval_simulator import Chunk, format_context
from common.utils import banner, base_parser, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")


@dataclass
class Verified:
    text: str
    path: str  # draft | regenerated | extractive_fallback
    problems: list[str]


def facts_table(context: str) -> str:
    rows = [f"- {a.kind}: {a.value}" for a in sorted(det.extract_atoms(context), key=lambda a: (a.kind, a.value))]
    return "FACTS TABLE (copy values exactly; do not use any value not listed):\n" + "\n".join(rows)


def verify_or_repair(client: LLMClient, task: str, chunks: list[Chunk]) -> Verified:
    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nTask: {task}"

    draft = client.complete(prompt, system=rep.SYSTEM)
    bad = det.unsupported_atoms(draft, context)
    if not bad:
        return Verified(draft, "draft", [])

    retry = client.complete(f"{facts_table(context)}\n\n{prompt}", system=rep.SYSTEM)
    bad_retry = det.unsupported_atoms(retry, context)
    if not bad_retry:
        return Verified(retry, "regenerated", [f"{a.kind}:{a.value}" for a in bad])

    excerpts = "\n".join(f"- {c.text}" for c in chunks)
    return Verified(
        "I couldn't produce a verified summary. Relevant documentation:\n" + excerpts,
        "extractive_fallback",
        [f"{a.kind}:{a.value}" for a in bad_retry],
    )


def main() -> None:
    args = base_parser(__doc__).parse_args()
    banner("10 Mitigate: verify -> regenerate with facts table -> extractive fallback")

    client = get_client(rep.simulated_model, live=args.live)
    result = verify_or_repair(client, rep.QUESTION, rep.CHUNKS)
    status(result.path != "extractive_fallback", f"path={result.path}", f"caught before delivery: {result.problems}")
    print(f"   -> {result.text}\n")

    print("If regeneration also fails (stub that always produces the faulty answer):")
    stubborn = get_client(lambda system, user, temperature: rep.FAULTY)
    result = verify_or_repair(stubborn, rep.QUESTION, rep.CHUNKS)
    status(result.path != "extractive_fallback", f"path={result.path}", f"remaining problems: {result.problems}")
    print(f"   -> {result.text[:140]}...")


if __name__ == "__main__":
    main()
