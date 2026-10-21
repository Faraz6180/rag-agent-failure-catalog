"""Mitigation: a completion gate that rejects an unfinished final answer.

Run:  python catalog/05_premature_termination/mitigate.py

When the agent tries to finish, the gate scores the answer against the checklist. If items are
missing it sends the agent back to work with a message naming exactly what is missing, up to a
rejection limit. After the limit it halts with an explicit partial result rather than looping
forever or passing the answer through (the 04 failure in the other direction).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Intervention, Trace, run_agent
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")


class CompletionGate:
    def __init__(self, checklist: dict[str, list[str]], max_rejections: int = 2) -> None:
        self.checklist = checklist
        self.max_rejections = max_rejections
        self.rejections = 0

    def __call__(self, trace: Trace, answer: str) -> Optional[Intervention]:
        cov = det.checklist_coverage(answer, self.checklist)
        if not cov.missing:
            return None
        if self.rejections >= self.max_rejections:
            return Intervention("halt", f"PARTIAL (missing: {', '.join(cov.missing)}): {answer}")
        self.rejections += 1
        return Intervention("feedback", f"Not done. Missing: {', '.join(cov.missing)}. Gather these before finalizing.")


def main() -> None:
    banner("05 Mitigate: completion gate")
    print(f"Task: {rep.TASK}\n")
    trace = run_agent(
        rep.TASK, rep.premature_policy, {"search_docs": rep.search_docs}, max_steps=10, final_hook=CompletionGate(rep.CHECKLIST)
    )
    print(f"\nFinal answer:\n  {trace.final_answer}\n\nstop_reason={trace.stop_reason}, tool calls={len(trace.steps)}")
    status(det.checklist_coverage(trace.final_answer or "", rep.CHECKLIST).ratio == 1.0, "final answer covers every checklist item")

    print("\nWith a gate that allows no retries, the same agent returns an explicit partial instead of a confident one:")
    trace = run_agent(
        rep.TASK, rep.premature_policy, {"search_docs": rep.search_docs}, max_steps=10, final_hook=CompletionGate(rep.CHECKLIST, max_rejections=0), verbose=False
    )
    print(f"  {trace.final_answer}")


if __name__ == "__main__":
    main()
