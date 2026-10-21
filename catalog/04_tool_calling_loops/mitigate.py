"""Mitigation: a loop guard that stops the run and says so.

Run:  python catalog/04_tool_calling_loops/mitigate.py

The guard watches every step. When the agent repeats an observation or re-issues near-identical
calls, it halts the run with an explicit INCOMPLETE result that names the reason. The important
design choice is the *shape of the stop*: the caller gets a structured "could not complete"
instead of an empty answer or a fabricated one, so downstream code and humans can tell.

Wire the same logic into any framework's callback or middleware: it only needs the history of
(tool, args, observation).
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


class LoopGuard:
    """Step hook: halt on repeated observations or near-duplicate calls; hard cap on steps."""

    def __init__(self, max_repeats: int = 3, max_steps: int = 8) -> None:
        self.max_repeats = max_repeats
        self.max_steps = max_steps

    def __call__(self, trace: Trace) -> Optional[Intervention]:
        report = det.detect_loop(trace, repeat_threshold=self.max_repeats)
        reasons = [f for f in report.flags if f != "budget_exhausted"]
        if reasons:
            return Intervention(
                "halt",
                f"INCOMPLETE: stopped after {len(trace.steps)} steps without progress "
                f"({', '.join(reasons)}). The requested fact may not exist in the available sources.",
            )
        if len(trace.steps) >= self.max_steps:
            return Intervention("halt", f"INCOMPLETE: step budget of {self.max_steps} reached.")
        return None


def main() -> None:
    banner("04 Mitigate: loop guard")
    print(f"Task: {rep.TASK}\n")
    trace = run_agent(
        rep.TASK, rep.looping_policy, {"search_docs": rep.search_docs}, max_steps=12, step_hook=LoopGuard()
    )
    print(f"\nstop_reason={trace.stop_reason}  steps={len(trace.steps)}  ~tokens={trace.approx_tokens}")
    print(f"result: {trace.final_answer}")
    status(trace.stop_reason == "halted" and len(trace.steps) < 12, "run ended early with an explicit, machine-readable incomplete result")


if __name__ == "__main__":
    main()
