"""Detect silent tool failures.

Run:  python catalog/09_silent_tool_failure/detect.py

A swallowed error is indistinguishable from a real empty result *in the trace*, so trace analysis
alone is not enough. Use three layers:

  1. canary      call each tool with an input that is known to return data. Empty or error
                 there means the tool (or its wrapper) is broken. Run on a schedule and at deploy.
  2. audit_trace flag runs where the agent makes a negative claim ("no ...", "none") based only
                 on empty observations, or ignores an explicit error. These are review candidates,
                 not proof.
  3. instrumentation  log every exception *inside* tool wrappers, even ones you convert to
                 results. This is the layer that would have caught the bug in reproduce.py.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Step, Trace, run_agent
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")

ERROR_PAT = re.compile(r"\b(error|timeout|timed out|exception|failed|unavailable|5\d\d)\b", re.I)
NEGATIVE_PAT = re.compile(r"\b(no open|none|no results|not found|doesn't have|does not have|has no|there are no)\b", re.I)
ACK_PAT = re.compile(r"\b(couldn't|could not|unable|unavailable|failed|try again)\b", re.I)


def classify_observation(obs: str) -> str:
    text = obs.strip()
    if text.startswith("TOOL_ERROR") or ERROR_PAT.search(text):
        return "error"
    if text in ("", "[]", "{}", "null", "None") or text.startswith("TOOL_OK_EMPTY"):
        return "empty"
    return "ok"


@dataclass
class AuditFinding:
    kind: str
    detail: str


def audit_trace(trace: Trace) -> list[AuditFinding]:
    findings = []
    answer = trace.final_answer or ""
    kinds = [classify_observation(s.observation) for s in trace.steps]
    if "error" in kinds and not ACK_PAT.search(answer):
        findings.append(AuditFinding("error_ignored", "a tool returned an error and the final answer does not mention it"))
    if kinds and all(k == "empty" for k in kinds) and NEGATIVE_PAT.search(answer):
        findings.append(AuditFinding("negative_claim_from_empty", "negative claim based only on empty tool results; verify the tool"))
    return findings


def canary(tool: Callable[..., str], known_good_args: dict) -> bool:
    """True if the tool returns data for an input that must return data."""
    return classify_observation(tool(**known_good_args)) == "ok"


def main() -> None:
    banner("09 Detect: canary + trace audit")
    for down in (False, True):
        tool = rep.make_swallowing_tool(rep.OrderService(down=down))
        status(canary(tool, {"customer_id": "C-1042"}), f"canary on swallowing wrapper (service {'DOWN' if down else 'up'})")

    trace = run_agent(rep.TASK, rep.agent_policy, {"lookup_orders": rep.make_swallowing_tool(rep.OrderService(down=True))}, verbose=False)
    for f in audit_trace(trace):
        status(False, f"audit: {f.kind}", f.detail)
    print("\nThe canary catches the broken tool; the audit alone flags the run for review but cannot prove the cause.")


if __name__ == "__main__":
    main()
