"""Mitigation: typed tool results, plus a gate for answers that ignore failures.

Run:  python catalog/09_silent_tool_failure/mitigate.py

The core fix is a contract: a tool result is exactly one of OK, EMPTY or ERROR, and the three
are rendered differently to the agent. "Nothing found" and "could not look" must never share a
representation.

The second layer is a final-answer gate. If any tool call failed and the answer does not
acknowledge it, the answer is rejected. This protects against policies that ignore errors.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Intervention, Trace, run_agent
from common.utils import banner, load_sibling, status

rep = load_sibling(__file__, "reproduce")
det = load_sibling(__file__, "detect")
log = logging.getLogger("tools")


@dataclass
class ToolResult:
    status: str  # ok | empty | error
    data: Any = None
    error: Optional[str] = None

    def render(self) -> str:
        if self.status == "error":
            return f"TOOL_ERROR: {self.error}"
        if self.status == "empty":
            return "TOOL_OK_EMPTY: the lookup succeeded and returned 0 results"
        return json.dumps(self.data)


def typed_tool(fn: Callable[..., Any]) -> Callable[..., str]:
    """Wrap a function so success, emptiness and failure are distinct, and failures are logged."""

    def wrapper(**kwargs: Any) -> str:
        try:
            data = fn(**kwargs)
        except Exception as exc:
            log.warning("tool %s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
            return ToolResult("error", error=f"{type(exc).__name__}: {exc}").render()
        return ToolResult("empty" if not data else "ok", data=data).render()

    wrapper.__name__ = fn.__name__
    return wrapper


class FailureAwareGate:
    """Final hook: reject answers that ignore a failed tool call."""

    def __call__(self, trace: Trace, answer: str) -> Optional[Intervention]:
        failed = [s for s in trace.steps if det.classify_observation(s.observation) == "error"]
        if failed and not det.ACK_PAT.search(answer):
            return Intervention("feedback", f"Tool {failed[-1].tool} failed; do not state results you could not verify.")
        return None


def main() -> None:
    banner("09 Mitigate: typed results + failure-aware gate")
    logging.basicConfig(level=logging.WARNING, format="  [tool-log] %(message)s", stream=sys.stdout)
    service = rep.OrderService(down=True)

    def lookup_orders(customer_id: str):
        return service.lookup(customer_id)

    tools = {"lookup_orders": typed_tool(lookup_orders)}

    print("A) typed tool, agent that heeds explicit errors")
    trace = run_agent(rep.TASK, rep.agent_policy, tools, max_steps=6, final_hook=FailureAwareGate())
    print(f"   -> {trace.final_answer}")
    status("couldn't" in (trace.final_answer or ""), "no false negative; failure surfaced to the user")

    print("\nB) typed tool, agent that ignores errors, gate on")
    ignoring = lambda state: rep.agent_policy(state, heeds_errors=False)  # noqa: E731
    trace = run_agent(rep.TASK, ignoring, tools, max_steps=6, final_hook=FailureAwareGate())
    print(f"   -> {trace.final_answer}")
    status("couldn't" in (trace.final_answer or ""), "gate rejected the false negative and forced an honest answer")

    print("\nC) same tools once the service is healthy")
    service.down = False
    trace = run_agent(rep.TASK, rep.agent_policy, tools, max_steps=6, final_hook=FailureAwareGate(), verbose=False)
    print(f"   -> {trace.final_answer}")


if __name__ == "__main__":
    main()
