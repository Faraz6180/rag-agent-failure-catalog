"""Failure 09 - Silent tool failure.

Run:  python catalog/09_silent_tool_failure/reproduce.py

Scenario
--------
The order service is down. The tool wrapper was written to be "robust": it catches every
exception and returns an empty list. The agent receives "[]", which is also what a customer
with no open orders looks like, and confidently reports that there are none. Nothing raised,
nothing was logged, and the answer is false.

Policy is scripted; the point is the contract between tool and agent, which is a property of
your code, not the model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Action, AgentState, Trace, run_agent
from common.utils import banner, verdict

TASK = "Does customer C-1042 have any open orders?"
ORDERS = {
    "C-0001": [{"id": "A-12", "status": "shipped"}],
    "C-1042": [{"id": "A-77", "status": "open"}, {"id": "A-81", "status": "open"}],
}


class OrderService:
    """Stand-in for a flaky upstream dependency."""

    def __init__(self, down: bool = False) -> None:
        self.down = down

    def lookup(self, customer_id: str) -> list[dict]:
        if self.down:
            raise TimeoutError("upstream order service timed out after 5s")
        return ORDERS.get(customer_id, [])


def make_swallowing_tool(service: OrderService):
    """The bug: every failure becomes an empty result. Common in 'defensive' wrappers."""

    def lookup_orders(customer_id: str) -> str:
        try:
            return json.dumps(service.lookup(customer_id))
        except Exception:
            return "[]"

    return lookup_orders


def agent_policy(state: AgentState, heeds_errors: bool = True) -> Action:
    """Reads observations the way agents commonly do: an explicit error gets a retry and an honest
    report; an empty result is taken at face value."""
    if not state.steps:
        return Action("Check the customer's orders.", "lookup_orders", {"customer_id": "C-1042"})
    errors = [s for s in state.steps if s.observation.startswith("TOOL_ERROR")]
    last = state.steps[-1].observation.strip()
    if errors and heeds_errors and not state.feedback:
        if len(errors) < 2:
            return Action("The tool reported an error; retrying once.", "lookup_orders", {"customer_id": "C-1042"})
        return Action(final="I couldn't check C-1042's orders: the order service is unavailable. Please try again shortly.")
    if state.feedback:  # a gate objected to an answer that ignored an error
        return Action(final="I couldn't verify C-1042's orders because the order service failed. Please try again shortly.")
    if last in ("[]", "") or last.startswith("TOOL_OK_EMPTY"):
        return Action(final="Customer C-1042 has no open orders.")
    return Action(final=f"Customer C-1042 has open orders: {last}")


def main() -> None:
    banner("09 Silent tool failure")
    print(f"Task: {TASK}   (the order service is DOWN; C-1042 really has 2 open orders)\n")
    trace = run_agent(TASK, agent_policy, {"lookup_orders": make_swallowing_tool(OrderService(down=True))}, max_steps=6)
    print(f"\nFinal answer: {trace.final_answer}")
    verdict(
        "no open orders" in (trace.final_answer or ""),
        "the tool failed, returned [], and the agent asserted a false negative with full confidence.",
    )


if __name__ == "__main__":
    main()
