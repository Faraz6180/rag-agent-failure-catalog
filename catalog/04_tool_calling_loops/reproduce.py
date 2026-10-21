"""Failure 04 - Tool-calling loop that looks productive but never terminates.

Run:  python catalog/04_tool_calling_loops/reproduce.py

Scenario
--------
The agent is asked for a fact the documentation does not contain. A healthy agent would search
once or twice and report "not found". This one keeps reformulating the query. Every step has a
plausible thought ("let me refine my search"), every tool call succeeds, and the logs look like
diligent work. The run ends only when the step budget does.

The policy is scripted because the failure is a property of the *trace*, and the detector and
guard in this entry work on traces from any framework.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Action, AgentState, Trace, run_agent
from common.retrieval_simulator import KeywordRetriever
from common.utils import banner, verdict

TASK = "What percentage discount do Enterprise customers get for annual prepayment?"
_RETRIEVER = KeywordRetriever()

QUERY_VARIANTS = [
    "Enterprise annual prepayment discount percentage",
    "Enterprise prepay discount",
    "annual prepayment discount Enterprise customers",
    "Enterprise discount percentage annual",
    "Enterprise plan prepayment discounts",
    "discount for prepaying annually Enterprise",
]


def search_docs(query: str) -> str:
    """Returns the single best chunk. It always returns *something*, which is what keeps the loop alive."""
    chunk, _ = _RETRIEVER.retrieve(query, k=1)[0]
    return chunk.text


def looping_policy(state: AgentState) -> Action:
    """Never decides the answer is absent; always 'refines' the query."""
    i = len(state.steps)
    return Action(
        thought="The result doesn't quite answer the question. Let me refine my search.",
        tool="search_docs",
        args={"query": QUERY_VARIANTS[i % len(QUERY_VARIANTS)]},
    )


def run_unguarded(max_steps: int = 12, verbose: bool = True) -> Trace:
    return run_agent(TASK, looping_policy, {"search_docs": search_docs}, max_steps=max_steps, verbose=verbose)


def main() -> None:
    banner("04 Tool-calling loop")
    print(f"Task: {TASK}\n")
    trace = run_unguarded()
    print(f"\nstop_reason={trace.stop_reason}  steps={len(trace.steps)}  ~tokens={trace.approx_tokens}  final_answer={trace.final_answer!r}")
    verdict(
        trace.stop_reason == "max_steps",
        "12 successful tool calls cycling through the same few irrelevant chunks; no answer and no 'not found'.",
    )


if __name__ == "__main__":
    main()
