"""Failure 05 - Premature termination.

Run:  python catalog/05_premature_termination/reproduce.py

Scenario
--------
The task has three parts: price, rate limit and log retention for two plans. The agent looks up
pricing, writes a clean, confident summary of that, and stops. Nothing is wrong with what it
said. It just said a third of what was asked, in a tone that signals completion.

Policy is scripted; the completion check in this entry operates on any final answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Action, AgentState, Trace, run_agent
from common.retrieval_simulator import KeywordRetriever
from common.utils import banner, verdict

TASK = "For Starter and Pro, give the monthly price, the API rate limit and the log retention period."

# What a complete answer must contain, as regexes. This checklist is the "definition of done".
CHECKLIST = {
    "price": [r"\$29", r"\$99"],
    "rate limit": [r"\b100\b", r"1,?000"],
    "log retention": [r"\b30 days", r"\b90 days"],
}

_RETRIEVER = KeywordRetriever()

PARTIAL = "Starter is $29 per month and Pro is $99 per month. Pro is the better fit for growing teams."
COMPLETE = (
    "Starter: $29 per month, 100 requests per minute, logs kept 30 days. "
    "Pro: $99 per month, 1,000 requests per minute, logs kept 90 days."
)


def search_docs(query: str) -> str:
    return _RETRIEVER.retrieve(query, k=1)[0][0].text


def premature_policy(state: AgentState) -> Action:
    """Stops after the first lookup. If a gate objects, it does the missing work and tries again."""
    searched = " ".join(s.args.get("query", "") for s in state.steps).lower()
    if not state.steps:
        return Action("I'll start with pricing.", "search_docs", {"query": "Starter Pro pricing"})
    if not state.feedback:
        return Action(final=PARTIAL)
    needed = state.feedback[-1].lower()
    if "rate limit" in needed and "rate limit" not in searched:
        return Action("Need the rate limits.", "search_docs", {"query": "API rate limits"})
    if "log retention" in needed and "retention" not in searched:
        return Action("Need log retention.", "search_docs", {"query": "log retention"})
    return Action(final=COMPLETE)


def run_unguarded() -> Trace:
    return run_agent(TASK, premature_policy, {"search_docs": search_docs}, max_steps=8)


def main() -> None:
    banner("05 Premature termination")
    print(f"Task: {TASK}\n")
    trace = run_unguarded()
    print(f"\nFinal answer:\n  {trace.final_answer}\n\nstop_reason={trace.stop_reason}, tool calls={len(trace.steps)}")
    verdict(
        trace.stop_reason == "final",
        "the agent reported success after covering 1 of 3 requested items; nothing flagged it as incomplete.",
    )


if __name__ == "__main__":
    main()
