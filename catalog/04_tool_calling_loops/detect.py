"""Detect non-terminating or non-progressing agent loops from a trace.

Run:  python catalog/04_tool_calling_loops/detect.py

Signals, in order of how cheap and reliable they are:

    repeated_observation   the same tool output appears N or more times
    near_duplicate_calls   N or more calls whose arguments are near-identical (token Jaccard)
    low_novelty            few of the recent observations are new
    budget_exhausted       the run ended by hitting the step limit

None of these needs the model's reasoning text, so they work on any framework's trace.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.agent_sim import Action, AgentState, Trace, run_agent
from common.utils import banner, content_tokens, load_sibling, status

rep = load_sibling(__file__, "reproduce")


def _obs_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


@dataclass
class LoopReport:
    flags: list[str]
    repeated_observations: int
    near_duplicate_calls: int
    novelty: float

    @property
    def looping(self) -> bool:
        return bool(self.flags)


def detect_loop(trace: Trace, repeat_threshold: int = 3, jaccard: float = 0.5, window: int = 5) -> LoopReport:
    hashes = [_obs_hash(s.observation) for s in trace.steps]
    repeated = max((hashes.count(h) for h in set(hashes)), default=0)

    sigs = [set(content_tokens(" ".join(str(v) for v in s.args.values()))) for s in trace.steps]
    near_dupes = 0
    for i, sig in enumerate(sigs):
        near_dupes = max(near_dupes, 1 + sum(_jaccard(sig, other) >= jaccard for other in sigs[:i]))

    # Novelty: of the last `window` observations, how many had never been seen before?
    seen: set[str] = set()
    is_new = []
    for h in hashes:
        is_new.append(h not in seen)
        seen.add(h)
    recent = is_new[-window:]
    novelty = sum(recent) / len(recent) if recent else 1.0

    flags = []
    if repeated >= repeat_threshold:
        flags.append("repeated_observation")
    if near_dupes >= repeat_threshold:
        flags.append("near_duplicate_calls")
    if len(trace.steps) >= window and novelty < 0.4:
        flags.append("low_novelty")
    if trace.stop_reason == "max_steps":
        flags.append("budget_exhausted")
    return LoopReport(flags, repeated, near_dupes, novelty)


def _healthy_trace() -> Trace:
    """A trace that makes progress: three different tools, different outputs, then a final answer."""
    plan = [("search_docs", {"query": "pricing"}), ("search_docs", {"query": "rate limits"}), ("search_docs", {"query": "log retention"})]

    def policy(state: AgentState) -> Action:
        if len(state.steps) < len(plan):
            tool, args = plan[len(state.steps)]
            return Action("Gathering facts.", tool, args)
        return Action(final="Done.")

    return run_agent("healthy", policy, {"search_docs": rep.search_docs}, verbose=False)


def main() -> None:
    banner("04 Detect: loop signals on a trace")
    for name, trace in [("looping agent", rep.run_unguarded(verbose=False)), ("healthy agent", _healthy_trace())]:
        r = detect_loop(trace)
        status(not r.looping, name, f"flags={r.flags} repeated_obs={r.repeated_observations} near_dupes={r.near_duplicate_calls} novelty={r.novelty:.2f}")


if __name__ == "__main__":
    main()
