"""A minimal agent loop for reproducing agent failures without an LLM in the loop.

The agent-level failures in this catalog (loops, premature termination, silent tool failure)
are visible in the *trace*: the sequence of tool calls, observations and the final answer.
Detectors and guards that operate on traces work with any framework, so the examples use
scripted policies (plain functions) that reproduce the patterns seen in real agents.

    policy(state) -> Action        # either a tool call or a final answer
    tools[name](**args) -> str     # observation text
    step_hook(trace)  -> Intervention | None     # runs after every tool call
    final_hook(trace, answer) -> Intervention | None   # runs when the policy wants to finish
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


@dataclass
class Action:
    thought: str = ""
    tool: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)
    final: Optional[str] = None


@dataclass
class Step:
    index: int
    thought: str
    tool: str
    args: dict[str, Any]
    observation: str
    raised: bool = False  # True only if the tool raised; swallowed errors look like normal steps


@dataclass
class Trace:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    stop_reason: str = "running"  # final | max_steps | halted

    @property
    def approx_tokens(self) -> int:
        """Rough cost proxy: ~4 characters per token over thoughts, args and observations."""
        chars = sum(len(s.thought) + len(str(s.args)) + len(s.observation) for s in self.steps)
        return chars // 4


@dataclass
class Intervention:
    kind: str  # "feedback" (continue, tell the policy) | "halt" (stop with this message)
    message: str


@dataclass
class AgentState:
    task: str
    steps: list[Step]
    feedback: list[str]


Policy = Callable[[AgentState], Action]
StepHook = Callable[[Trace], Optional[Intervention]]
FinalHook = Callable[[Trace, str], Optional[Intervention]]


def run_agent(
    task: str,
    policy: Policy,
    tools: Mapping[str, Callable[..., str]],
    max_steps: int = 10,
    step_hook: Optional[StepHook] = None,
    final_hook: Optional[FinalHook] = None,
    verbose: bool = True,
) -> Trace:
    trace = Trace(task=task)
    feedback: list[str] = []

    for _ in range(max_steps):
        action = policy(AgentState(task, list(trace.steps), list(feedback)))

        if action.final is not None:
            iv = final_hook(trace, action.final) if final_hook else None
            if iv and iv.kind == "halt":
                trace.final_answer, trace.stop_reason = iv.message, "halted"
                return trace
            if iv and iv.kind == "feedback":
                feedback.append(iv.message)
                if verbose:
                    print(f"  [gate] final answer rejected: {iv.message}")
                continue
            trace.final_answer, trace.stop_reason = action.final, "final"
            return trace

        raised = False
        try:
            observation = str(tools[action.tool](**action.args))
        except Exception as exc:  # a tool that raises is the *easy* case; see entry 09 for the hard one
            observation, raised = f"ERROR: {type(exc).__name__}: {exc}", True
        step = Step(len(trace.steps) + 1, action.thought, str(action.tool), action.args, observation, raised)
        trace.steps.append(step)
        if verbose:
            short = observation if len(observation) <= 70 else observation[:67] + "..."
            print(f"  step {step.index:>2}: {step.tool}({step.args})  -> {short}")

        iv = step_hook(trace) if step_hook else None
        if iv and iv.kind == "halt":
            trace.final_answer, trace.stop_reason = iv.message, "halted"
            return trace
        if iv and iv.kind == "feedback":
            feedback.append(iv.message)
            if verbose:
                print(f"  [guard] {iv.message}")

    trace.stop_reason = "max_steps"
    return trace
