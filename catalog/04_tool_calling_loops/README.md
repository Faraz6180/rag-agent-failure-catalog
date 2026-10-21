# 04 - Tool-calling loops that look productive

**One line:** the agent keeps calling tools with slightly different arguments, every call succeeds, and the run never reaches an answer.

## What happens

When the requested information does not exist (or the tool cannot surface it), a model that treats "not found" as failure will keep refining. Each step comes with a reasonable-sounding thought and a successful tool call, so nothing trips an error handler. The run ends when the step or token budget does, and the caller gets an empty or truncated result, or worse, whatever the last intermediate message was.

## Why it is expensive

- Token and API cost scales with steps, and these runs always consume the full budget.
- The failure is not an exception, so it is missing from error dashboards.
- Users wait for the whole run, then get nothing useful.
- In agents with write tools, a loop can repeat a side effect.

## Reproduce

```bash
python catalog/04_tool_calling_loops/reproduce.py
```

The task asks for a discount percentage the docs do not contain. The scripted policy re-queries 12 times; the retriever returns a chunk each time (cycling through the same three irrelevant ones), so every step "succeeds".

## Detect

`detect_loop(trace)` returns flags from the trace alone:

- `repeated_observation`: identical tool output N+ times.
- `near_duplicate_calls`: N+ calls with near-identical arguments (token Jaccard).
- `low_novelty`: few new observations in the recent window.
- `budget_exhausted`: the run ended by hitting the step cap.

Log these per run. The rate of `budget_exhausted` alone is a useful health metric.

## Mitigate

`LoopGuard` is a step hook that halts the run on repetition or budget and returns an explicit `INCOMPLETE:` result with the reason. Choose the stop shape deliberately: structured "could not complete" beats silence, and beats a fabricated answer.

Also consider: telling the agent in its prompt that "not found" is an acceptable final answer, and giving it a tool-level way to say so.

## Limits

- Legitimate multi-step work also revisits similar queries. Tune thresholds on traces of tasks that succeeded.
- Exact-hash repetition misses loops where observations differ slightly each time (timestamps, ordering). The Jaccard and novelty signals cover some of that.
- The policy here is scripted. The detector is meant to be validated on your own agent's traces.
