# 09 - Silent tool failure

**One line:** a tool fails or returns nothing useful, and the agent carries on as if it had succeeded.

## What happens

Three common causes:

- **Swallowed exceptions.** A wrapper catches everything and returns `[]`, `""` or `None`. The agent cannot tell "nothing found" from "could not look".
- **Ambiguous success.** The tool returns HTTP 200 with an error field in the body, or a partial result, and the agent reads only the happy path.
- **Error text treated as content.** The agent receives `"Error: timeout"` as a string observation and summarizes it as if it were data.

The model then reasons correctly over wrong inputs. It says "no open orders", "no matching documents", "the migration is complete".

## Why it is expensive

False negatives are the worst class: the user stops looking because the system said there is nothing to find. These failures correlate with outages, which is when traffic and stakes are highest. And because nothing throws, the incident shows up first as a customer complaint.

## Reproduce

```bash
python catalog/09_silent_tool_failure/reproduce.py
```

The order service is down; the wrapper converts the timeout to `[]`; the agent reports that a customer with two open orders has none.

## Detect

Trace analysis cannot distinguish a swallowed error from a real empty result, so `detect.py` uses layers:

1. **Canary:** call each tool with an input that must return data; empty or error means the tool is broken. Run on schedule and at deploy.
2. **Trace audit:** flag runs that make a negative claim from empty results only, or ignore an explicit error. Review candidates, not proof.
3. **Instrumentation:** log every exception inside tool wrappers, even ones you convert to results. This is the cheapest and most reliable layer.

## Mitigate

`mitigate.py` implements the contract fix: `ToolResult` is `ok`, `empty` or `error`, rendered differently (`TOOL_ERROR: ...` vs `TOOL_OK_EMPTY: ...`), and exceptions are logged rather than swallowed. A `FailureAwareGate` rejects final answers that ignore a failed call.

Related design rules: have tools return structured status fields rather than prose, make "no results" an explicit state, and decide per tool whether the agent should retry, escalate or stop.

## Limits

- The scripted policy models how agents commonly treat explicit errors versus ambiguous emptiness. Verify with your own model and prompts.
- Regex-based error detection in `classify_observation` will have false positives (an answer that mentions "failed deployments"). Prefer structured status fields where you control the tool.
- Retrying write tools can duplicate side effects. Make them idempotent first.
