# 05 - Premature termination

**One line:** the agent stops and reports success before the task is actually complete.

## What happens

Agents decide when they are done, and "done" is a judgement call made by the same model that does the work. Common triggers: the first tool result looks like an answer; the task has several parts and the model summarizes the part it did; context fills with intermediate output; the model optimizes for a tidy final message. The final answer is well-formed and confident, so it passes every format check.

## Why it is expensive

It is the agent version of a partially filled form. Reviewers see a clean answer and assume it is complete. Missing items are found later, often by the person who relied on them: the compare-plans summary with no rate limits, the migration plan that skipped a service, the report missing a quarter.

## Reproduce

```bash
python catalog/05_premature_termination/reproduce.py
```

The task asks for three items for two plans. The agent looks up pricing, answers with prices and a recommendation, and stops.

## Detect

`checklist_coverage(answer, checklist)` scores the answer against a "definition of done" written as regexes. Each item is covered only if all its patterns match (for example both plans' values). Coverage below 100% on a run that ended on its own is a candidate for review.

For open-ended tasks, use the same interface with an LLM judge: given the task and the answer, list requested items the answer does not address.

## Mitigate

`CompletionGate` is a final-answer hook. If items are missing it rejects the answer with a message naming them, up to a limit, then halts with an explicit `PARTIAL (missing: ...)` result. Bounding the retries matters; an unbounded gate converts this failure into entry 04.

Where possible, have the planner emit the checklist up front (from the task), so "done" is defined before the work starts.

## Limits

- Regex checklists verify presence, not correctness. Pair with entries 01 and 10 for factual checks.
- If the checklist is derived from the same model that terminates early, it can inherit the same blind spot. Prefer checklists from the task text, a schema, or a separate call.
- The policy is scripted; validate on your own agent's final answers.
