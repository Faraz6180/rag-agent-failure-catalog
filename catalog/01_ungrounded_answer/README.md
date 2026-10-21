# 01 - Ungrounded answer

**One line:** the model returns a fluent, specific answer that nothing in the retrieved context supports.

## What happens

Retrieval always returns the top-k nearest chunks, even when none of them contain the answer. If the prompt says "answer using the context" but gives the model no cheap way out, the model fills the gap from its priors. The result reads like documentation: specific numbers, confident tone, no hedging.

## Why it is expensive

- It is invisible to users and to most dashboards. Latency, error rate and "answered" rate all look healthy.
- The invented details are the kind people act on: refund windows, trial lengths, limits, prices.
- Support, legal and finance teams pay for it later, usually as a customer-facing correction.

## Reproduce

```bash
python catalog/01_ungrounded_answer/reproduce.py          # offline, deterministic
python catalog/01_ungrounded_answer/reproduce.py --live   # real model
```

The corpus has no free-trial information. The stub answers "14-day free trial, no credit card" anyway, and the script shows which answer terms have no support in the retrieved context.

## Detect

`detect.py` splits the answer into sentences and scores each against the context by weighted token coverage (numbers count double). Sentences below `0.8` are flagged, as is any sentence containing a number the context never mentions. `evaluate()` reports precision and recall on a ten-example labeled set. The set is tiny and written by the author; treat the numbers as a regression check for the detector, not as a benchmark.

Signals worth logging in production:

- Fraction of answer sentences below the support threshold.
- Answers containing numbers or named entities absent from the context (see [10](../10_numeric_hallucination/)).
- Low top-1 retrieval score combined with a non-abstaining answer (see [06](../06_overconfident_oos/)).

## Mitigate

`mitigate.py` layers two controls:

1. A prompt that allows abstention and defines the sentinel (`INSUFFICIENT_CONTEXT`).
2. A deterministic gate that checks the draft against the context and substitutes an explicit "couldn't find it" response.

The prompt reduces the rate. The gate bounds the damage when the prompt fails.

## Limits

- The lexical check is a baseline. It misses unsupported claims phrased with words that happen to appear in the context, and it can flag valid paraphrases. Swap `support_detail` for an NLI model or LLM judge and compare with `evaluate()`.
- A gate adds false abstentions. Track them alongside the failures you catch.
- Grounded is not the same as correct: if the corpus is wrong or stale, a grounded answer is wrong too.
