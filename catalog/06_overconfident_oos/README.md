# 06 - Over-confident answers on out-of-scope or ambiguous questions

**One line:** when the right response is "I don't know" or "which one do you mean?", the system answers anyway, in the same assured tone it uses for questions it can answer.

## What happens

Three sub-cases look identical to the user:

- **Out-of-scope:** the knowledge base does not cover the topic (free trial, competitor comparisons, general advice).
- **Ambiguous:** the question is under-specified ("my plan", "the limit") and the answer differs by the missing detail. The model silently picks one.
- **Opinion/advisory:** the question invites a judgement the product should not be making.

Models are trained to be helpful and fluent. Fluency is not calibrated to evidence, so confidence in the *tone* carries no information about how much support the answer has.

## Why it is expensive

Users calibrate on tone. An unhedged wrong answer gets acted on. For ambiguous questions the answer is often correct for *some* plan, which makes it look correct in review and wrong in the case that matters.

## Reproduce

```bash
python catalog/06_overconfident_oos/reproduce.py
```

Six questions, one tone. The script prints top-1 retrieval score next to each answer: the in-scope questions score about 0.8 or above, the out-of-scope ones below 0.4, and the answers do not change their tone at all.

## Detect

`detect.py` combines two signals: weak retrieval evidence (top-1 score below a threshold) and ambiguity (no plan named in the question, but the best chunk differs by plan), and flags answers with neither. `evaluate()` reports catch rate on out-of-scope/ambiguous questions and false-flag rate on in-scope ones.

The threshold here is tuned on six examples for readability. In production, calibrate it on labeled traffic, and track both rates: a gate that abstains on everything has a perfect catch rate.

## Mitigate

`ScopeGate` decides `answer`, `clarify` or `abstain` before generation. Gating beforehand is cheaper than post-hoc confidence estimation and keeps the behaviour inspectable. Log every non-answer decision; it tells you which questions your docs do not cover.

## Limits

- Retrieval score is a proxy for coverage. With dense retrievers, scores are compressed and thresholds need per-corpus calibration.
- The ambiguity rule is a hand-written heuristic for one domain. Generalize with a classifier or an LLM call that lists the missing parameters.
- Over-gating is its own failure. Measure false abstentions.
