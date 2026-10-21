# 08 - Inconsistent behaviour across similar queries

**One line:** two users ask the same thing in different words, or one user asks twice, and they get different facts.

## What happens

- **Sampling variance:** at temperature > 0 the same prompt yields different answers.
- **Paraphrase sensitivity:** at temperature 0 the same prompt yields the same answer, but different phrasings of one question can yield different answers. Retrieval shifts with the wording, and the model reacts to surface form.
- **Context order and non-determinism:** hosted models are not guaranteed bit-identical at temperature 0, and ties in retrieval can reorder chunks.

No single response shows the problem. You only see it by comparing responses.

## Why it is expensive

Consistency is what users read as reliability. Two support agents (or one bot at two moments) giving different refund windows produces escalations, and makes any single wrong answer impossible to reproduce and fix. It also quietly breaks evaluation: a one-shot eval of a stochastic system is a coin flip.

## Reproduce

```bash
python catalog/08_inconsistent_behavior/reproduce.py          # offline, seeded
python catalog/08_inconsistent_behavior/reproduce.py --live   # real model
```

Five paraphrases, each at temperature 0 and three samples at 0.7. The script reduces each answer to its fact signature and prints agreement. The offline stub models both effects deliberately; the live run measures yours. Strong modern models may show high agreement on this easy question. Use harder, more ambiguous questions from your own logs.

## Detect

`consistency_report(answers, expected_fact)` returns two numbers:

- **agreement:** share of answers matching the most common fact signature;
- **correctness:** share matching the documented fact.

Track both. Agreement alone approves a system that is consistently wrong.

Run it as a nightly job over your top intents, and on every model or prompt change.

## Mitigate

`StableAnswerer` canonicalizes recognized intents, samples K answers and returns the majority, then caches by intent. This trades a small amount of latency and cost for stable output.

This is a consistency fix, not a correctness fix. If the modal answer is wrong, you have made it reliably wrong; keep reviewing cached answers for high-stakes intents. The offline demo makes this concrete: across 12 stub seeds, 5-vote majority was correct in only about two thirds of runs, while agreement was 100% in all of them.

## Limits

- Canonicalization by rules does not scale. Use embeddings or an LLM rewrite to map paraphrases to intents, and monitor misroutes.
- Fact extraction is task-specific. Numbers work for this example; yes/no, entities and structured fields each need their own extractor.
- Caching by intent hides genuine changes in the corpus. Invalidate on document updates.
