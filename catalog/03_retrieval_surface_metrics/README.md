# 03 - Retrieval that looks good but misses intent

**One line:** the retriever returns plausible, on-topic chunks every time, and the chunk that answers the question is not among them.

## What happens

Surface retrieval metrics (similarity scores, "returned k results", topical keyword overlap, even high average relevance on a generic benchmark) measure whether the results *resemble* the query. User intent is whether the results *contain the answer*. They diverge when:

- users and documentation use different vocabulary ("money back" vs "refund");
- many chunks share the head terms of the query (every billing chunk mentions "plan");
- the answer lives in a chunk whose text is short on the query's words.

The generator then does one of two things: abstains (an annoying but visible failure) or answers from priors (see [01](../01_ungrounded_answer/)).

## Why it is expensive

Retrieval quality is usually measured once, on a generic benchmark, and then assumed. Meanwhile the real failure rate is concentrated in the long tail of real phrasings, which is exactly where users are most likely to be new, frustrated or in a hurry. It also hides behind generation: when the answer is wrong, the instinct is to tune the prompt, not the retriever.

## Reproduce

```bash
python catalog/03_retrieval_surface_metrics/reproduce.py
```

Six user phrasings are run through a lexical retriever. The table shows that every query returns `k` results and the top chunk always shares a keyword, while three of six miss the chunk that answers them. In real dense retrieval the similarity scores are compressed into a high band regardless of relevance, so a threshold-style score check shows the same blindness.

## Detect

`detect.py` runs the answerability eval set: each case is a real phrasing plus a fact that must appear in the retrieved context. It reports `answer_recall@k` and, for each miss, the rank the answer chunk had over the whole corpus. That rank tells you whether to fix candidate generation (answer not in top 50) or ranking (answer at rank 5, k = 3).

How to build the set: sample production queries, have someone mark the passage that answers each, and keep the phrase. Fifty to a hundred cases cover a lot.

## Mitigate

`mitigate.py` adds a query expander and uses the eval set as a release gate. The glossary is a stand-in; in production the equivalent is hybrid retrieval, an LLM query rewriter, or a synonym table mined from search logs. Whatever you choose, measure it on the eval set, including queries that already worked.

## Limits

- The eval set is the whole detector. If it is not drawn from real traffic, it will not find real failures.
- The demo's recall numbers are not a benchmark; the glossary and the eval set share an author.
- Query expansion can hurt precision. Check both directions.
