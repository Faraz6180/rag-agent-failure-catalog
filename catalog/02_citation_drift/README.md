# 02 - Citation drift

**One line:** the answer carries citations, but the cited chunk does not support the claim it is attached to.

## What happens

Models are good at producing citation-shaped text. They are less reliable at tracking which chunk a specific claim came from, especially when several chunks cover related topics, when chunk order changes between retrieval and prompt, or when the answer merges facts from multiple chunks. You get two variants:

- **Drift:** the claim is in the context, but the citation points at another chunk.
- **Decoration:** the claim is in no chunk at all, but a citation makes it look sourced.

## Why it is expensive

Citations are a trust signal. Users and reviewers verify one or two, find them correct, and stop checking. A wrong citation is worse than none because it converts an unverified claim into an apparently verified one. In regulated domains the citation *is* the audit trail.

## Reproduce

```bash
python catalog/02_citation_drift/reproduce.py
```

The answer cites `[1]`, `[2]`, `[2]`, `[1]` for four claims. Only the first two are right. The script prints support for each citation.

## Detect

`detect.py` classifies each cited sentence as `supported`, `drift`, `unsupported` or `no_citation` by comparing support against the cited chunk and against all chunks. `evaluate()` scores the verdicts on a six-example set.

Production signals: citation validity rate (share of citations whose chunk supports the claim), share of answer sentences with no citation, and out-of-range citation indices.

## Mitigate

`mitigate.py` has two patterns:

1. `repair_citations` rebinds each claim to its best-supporting chunk and drops unsupported claims.
2. `verify_quotes` requires the model to return a verbatim quote per claim and accepts the claim only if the quote is an exact substring of the cited chunk.

Prefer the second where you can afford a structured output format. Substring checks do not care about topical similarity.

## Limits

- Lexical support cannot distinguish two chunks that discuss the same entity with different values. Use the quote check or an NLI model there.
- Repair can rebind a claim to a chunk that supports it only partially. Keep a threshold and log what you repaired.
- If the model merges two chunks in one sentence, one citation per sentence is too coarse. Ask for claim-level output.
