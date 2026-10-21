# 10 - Numeric, date and entity hallucination that looks plausible

**One line:** the answer is fluent and mostly right, and contains one wrong number, date or name that sits close enough to the truth to pass a skim.

## What happens

Models reproduce the *shape* of facts reliably and the *values* less so. Typical errors are neighbours of the real value: 60 days for 90, 99.99% for 99.9%, a region that follows the naming pattern but does not exist, a date shifted by a few days, a unit swapped. These errors tend to appear when the model summarizes several numbers at once, converts units, or pulls a value from a near-duplicate passage.

## Why it is expensive

- Overlap-based groundedness checks (entry 01) score these answers as highly supported because nearly every word is in the context.
- Reviewers verify the claims that look surprising, not the ones that look normal.
- The cost is concentrated: prices, limits, dates, dosages and legal terms are exactly where the numbers matter.

## Reproduce

```bash
python catalog/10_numeric_hallucination/reproduce.py
```

The context is correct. The summary says 60 days (docs: 90) and adds `us-west-2` (docs list two regions). The script shows overall lexical support is high and lists the terms with no support.

## Detect

`detect.py` extracts atoms (money, percentages, quantities with units, ISO and written dates, region names), normalizes them (`1,000` equals `1000`, `days` equals `day`) and requires every atom in the answer to appear in the context. Units are part of the atom, so `90 hours` does not match `90 days`. `evaluate()` scores the check on eight labeled examples.

Extend `ATOM_PATTERNS` with whatever your users act on: SKUs, versions, plan names, IDs.

## Mitigate

`mitigate.py` implements a ladder: draft, verify atoms, regenerate once against a facts table extracted from the context, verify again, then fall back to verbatim excerpts. The second verification is what makes the pipeline safe: regeneration helps often but not always.

For high-stakes fields, avoid free-text generation of the value at all: extract it with code and let the model write only the surrounding sentence.

## Limits

- "Present in the context" is not "correct for this question". The model can pick the right-looking number from the wrong row (see [02](../02_citation_drift/)). Pair with quote verification.
- Computed values (totals, percentages derived from context numbers) will be flagged because they are not literally present. Either allow-list derived values with a calculator tool or route them to a different check.
- Regex atoms are a baseline. Locale formats (`1.000,50`, `14/03/2025`) need handling for your user base.
