# Contributing

Thanks for considering a contribution. The most useful additions are failure modes you have seen in a real system, with a detector that worked (or an honest account of why it did not).

## What belongs in the catalog

A failure mode qualifies if it is:

- **Silent:** the system returns a normal-looking response; no exception or error status.
- **Costly:** it affects decisions, money, trust or safety, not just style.
- **Reproducible:** you can trigger it on demand, or model it faithfully enough to test a detector.
- **Detectable:** there is a measurable signal, even if imperfect.

## Folder structure

Create `catalog/NN_short_name/` (next free number, lowercase, underscores) with exactly these files:

```
catalog/NN_short_name/
├── README.md
├── reproduce.py
├── detect.py
└── mitigate.py
```

### README.md

Use the same sections as the existing entries: one-line summary, what happens, why it is expensive, reproduce, detect, mitigate, limits. The **Limits** section is required. State where the detector misses, where the mitigation costs something, and what you did not test.

### reproduce.py

- Runs as `python catalog/NN_short_name/reproduce.py` with no setup and no network.
- Starts with the path bootstrap used by the other entries so `common` imports resolve.
- Deterministic: fixed seeds, scripted policies or stub handlers. If the failure is model behaviour, ship an offline handler that models it and support `--live` (`common.utils.base_parser`) so people can test a real model.
- Prints the evidence, then ends with `verdict(reproduced, "...")` from `common.utils`. The test suite looks for `FAILURE REPRODUCED`.
- Comment the *why* of the scenario, and say what the stub models and what it does not.

### detect.py

- Exposes a function that takes the inputs and returns findings, so it can be reused outside the demo.
- Includes `evaluate()` over a small labeled set, returning numbers (precision/recall, accuracy, agreement). Keep the set small and say in the README that it is a regression check, not a benchmark.
- Explain what the detector cannot see.

### mitigate.py

- Shows before and after on the same scenario.
- Prefers a deterministic gate or contract change over "improve the prompt". A prompt can be part of it, not all of it.
- States the cost: false abstentions, lost recall, latency, hidden wrong answers.

## Code standards

- Python 3.10+, type hints where they help, docstrings on public functions.
- Comments explain *why*, not *what*.
- Standard library first. Add a dependency only when an example cannot work without it, and import it lazily.
- API keys only from environment variables. Never commit `.env`. Do not paste real keys, customer data or proprietary text into examples; use the fictional Acme Cloud corpus or invent a small one.
- Tone in docs: direct and specific. No hype, no claims about how often a model fails unless you measured it and say how.

## Tests

Add a test to `tests/test_catalog.py` that checks the one or two behaviours your README promises (the detector flags the failure, the mitigation fixes it). The parametrized smoke tests pick up your folder automatically.

```bash
pip install pytest
pytest -q
```

## Pull request checklist

- [ ] Folder has all four files and follows the naming pattern
- [ ] `reproduce.py` runs offline and prints `FAILURE REPRODUCED`
- [ ] `detect.py` has `evaluate()` and the README states its limits
- [ ] `mitigate.py` shows before/after and the README states the cost
- [ ] The failure is added to the table in the main README
- [ ] `pytest -q` passes
- [ ] No secrets, no real user data

## Reporting problems with existing entries

If a detector gives a false positive or misses a case on real data, open an issue with the (sanitized) input and output. Those reports are more valuable than new entries.
