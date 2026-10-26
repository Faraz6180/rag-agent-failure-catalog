# rag-agent-failure-catalog

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Faraz6180/rag-agent-failure-catalog/actions/workflows/ci.yml/badge.svg)

A catalog of common, expensive *silent* failures in RAG systems and AI agents, each with a minimal reproduction, a detector and a mitigation you can lift into your own code.

## Why this exists

Most production incidents in LLM systems are not crashes. The request succeeds, latency is normal, the output is fluent and well-formatted, and it is wrong, incomplete or unsupported. Standard monitoring does not see it: no exception, no status code, no spike. It surfaces later as a customer complaint, a corrected report or a decision made on a number that was never in the source.

These failures share a few traits:

- **They look like success.** The answer is confident and specific.
- **They are statistical.** Same input, different day, different result, which makes them hard to reproduce and easy to dismiss.
- **They cluster in the expensive places:** prices, limits, dates, policies, "no results found".
- **They are caught by evals, not by logs.** If you do not measure them, you do not know your rate.

This repo treats each failure as an engineering object: what it is, why it costs money, how to trigger it on demand, how to detect it, and what a reasonable guardrail looks like.

## The catalog

| # | Failure | One line | Typical signal |
|---|---------|----------|----------------|
| [01](catalog/01_ungrounded_answer/) | Ungrounded answer | Fluent response with no support in the retrieved context | Low sentence-level support against context |
| [02](catalog/02_citation_drift/) | Citation drift | `[1]` is cited, but chunk 1 does not support the claim | Citation validity rate |
| [03](catalog/03_retrieval_surface_metrics/) | Retrieval that looks good but misses intent | On-topic chunks every time; the answering chunk is absent | Answer recall on an intent eval set |
| [04](catalog/04_tool_calling_loops/) | Tool-calling loops | Productive-looking steps that never terminate | Repeated observations, budget exhaustion rate |
| [05](catalog/05_premature_termination/) | Premature termination | Agent stops and reports success before the task is done | Checklist coverage of final answers |
| [06](catalog/06_overconfident_oos/) | Over-confident out-of-scope answers | Confident tone on questions the system cannot answer | Weak evidence + no hedging |
| [07](catalog/07_context_pollution/) | Context pollution / lost in the middle | The answer is in the prompt and the model misses it | Position sweep, pollution ratio |
| [08](catalog/08_inconsistent_behavior/) | Inconsistent behaviour | Similar queries (or repeats) get different facts | Paraphrase/resample agreement |
| [09](catalog/09_silent_tool_failure/) | Silent tool failure | Tool errors or returns empty; the agent carries on | Canary calls, negative claims from empty results |
| [10](catalog/10_numeric_hallucination/) | Numeric / date / entity hallucination | Plausible value that is not in the source | Unsupported atoms (numbers, dates, names) |

Every entry has the same four files:

```
catalog/0X_name/
├── README.md      what it is, why it is expensive, how to detect and mitigate, limits
├── reproduce.py   trigger the failure on demand and print a clear result
├── detect.py      a detector, plus a small evaluation that scores it
└── mitigate.py    a guardrail or pattern, shown before and after
```

## Quick start

Python 3.10+. Offline mode has **no dependencies** and needs no API keys.

```bash
git clone https://github.com/Faraz6180/rag-agent-failure-catalog.git
cd rag-agent-failure-catalog

python catalog/01_ungrounded_answer/reproduce.py     # see the failure
python catalog/01_ungrounded_answer/detect.py        # detect it
python catalog/01_ungrounded_answer/mitigate.py      # guard against it
```

Run all reproductions:

```bash
for f in catalog/*/reproduce.py; do python "$f"; done
```

Each `reproduce.py` ends with `FAILURE REPRODUCED: ...` when the failure was triggered.

### Against a real model

Entries that call a model (01, 02, 06, 07, 08, 10) accept `--live`:

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY or ANTHROPIC_API_KEY
python catalog/07_context_pollution/reproduce.py --live
```

Keys come from environment variables (or `.env`). Nothing is hardcoded. Provider and model are set with `RAG_CATALOG_PROVIDER`, `OPENAI_MODEL` and `ANTHROPIC_MODEL`.

### Tests

```bash
pip install pytest
pytest -q
```

The tests run every script offline and check that each detector and mitigation behaves as its README says.

## What the offline stubs are (and are not)

Silent failures are statistical in real models, so a demo against a live model may or may not show the failure on a given day, and a unit test cannot depend on it. The offline mode therefore uses small, deterministic handlers that **model** each failure: a stub that answers a trial-length question from priors, a toy reader with position bias, a seeded sampler that varies across paraphrases.

What that gives you: reproducible demos, and tests for the detectors and mitigations, which are the real code in this repo.

What it does not give you: evidence about how often any specific model fails. For that, run `--live` on your model with your prompts and your data, and treat the detectors as starting points for your own eval harness.

The agent entries (04, 05, 09) use scripted policies instead of a live agent. Their detectors and guards operate on traces (tool, arguments, observation, final answer), so they port to any framework that exposes those.

Other limits, stated up front:

- Detectors are lexical baselines (token overlap, regex, exact match). They are cheap, deterministic and explainable, and they have false positives and misses. Each README names where an NLI model or LLM judge would replace them, and each `detect.py` includes a small `evaluate()` so you can check a replacement.
- The bundled eval sets are tiny and written by the author. They are regression checks, not benchmarks.
- Thresholds are tuned to keep demos readable. Calibrate on your own labeled data.

## Design principles

1. **Silent first.** If it throws, your monitoring already sees it. The catalog is about outputs that look fine.
2. **Reproduce before you fix.** Every failure can be triggered on demand, in under a second, without credentials.
3. **Detection is the main deliverable.** A mitigation you cannot measure is a hope. Each entry has a detector and a way to score it.
4. **Prefer deterministic checks over more prompting.** Prompts lower the failure rate; gates bound the damage when they fail.
5. **Say what the mitigation costs.** Gates add false abstentions, trimming loses recall, voting hides wrong answers. Each README lists the trade-off.
6. **Minimal dependencies.** Standard library by default. Provider SDKs are imported lazily and only for `--live`.

## Repository layout

```
rag-agent-failure-catalog/
├── catalog/            the ten failure modes
├── common/
│   ├── llm_client.py          provider-agnostic wrapper (OpenAI, Anthropic) + offline stub
│   ├── retrieval_simulator.py deterministic retriever over a small fictional corpus
│   ├── agent_sim.py           minimal agent loop with step and final-answer hooks
│   └── utils.py               tokenization, support scoring, output helpers
├── examples/quickstart.ipynb  walkthrough of three failures
├── tests/                     offline smoke and behaviour tests
├── CONTRIBUTING.md
└── LICENSE
```

## Contributing

New failure modes are welcome, especially ones you have hit in production. See [CONTRIBUTING.md](CONTRIBUTING.md). In short: same four-file folder, offline-first reproduction, a detector with a small evaluation, a mitigation with its trade-offs, and a test.

## Roadmap

Candidates for the next entries, roughly in order of interest:

- Prompt injection through retrieved content
- Stale or superseded documents served as current
- Cross-tenant or permission leakage in retrieval
- Tool argument hallucination (valid call shape, invented IDs)
- Structured-output validity without semantic validity
- Chunk-boundary splits that separate a claim from its qualifier
- Memory drift in long-running agents

Infrastructure:

- Recorded `--live` runs per entry, so results on specific models can be compared over time
- LLM-judge variants of the detectors with agreement numbers against the lexical baselines
- Callback adapters for popular agent frameworks (loop guard, completion gate, failure-aware gate)
- A single `python -m catalog` runner with a summary table

## License

MIT. See [LICENSE](LICENSE).

## Author

Maintained by Faraz, an applied AI engineer working on LLM integration, RAG pipelines and agent orchestration. Issues, corrections and pull requests are welcome, particularly reports of where a detector here fails on real traffic.
