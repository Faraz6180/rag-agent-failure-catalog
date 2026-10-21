# 07 - Context pollution and lost-in-the-middle

**One line:** the answer is in the prompt, but extra context around it, or its position within the context, makes the model miss it.

## What happens

Two related effects:

- **Pollution:** each irrelevant chunk is a chance for the model to latch onto something plausible but wrong, and a dilution of the relevant evidence.
- **Position sensitivity:** many models use information at the start and end of a long context more reliably than information in the middle. Whether this affects *your* model depends on the model, the prompt and the task.

Both fail quietly. The model does not error; it answers from what it noticed, or says the documentation does not cover something it does.

## Why it is expensive

- It scales with the habit of "retrieve more to be safe". Teams raise k to fix recall and quietly buy a precision problem.
- It makes behaviour depend on retrieval order, which nobody is checking.
- It shows up as flaky quality that disappears when you debug with a smaller context.

## Reproduce

```bash
python catalog/07_context_pollution/reproduce.py          # toy position-biased reader
python catalog/07_context_pollution/reproduce.py --live   # measure a real model
```

Offline, the reader is a toy that models strong edge attention. It demonstrates the *test*, not the size of the effect on any real model. The `--live` run is what you should trust: the same 12 chunks, the answer chunk moved through every position, three repeats per position.

## Detect

`detect.py` provides:

- `sensitivity(acc)`: edge-vs-middle accuracy gap from a position sweep. Run this on your actual model and prompt template.
- `pollution_ratio(chunks, relevant_ids)`: share of context tokens that are irrelevant. Log it per request.

## Mitigate

`mitigate.py` implements a pipeline: dedupe near-identical chunks, rank, trim to `k` and a token budget, then order so the best evidence sits at the edges.

The trade-off is recall: if `k` is too small the answer chunk is dropped. Choose `k` using answer recall on an eval set (see [03](../03_retrieval_surface_metrics/)), and re-run the sweep when you change models.

## Limits

- The position effect varies by model and version, and may be small or absent for your setup. That is a valid result; keep the sweep as a regression test.
- Edge ordering helps only if ranking is good. It amplifies a bad ranker.
- For tasks that need many chunks (summarization, comparison), trimming is the wrong tool; consider map-reduce or per-chunk extraction.
