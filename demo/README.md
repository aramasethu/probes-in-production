# Truthfulness linear-probe demo

A minimal, end-to-end demonstration that a **linear read** of a small instruct
model's residual stream recovers whether a factual statement is true or false —
and that this signal is **localized** to particular layers.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (open weights, 28 layers), running on Apple
Silicon (`mps`). float32, last-token hidden state, raw declarative statements.

## Pipeline

```bash
cd demo
../.venv/bin/python 01_make_data.py     # build balanced true/false dataset
../.venv/bin/python 02_extract.py        # forward pass -> last-token hidden states, all layers
../.venv/bin/python 03_sweep.py          # one probe per layer x 10 entity-grouped splits, plot mean ± SE
../.venv/bin/python 04_live.py           # probe fresh held-out statements
../.venv/bin/python 04_live.py generate  # generate -> hidden state from the same generate() call -> probe
../.venv/bin/python 04_live.py "Your own statement here."
../.venv/bin/python 05_cost.py           # benchmark what the probe costs at inference
```

Each step writes artifacts the next one reads:

| file | produced by | contents |
|------|-------------|----------|
| `statements.jsonl` | `01` | `{statement, label, kind, group}`, balanced 50/50; `group` = entity (split unit) |
| `activations.npz`  | `02` | `X [n, 29, 1536]` last-token vectors, `y`, `kind` |
| `layer_sweep.png`  | `03` | accuracy-vs-layer, mean ± SE band over 10 entity-grouped splits |
| `best_probe.joblib`| `03` | probe at the layer chosen by inner (train-only) validation, refit on all data |

## What the result shows

**Evaluation protocol.** 312 statements in 106 entity groups (a country's
capital + continent rows, an element's twin rows, a number pair's twin rows).
Splits are by group (`GroupShuffleSplit`, 30% of groups held out), so a
statement's matched true/false twin is never on the other side of the split.
Repeated over 10 splits; the layer is chosen by 5-fold `GroupKFold` *inside the
training set*, never on test.

**Layer sweep** (held-out accuracy, mean ± SE over 10 splits, test n ≈ 96, chance = 0.50):

- Layer 0: 0.50 (the last token is always ".", so the embedding carries nothing).
- Layers 1–6: 0.53–0.70 — above chance, but driven by surface cues in the
  dataset (e.g. "Europe" is the true continent 21× vs false 4×; the
  `compare` slice is already 0.86 by layer 4), not by the truth computation.
- Layers 7–12: sharp climb (0.80 → 0.90).
- **Layers 13–28: plateau at 0.91–0.92**, every layer within 0.02 of the best
  mean and within split-to-split noise of each other. There is no
  distinguishable single peak layer.
- **Nested estimate** (layer picked by inner validation, then scored once on
  test): **0.918 ± 0.005**. Inner validation picked layers 14–26 across splits;
  layer 15 most often, so that is the saved probe.

This rise-then-plateau is the point: the "is this true?" signal becomes
linearly readable in the middle of the network and stays readable from there to
the output. A linear probe suffices to read it.

**Per-category accuracy** at each split's chosen layer, pooled over the 10
splits (`n_unique` = distinct statements ever in a test set; SE uses it):

| category   | accuracy | ± SE | n_unique (groups) |
|------------|----------|------|-------------------|
| capital    | 1.00 | 0.01 | 98 (50) |
| continent  | 0.95 | 0.02 | 98 (50) |
| element    | 0.95 | 0.04 | 32 (16) |
| compare    | 0.95 | 0.04 | 38 (20) |
| arith      | 0.60 | 0.08 | 40 (20) |

Truth-as-a-direction is cleanest for **factual recall the model reliably knows**
(capitals, continents, element symbols). Arithmetic truth is encoded far less
linearly — a useful caveat: a truth probe reads what the model *represents as
true*, which tracks knowledge, not ground truth. Small slices (element,
compare, arith: 16–20 groups each) carry ±0.04–0.08 uncertainty.

## The closed loop (`04_live.py generate`)

The model generates a completion and the probe reads the layer-15 hidden state
**out of the same `generate()` call** (`output_hidden_states=True,
return_dict_in_generate=True`) at the last generated content token (the
closing "."). No extra model call. (Verified: identical, to 1e-6 relative error,
to a teacher-forced forward pass over prompt + completion.)

**Caveat this surfaced:** the probe was trained on raw statements, and it does
**not** transfer as-is to the assistant-turn context — in the chat template it
says FALSE to everything (accuracy 0.50 on the training statements re-framed as
assistant replies), even though its logit still ranks true above false (AUC
0.95). A probe trained on in-context activations reaches 0.93 ± 0.01 on the
same grouped splits. So the closed loop needs a probe trained in the
deployment context; the generate mode also prints a labelled comparison score
from a *second* forward pass on the generated text alone, which is the
in-distribution (raw-statement) reading.

## What it costs at inference (`05_cost.py`)

The probe is cheap to *train*; the talk-worthy question is what it costs to
*run* alongside generation. Three things are measurable on a laptop (MPS), no
GPU/vLLM required. All numbers below are from `Qwen2.5-1.5B-Instruct` on an
M-series Mac: medians (IQR in `05_cost.py` output), `torch.mps.synchronize()`
around every measured region, variants interleaved, overheads with 95%
bootstrap CIs (decode loop: 20 runs per variant).

| cost | measured | meaning |
|------|----------|---------|
| `output_hidden_states=True` | −0.2% [95% CI −1.0, +0.8] — **within noise** | activations are already computed; keeping them is free |
| probe compute, isolated | on-device 42µs vs `.cpu()` 109µs (2.6×) | probe FLOPs are trivial; the 42µs is mostly MPS kernel-launch overhead |
| **probe in a 48-tok decode loop** | on-device −0.6% [−1.9, +2.2] — **within noise** vs `.cpu()`/token **+19.1%** [+17.7, +21.7] | the headline |

**The headline:** the *same probe*, applied two ways, is either unmeasurable or
a ~19% decode-latency tax (+5.3 ms/token; 36.2 → 30.4 tok/s). The entire
difference is a `hidden.cpu().numpy()` per token, which forces a device sync
each step and serializes the decode loop. Keep the probe on-device (fold
StandardScaler+LogisticRegression into one `W @ h + b` tensor — see
`fold_probe()` in `05_cost.py`) and its cost is below what 20 runs can resolve
(CI upper bound ≈ +2%).

> Takeaway for the talk: **the probe is free; reading activations *off-device*
> is the bill.** And at real scale the bill grows — getting activations out of
> an optimized engine (vLLM CUDA graphs, continuous batching, TP, fp8) is the
> actual productionization cost, not the probe.

The +19.1% is a laptop lower bound (unified memory makes the on-device path ~free
on a Mac, while discrete GPUs pay a real device→host transfer). **GPU phase:**
[`gpu/README.md`](gpu/README.md) has the HF / vLLM v0.31.0 / SGLang v0.5.21
benchmark scripts (written and smoke-tested on the Mac, **not yet run on a GPU**,
so no GPU numbers exist yet). The engine source evidence, pinned to those tags,
is in [`inference_engine_exhibits.md`](inference_engine_exhibits.md).

## Notes / knobs

- `common.py` holds the model id and paths. Left padding is used so index `-1`
  is the true last token for every sequence in a batch.
- `hidden_states` is a tuple of length `num_layers + 1`; index 0 is the
  embedding layer, so "layer L" in the plot = `hidden_states[L]`.
- Determinism: fixed `SEED` for data generation and the 10 group splits.
- To probe an instruct-formatted prompt instead of a raw statement, wrap the
  text with the chat template before extraction (see `04_live.py`'s generate
  mode for the template call).
