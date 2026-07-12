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
../.venv/bin/python 03_sweep.py          # train one probe per layer, plot accuracy vs depth
../.venv/bin/python 04_live.py           # probe fresh held-out statements
../.venv/bin/python 04_live.py generate  # generate -> read hidden state -> probe (full loop)
../.venv/bin/python 04_live.py "Your own statement here."
../.venv/bin/python 05_cost.py           # benchmark what the probe costs at inference
```

Each step writes artifacts the next one reads:

| file | produced by | contents |
|------|-------------|----------|
| `statements.jsonl` | `01` | `{statement, label, kind}`, balanced 50/50 |
| `activations.npz`  | `02` | `X [n, 29, 1536]` last-token vectors, `y`, `kind` |
| `layer_sweep.png`  | `03` | accuracy-vs-layer curve |
| `best_probe.joblib`| `03` | best-layer logistic-regression probe (refit on all data) |

## What the result shows

**Layer sweep** (held-out accuracy, chance = 0.50):

- Layers 0–5: ~0.50 — the truth signal is not yet linearly readable.
- Layers 6–10: sharp climb.
- Layer 14: peak **0.91**, then a high plateau through the late layers.

This rise-then-plateau is the whole point: the "is this true?" signal is
**computed in the middle of the network**, not present at the input and not
trivially decodable early. A linear probe suffices to read it.

**Per-category accuracy at the best layer:**

| category   | accuracy |
|------------|----------|
| capital    | 1.00 |
| element    | 1.00 |
| continent  | 0.97 |
| compare    | 0.92 |
| arith      | 0.60 |

Truth-as-a-direction is cleanest for **factual recall the model reliably knows**
(capitals, element symbols, continents). Arithmetic truth is encoded far less
linearly — a useful caveat: a truth probe reads what the model *represents as
true*, which tracks knowledge, not ground truth.

## The closed loop (`04_live.py generate`)

The model generates a completion, and the same forward-pass hidden state is fed
to the probe to emit a live true/false signal — `generate -> read hidden state
-> probe -> safety signal`, all in one model.

## What it costs at inference (`05_cost.py`)

The probe is cheap to *train*; the talk-worthy question is what it costs to
*run* alongside generation. Three things are measurable on a laptop (MPS), no
GPU/vLLM required. All numbers below are from `Qwen2.5-1.5B-Instruct` on an M-series
Mac (medians, with `torch.mps.synchronize()` around every measured region):

| cost | measured | meaning |
|------|----------|---------|
| `output_hidden_states=True` | **~0%** (−0.7%) | activations are already computed; keeping them is free |
| probe compute, isolated | on-device 40µs vs `.cpu()` 112µs | probe FLOPs are trivial; the 40µs is mostly MPS kernel-launch overhead |
| **probe in a 48-tok decode loop** | on-device **−0.4%** (free) vs `.cpu()`/token **+24.3%** | the headline |

**The headline:** the *same probe*, applied two ways, is either free or a ~24%
throughput tax. The entire difference is a `hidden.cpu().numpy()` per token,
which forces a device sync each step and serializes the decode loop. Keep the
probe on-device (fold StandardScaler+LogisticRegression into one `W @ h + b`
tensor — see `fold_probe()` in `05_cost.py`) and you pay nothing.

> Takeaway for the talk: **the probe is free; reading activations *off-device*
> is the bill.** And at real scale the bill grows — getting activations out of
> an optimized engine (vLLM CUDA graphs, continuous batching, TP, fp8) is the
> actual productionization cost, not the probe.

## Notes / knobs

- `common.py` holds the model id and paths. Left padding is used so index `-1`
  is the true last token for every sequence in a batch.
- `hidden_states` is a tuple of length `num_layers + 1`; index 0 is the
  embedding layer, so "layer L" in the plot = `hidden_states[L]`.
- Determinism: fixed `SEED` for data generation and the train/test split.
- To probe an instruct-formatted prompt instead of a raw statement, wrap the
  text with the chat template before extraction (see `04_live.py`'s generate
  mode for the template call).
