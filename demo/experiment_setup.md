# Experiment Setup

## Scope and rationale

This first round is deliberately scoped to what a laptop can answer *validly*:
probe **correctness** and the **demand side of cost**, not production throughput.
All experiments run on a single MacBook Pro (Apple Silicon, MPS backend) with no
compute grant. Production inference engines (vLLM, SGLang) are not used here —
their Apple-Silicon support is still maturing and they do not run reliably on a
laptop — so any true throughput-overhead measurement is deferred to a later
GPU phase.

## Model and hardware

- **Model:** `Qwen2.5-1.5B-Instruct` (open weights; 28 transformer layers,
  hidden size 1536, yielding 29 hidden-state tensors including the embedding
  layer).
- **Hardware / backend:** Apple Silicon MacBook Pro, PyTorch on the `mps`
  device, single-precision (`float32`).
- **Precision note:** the residual stream is read in the model's native compute
  dtype (fp32 on this machine); a production bf16/fp16 deployment would halve the
  per-activation byte cost.

## Task and dataset

- **Task:** binary truthfulness — given a declarative statement, decide whether
  it is true or false from a linear read of the model's internal state.
- **Dataset:** a mechanically constructed, label-clean set of balanced
  true/false statements (50/50). Each true statement is paired with a matched
  false one formed by swapping in a plausible wrong value, so surface form is
  balanced across labels.
- **Categories:** factual-recall dominant — `capital`, `continent`, `element`
  (symbols) — plus a smaller, deliberately harder `compare` and `arith` slice.
  The category split lets us report *where* a linear truth signal is clean vs.
  weak, not just an aggregate number.
- **Split:** a fixed-seed train/test split; the seed is held constant for both
  data generation and the split to keep the pipeline deterministic.

## Activation extraction

- One forward pass per batch with `output_hidden_states=True`.
- For each statement we keep the **last-token** residual-stream vector (the model
  has "read" the full statement by that position) at **every** layer. Left
  padding is used so index `-1` is the true final token for every sequence in a
  batch.
- Output tensor: `X ∈ [n_statements, 29, 1536]`, aligned with labels and category
  tags.

## Probe

- **Probe:** a linear classifier — `StandardScaler` + logistic regression — mapping
  a single hidden vector to true/false.
- **Layer sweep:** one probe is trained and scored per layer on the held-out
  split, producing an accuracy-vs-depth curve to localize the signal. The best
  layer's probe is refit on all data and saved for the live loop.
- **Closed loop:** `generate → read hidden state → probe`. The model generates a
  completion and the same forward-pass hidden state that produced the text is fed
  to the probe, so the truthfulness signal rides along the generation with no
  extra model call.

## Cost measurement protocol

Inference cost is measured three ways on MPS. Timing follows standard async-GPU
hygiene: `torch.mps.synchronize()` brackets every measured region, each region is
warmed up before timing, and all reported values are medians over repeated runs.
Tiny on-device ops are amortized over a queued loop with a single trailing sync
(the production pattern: queue work, sync only when the value is needed).

1. **Hidden-states overhead** — one full forward pass with
   `output_hidden_states=False` vs. `True`, isolating the cost of *emitting*
   activations.
2. **Probe compute** — the probe applied on-device (`sigmoid(h·w + b)`, with the
   `StandardScaler`+`LogisticRegression` folded into a single affine map) vs. the
   same math after pulling the activation off-device via `.cpu().numpy()`, which
   forces a per-call device sync.
3. **Realistic decode loop** — a fixed-length token-by-token decode, probing at
   every step, in three variants: model-only, model + on-device probe, model +
   off-device (`.cpu()`) probe. Reported as latency, throughput (tok/s), added
   latency per token, and percentage overhead vs. the model-only baseline.

The on-device vs. off-device contrast is the controlled A/B at the center of the
cost analysis: *identical probe math*, differing only in whether the activation
crosses the device→host boundary each step.

## Data-volume metric

Extraction also reports the data volume it implies — **bytes per token per
layer** (`hidden_size × dtype_bytes`), and its extrapolation across layers. This
is the demand-side quantity that dominates once activations are moved off-device
at scale (`bytes/token/layer × layers × tokens × requests`), and is what the
device→host transfer cost is ultimately a function of.

---

# Preliminary Results

Two questions this laptop round is scoped to answer: is the probe signal real and
localized (correctness), and what does reading activations cost (demand side).
Both come back cleanly.

## The probe signal is real and localized in depth

A linear probe swept across all 29 hidden-state layers rises from chance in the
early layers to a mid-network peak, then plateaus:

- **Layers 0–5:** ≈ 0.50 (chance) — truth is not yet linearly readable at the
  input.
- **Layers 6–10:** sharp climb (0.64 → 0.85).
- **Layer 14:** peak held-out accuracy **0.915** (chance 0.500).
- **Layers 15–28:** high plateau (≈ 0.85–0.90).

The rise-then-plateau is the substantive finding: the "is this true?" signal is
*computed* in the middle of the network, not present at the input and not
trivially decodable early. A single linear read suffices to recover it.

## The signal tracks knowledge, not ground truth

Per-category accuracy at the best layer (14), on the held-out split:

| Category  | Accuracy | n  |
|-----------|----------|----|
| capital   | 1.000    | 23 |
| element   | 1.000    | 11 |
| continent | 0.969    | 32 |
| compare   | 0.923    | 13 |
| arith     | 0.600    | 15 |

Truth-as-a-direction is cleanest for factual recall the model reliably knows
(capitals, element symbols, continents) and markedly weaker for arithmetic. The
useful caveat: the probe reads what the model *represents* as true, which tracks
its knowledge, not external ground truth.

## Data volume the extraction implies

Reading the residual stream costs, per token per layer, one `hidden`-dim vector:

- **6,144 B = 6.0 KiB per token per layer** (hidden 1536 × 4 B, fp32 on this
  machine; a bf16/fp16 deployment halves this to 3.0 KiB).
- **178,176 B ≈ 174 KiB per token** across all 29 layers.

This is the demand-side quantity that scales as
`bytes/token/layer × layers × tokens × requests` and sets the size of the
device→host transfer once activations leave the accelerator.

## Inference cost: the probe is free; off-device access is the bill

*(Recorded `05_cost.py` run on MPS; medians with `torch.mps.synchronize()`
bracketing every measured region.)*

| Cost | Measured | Reading |
|------|----------|---------|
| `output_hidden_states=True` | **≈ 0%** (−0.7%) | activations are already computed; keeping them is ~free |
| Probe compute, isolated | on-device **40 µs** vs `.cpu()` **112 µs** | probe FLOPs are trivial; the gap is the forced per-call device sync |
| Probe in a 48-token decode loop | on-device **−0.4%** (free) vs `.cpu()`/token **+24.3%** | the headline |

The same probe, applied two ways, is either free or a ~24% throughput tax. The
entire difference is a `hidden.cpu().numpy()` per token, which forces a device
sync each step and serializes the decode loop. Kept on-device (scaler + logistic
regression folded into one `W·h + b`), it costs nothing.

**Bridge to the GPU phase.** Unified memory makes the on-device path ~free on a
Mac, so the measured +24% is a *lower bound* on the device→host transfer that
dominates on discrete GPUs — where it will be larger, and compounded by CUDA
graphs, continuous batching, tensor parallelism, and quantization. Whether that
infra cost ultimately justifies probes is a GPU-phase decision this laptop test
cannot settle.
