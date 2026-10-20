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
- **Groups:** every statement carries an entity `group` — the country (shared
  by its capital and continent rows), the element, or the unordered number
  pair. 312 statements fall into 106 groups; matched true/false twins always
  share a group.
- **Split:** entity-grouped (`GroupShuffleSplit`, 30% of groups held out, test
  n ≈ 90–100), repeated over 10 seeds, so no statement is ever scored while its
  twin or a same-entity sibling is in training. Seeds are fixed for data
  generation and splits, so the pipeline is deterministic.

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
- **Layer sweep:** one probe per layer per split; we report mean ± SE of
  held-out accuracy across the 10 splits and plot it as a band. SE across
  overlapping resampled splits understates true variance, so it is a lower
  bound on the noise.
- **Layer selection (nested):** within each split's training set, 5-fold
  `GroupKFold` validation picks the layer; that layer is scored once on the
  split's test set. The mean of those scores is the headline accuracy — the
  test set never chooses the layer. The deployed probe uses the layer chosen
  most often by the inner validation, refit on all data.
- **Per-category accuracy:** at each split's chosen layer, pooled over splits,
  reported with the number of distinct statements and groups behind it.
- **Closed loop:** `generate → read hidden state → probe`. The probe reads the
  layer-L hidden state from the *same* `generate()` call
  (`output_hidden_states=True, return_dict_in_generate=True`) at the last
  generated content token, so there is no extra model call.

## Cost measurement protocol

Inference cost is measured three ways on MPS. Timing follows standard async-GPU
hygiene: `torch.mps.synchronize()` brackets every measured region, each region is
warmed up before timing, and every value is a median with its interquartile
range (IQR). Overhead percentages carry a 95% bootstrap CI (5,000 resamples of
both arms' timing samples); an overhead whose CI spans zero is reported as
"within noise". Variants being compared are run interleaved (round-robin) so
thermal and background drift hit them equally. The decode loop uses 20 runs per
variant; the forward-pass comparison 30 per arm.
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
Both hold up under entity-grouped, repeated-split evaluation and repeated,
CI-reported timing; the one new caveat is that the probe must be trained in the
context it is deployed in.

## The probe signal is real and localized in depth

A linear probe swept across all 29 hidden-state layers, over 10 entity-grouped
splits (mean ± SE, test n ≈ 96, chance 0.500), rises from chance to a broad
mid-to-late plateau:

- **Layer 0:** 0.500 — the last token is always "." so the embedding is
  uninformative.
- **Layers 1–6:** 0.53–0.70. Above chance, but attributable to surface cues in
  the dataset rather than truth (e.g. "Europe" is the true continent in 21 of
  25 continent rows that name it; `compare` reaches 0.86 by layer 4).
- **Layers 7–12:** sharp climb (0.80 → 0.90).
- **Layers 13–28:** plateau at **0.91–0.92**; every layer is within 0.02 of the
  best mean (layer 16, 0.924 ± 0.006) and within split-to-split noise of its
  neighbours. No single peak layer is distinguishable.
- **Nested estimate** (layer chosen by inner validation on train, scored once on
  test): **0.918 ± 0.005**. Inner validation chose layers 14–26 across splits;
  layer 15 most often (3/10), which is the deployed probe.

The rise-then-plateau is the substantive finding: the "is this true?" signal is
*computed* in the middle of the network and stays linearly readable from there
on; it is not present at the input. A single linear read suffices to recover it.

## The signal tracks knowledge, not ground truth

Per-category accuracy at each split's inner-chosen layer, pooled over the 10
splits. `n_unique` is the number of distinct statements that appeared in any
test set (the SE uses it; pooled predictions repeat statements):

| Category  | Accuracy | ± SE  | n_unique | groups |
|-----------|----------|-------|----------|--------|
| capital   | 0.997    | 0.006 | 98       | 50     |
| continent | 0.953    | 0.021 | 98       | 50     |
| element   | 0.953    | 0.037 | 32       | 16     |
| compare   | 0.950    | 0.035 | 38       | 20     |
| arith     | 0.597    | 0.078 | 40       | 20     |

Truth-as-a-direction is cleanest for factual recall the model reliably knows
(capitals, continents, element symbols) and markedly weaker for arithmetic. The
useful caveat: the probe reads what the model *represents* as true, which tracks
its knowledge, not external ground truth. The small slices carry ±0.04–0.08
uncertainty, so only the arithmetic gap is clearly resolved.

## The probe needs to be trained in its deployment context

Reading the hidden state from the same `generate()` call works mechanically
(it matches a teacher-forced forward pass to 1e-6 relative error), but the
probe trained on *raw* statements does not transfer to the assistant-turn
context as-is. With the 312 statements re-framed as assistant replies inside
the chat template (layer 15, last "."):

| Probe | Accuracy |
|-------|----------|
| trained on raw statements, applied in-context | **0.500** (predicts FALSE for everything) — yet AUC **0.951** |
| same, after re-centring in-context activations to the raw mean | 0.872 ± 0.017 |
| trained on in-context activations (10 grouped splits) | **0.934 ± 0.008** |

The truth *direction* survives the context change (AUC 0.95); the offset does
not. A deployable closed loop therefore needs a probe trained on activations
extracted in the deployment format. *(Scratch measurement, not yet part of the
pipeline; making it so needs an in-context extraction mode in `02_extract.py`.)*

## Data volume the extraction implies

Reading the residual stream costs, per token per layer, one `hidden`-dim vector:

- **6,144 B = 6.0 KiB per token per layer** (hidden 1536 × 4 B, fp32 on this
  machine; a bf16/fp16 deployment halves this to 3.0 KiB).
- **178,176 B ≈ 174 KiB per token** across all 29 layers.

This is the demand-side quantity that scales as
`bytes/token/layer × layers × tokens × requests` and sets the size of the
device→host transfer once activations leave the accelerator.

## Inference cost: the probe is free; off-device access is the bill

*(Recorded `05_cost.py` run on MPS: medians with IQR, `torch.mps.synchronize()`
bracketing every measured region, compared variants interleaved, overheads with
95% bootstrap CIs. Forward pass: 30 runs per arm; decode loop: 20 runs per
variant of a 48-token greedy decode, probe at layer 15 every step.)*

| Cost | Median (IQR) | Overhead [95% CI] |
|------|--------------|-------------------|
| Forward pass, `output_hidden_states=False` | 32.45 ms (32.10–32.71) | — |
| Forward pass, `output_hidden_states=True` | 32.37 ms (32.12–32.71) | −0.2% [−1.0, +0.8] — **within noise** |
| Probe compute, on-device (per call) | 42.1 µs (41.6–43.2) | — |
| Probe compute, via `.cpu().numpy()` | 109.4 µs (107.7–110.7) | 2.6× [2.5×, 2.7×] |
| Decode loop, model only | 1327 ms (1308–1342), 36.2 tok/s | — |
| Decode loop + on-device probe | 1319 ms (1307–1365), 36.4 tok/s | −0.6% [−1.9, +2.2] — **within noise** |
| Decode loop + `.cpu()` probe per token | 1580 ms (1570–1627), 30.4 tok/s | **+19.1% [+17.7, +21.7]** |

The same probe, applied two ways, is either unmeasurable (on-device: the CI
spans zero, upper bound ≈ +2%) or a ~19% decode-latency tax (+5.3 ms/token).
The entire difference is a `hidden.cpu().numpy()` per token, which forces a
device sync each step and serializes the decode loop. Kept on-device (scaler +
logistic regression folded into one `W·h + b`), its cost is below what this
benchmark can resolve.

**Bridge to the GPU phase.** Unified memory makes the on-device path ~free on a
Mac, so the measured +19% is a *lower bound* on the device→host transfer that
dominates on discrete GPUs — where it will be larger, and compounded by CUDA
graphs, continuous batching, tensor parallelism, and quantization. Whether that
infra cost ultimately justifies probes is a GPU-phase decision this laptop test
cannot settle.
