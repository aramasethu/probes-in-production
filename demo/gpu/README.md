# GPU probe-cost benchmarks (HF / vLLM / SGLang)

The goal is to replace the laptop lower bound (MPS: on-device probe −0.6%, within
noise; `.cpu()` per token +19.1% [95% CI +17.7, +21.7], see
`../experiment_setup.md`) with real numbers from one rented NVIDIA GPU (L4, A100
or H100).
The question being tested: **the linear probe costs nothing, but getting mid-layer
activations out of a production engine during decode is where the cost is.**

Model: `Qwen/Qwen2.5-1.5B-Instruct`, run in bf16 in every engine. The probe is the
folded `StandardScaler + LogisticRegression` from `../best_probe.joblib`, stored as
one affine map `p = sigmoid(h @ w + b)` in `probe.npz`.

> **Status: written and smoke-tested on a Mac. Not yet run on a GPU.** See
> [What could not be verified without a GPU](#what-could-not-be-verified-without-a-gpu).

## Pinned versions

| venv | packages | why |
|---|---|---|
| `.venvs/hf` | `torch==2.13.0`, `transformers==5.9.0` | same transformers the probe was trained with; the StaticCache + compile path was smoke-tested against it |
| `.venvs/vllm` | `vllm==0.31.0` (pulls `torch==2.13.0`, `transformers>=5.10.4,<5.18`) | Model Runner V2 is the default; `extract_hidden_states` supports `include_output_tokens` |
| `.venvs/sglang` | `sglang==0.5.21` (pulls `torch==2.13.0`, `transformers==5.12.1`, `flashinfer_python[cu13]==0.6.18`) | **must stay pinned**: SGLang `main` has a decoder-layer refactor that breaks the `(hidden, residual)` hooks |

Both engines' 2026 wheels target CUDA 13, so the box needs an **NVIDIA driver ≥ 580**
(`nvidia-smi` should show `CUDA Version: 13.x`). Pick a provider image that has it.

## Which layer is "layer 14" (or 15)?

`probe.npz` stores `layer = L`, the index into HF `hidden_states`. Every script reads
`L` from that file. **Today `L = 15`.** The other laptop session re-ran the sweep on
2026-10-17 and the best layer moved from 14 to 15. If the probe changes again, re-run
`export_probe.py` and nothing else needs editing.

| what | which tensor |
|---|---|
| HF `hidden_states[L]` (what the probe was trained on) | output of `model.model.layers[L-1]` (0-indexed). `export_probe.py` checks this with a hook: max abs diff 0 |
| HF `hidden_states[0]` / `hidden_states[-1]` | embeddings / **post**-final-norm (`tie_last_hidden_states`) |
| vLLM `eagle_aux_hidden_state_layer_ids=[L]` | residual stream after `layers[L-1]`, `hidden + residual`, pre-final-norm (`models/interfaces.py:1592-1602`). So id `L` is the same tensor as HF `hidden_states[L]` |
| vLLM / SGLang forward hook | hook on `layers[L-1]`; residual stream = `out[0] + out[1]` (the residual add is fused into the next RMSNorm). Both fused kernels are **in place**, so the hook sums immediately and keeps no references |
| SGLang `return_hidden_states` | **final** layer, **post**-norm only. The layer-L probe cannot use it |

⚠️ The original plan said "SGLang hook on `model.layers.14`". That module outputs
`hidden_states[15]`. Hooking for `hidden_states[L]` means hooking `model.layers.{L-1}`,
and the scripts do that. (With today's `L = 15`, that happens to be `model.layers.14`.)

## Run it

### 0. On the laptop (already done; re-run if the probe changes)

```bash
cd demo/gpu && ../../.venv/bin/python export_probe.py
```

This writes `probe.npz` (w, b, layer, plus the laptop fp32 reference p(true)) and
`prompts.json` (32 distinct chat prompts of exactly 256 tokens each, as token ids,
plus the correctness statements).

### 1. Copy the folder to the GPU box

```bash
rsync -av --exclude .venvs --exclude results demo/gpu/ gpu-box:probe-gpu/
```

### 2. On the GPU box

```bash
cd probe-gpu && bash setup.sh
```

```bash
.venvs/hf/bin/python run.py hf
```

```bash
.venvs/hf/bin/python run.py vllm
```

```bash
.venvs/hf/bin/python run.py sglang
```

```bash
.venvs/hf/bin/python run.py plot
```

Or do everything with `.venvs/hf/bin/python run.py all`. `run.py` launches every
(engine, mode) in a **fresh process** using that engine's venv, so any interpreter
can start it. Run `hf` first: its `correctness` mode writes
`results/hf_reference.json`, which the vLLM and SGLang correctness checks compare
against. A mode that crashes is logged and skipped; the rest still run.

Subsets: `run.py vllm --modes default,extract_hs`. Knobs (env vars):
`PROBE_BATCH_SIZES=1,8,32`, `PROBE_OUTPUT_LEN=128`, `PROBE_REPEATS=5`, `PROBE_WARMUP=2`.
On an L4 (24 GB) everything fits; if memory is tight, lower `gpu_memory_utilization`
/ `mem_fraction_static` in the bench scripts.

Expected wall time: about 15 min for HF (eager bs=1 is the slowest), about 20 min for
vLLM (each mode reloads the model and recaptures graphs), and about 20 min for SGLang.
That fits in half a day with room for one re-run.

### Outputs (`results/`)

| file | contents |
|---|---|
| `results.csv` | one row per (engine, mode, batch size): tok/s median/p25/p75/min/max, TPOT and TTFT median/IQR, e2e, `probe_calls`, GPU, versions, notes |
| `correctness.csv` | p(true) per check statement and engine path, compared with HF fp32/bf16 and the laptop fp32 values |
| `overhead.png` / `.pdf` / `.md` | poster chart and the same numbers as a table |
| `logs/<engine>_<mode>.log` | full stdout of each run |
| `freeze_*.txt` | exact package versions in each venv |

## Benchmark protocol (the same for every engine)

- Same 32 prompts, 256 tokens each, passed as **token ids** so tokenizers can't differ.
  Batch size `b` submits `prompts[:b]` together; `b ∈ {1, 8, 32}`.
- Greedy decoding, `ignore_eos`, exactly 128 new tokens per request. The script
  asserts the token count.
- **Prefix caching off** everywhere: vLLM `enable_prefix_caching=False`, SGLang
  `disable_radix_cache=True`. Repeats reuse the same prompts, so with caching on,
  every repeat would be a cache hit.
- Each repeat measures `t_full = generate(b, 128)` and `t_1 = generate(b, 1)`.
  From those: **tok/s** = b·128/t_full, **TTFT** = t_1 (batch prefill plus first
  token), **TPOT** = (t_full − t_1)/127. This works identically through all three
  offline APIs. 2 warmups, 5 repeats; the CSV reports the median and the spread.
- Baselines for the chart:
  - HF probe modes are compared with the same loop without a probe (eager vs eager,
    compiled vs compiled).
  - vLLM and SGLang modes are compared with that engine's `default` (CUDA graphs on).

## Modes

### A. HF transformers: `bench_hf.py`

A hand-written greedy loop over a `StaticCache`. Prefill runs eager; only the 1-token
decode step is compiled.

| mode | what |
|---|---|
| `eager_none` / `compile_none` | no probe (baselines). Compiled = `torch.compile(step, mode="reduce-overhead", fullgraph=True)`, i.e. inductor + CUDA graphs |
| `eager_device` | probe `sigmoid(h_L @ w + b)` on device; values kept on the GPU, never synced |
| `compile_ingraph` | the probe dot product runs **inside** the compiled, CUDA-graphed step |
| `eager_cpu` / `compile_cpu` | `h_L.cpu().numpy()` every token (forces a sync each step) |
| `correctness` | HF p(true) on the check statements, in fp32 and bf16 |

Before timing, every mode checks that its static-cache loop reproduces `generate()`
greedy tokens.

### B. vLLM 0.31.0: `bench_vllm.py`

| mode | config |
|---|---|
| `default` | defaults (VLLM_COMPILE + CUDA graphs) — **baseline** |
| `eager` | `enforce_eager=True`. This disables **both** torch.compile and CUDA graphs (`config/vllm.py:1694-1696`) |
| `eager_hook_device` | eager + `llm.apply_model(...)` installs a `register_forward_hook` on `model.model.layers[L-1]`; probe runs on every flattened row, result stays on device |
| `eager_hook_cpu` | same hook, but `.cpu()` + numpy every forward |
| `default_hook_device` | exhibit: graphs on + the same hook. `probe_calls` in the CSV shows how often it actually fired per 128-token request |
| `extract_hs` | `speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [L]}}}`, `kv_transfer_config={"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/probe_vllm_hs"}}`, per request `SamplingParams(extra_args={"kv_transfer_params": {"include_output_tokens": True}})`. The timed region includes waiting for the async safetensors writes to finish (flock on `<file>.lock`) |
| `correctness_hook` / `correctness_extract` | p(true) at the last prompt token: from the hook row, and from the safetensors file |

Config sources: `examples/features/speculative_decoding/extract_hidden_states_offline.py:20-44`
and `docs/features/speculative_decoding/extract_hidden_states.md`. More citations are
in the docstring of `bench_vllm.py`.

All vLLM modes use `max_model_len=1024` and `max_num_batched_tokens=8192`. The docs
say chunked prefill must be off for `extract_hidden_states`, so these values keep
every mode unchunked and comparable.

All modes also run with `VLLM_ENABLE_V1_MULTIPROCESSING=0` (engine core in-process),
so hook and no-hook numbers come from the same process topology. `PROBE_VLLM_MP=1`
switches every mode to the default multiprocess engine core; the hook is then sent
over with `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (cloudpickle). It's worth running
`default` both ways once to see what in-process costs.

### C. SGLang 0.5.21: `bench_sglang.py` + `sglang_probe_hook.py`

| mode | config |
|---|---|
| `default` | defaults: decode `full` CUDA graph **and prefill `breakable` CUDA graph** on CUDA, overlap scheduler on — **baseline** |
| `nograph` | `disable_cuda_graph=True` (both phases) |
| `nograph_hook_device` | + `disable_overlap_schedule=True` + `forward_hooks=[{"name": ..., "target_modules": ["model.layers.{L-1}"], "hook_factory": "sglang_probe_hook:probe_hook_factory", "config": {...}}]`; probe runs on `out[0]+out[1]`, kept on device |
| `nograph_hook_cpu` | same hook, `.cpu()` every forward |
| `default_hook_device` | exhibit: graphs on + hook. Hooks are registered **after** graph capture and never run on replay (`model_runner_components/cuda_graph_setup.py:431-438`) |
| `return_hidden_last` | `return_hidden_states_mode="last"` + per request `return_hidden_states="last"`, graphs on. Returns the final post-norm layer only |
| `correctness_hook` | hook in `dump` mode maps rows to requests via `ForwardBatch.rids` / `extend_seq_lens_cpu` |
| `correctness_return_hidden` | cosine of SGLang's final vector vs HF's final post-norm vector (the probe doesn't apply) |

The hook runs in SGLang's scheduler **subprocess**, so it reports back through files
in `/dev/shm/probe_sglang_hook`: `calls.txt` is flushed at each prefill, and
`dump.jsonl` is written only for correctness. Patterns use `fnmatch`, where `*`
also matches dots, so the target is an exact module name.

## What could not be verified without a GPU

Smoke-tested on the Mac:
- `export_probe.py` end to end. Layer mapping assert, folded vs sklearn |Δ| < 1e-4.
- `bench_hf.py --smoke --device cpu`, all 6 modes plus correctness. That covers:
  - fp32 on CPU;
  - `torch.compile(fullgraph=True)` with `backend="aot_eager"`, so no graph breaks, including the `output_hidden_states` path;
  - the static-cache loop matching `generate()` 16/16 tokens;
  - HF bf16/fp32 reference p(true) matching the laptop values.
- `sglang_probe_hook.py` against a fake `ForwardBatch`. Row-to-request mapping for extend and decode checked numerically.
- `bench_vllm.install_probe_hook` against a fake `(hidden, residual)`-returning layer stack.
- `plot.py` on synthetic data.

All config keys and API names were checked by reading the pinned **v0.31.0 / v0.5.21
source** (shallow clones). None of the following has been executed yet:

1. **`mode="reduce-overhead"` (CUDA graphs) on the HF decode step.** On CPU it was
   only tested with `aot_eager`. Possible failures: cudagraph-tree skips caused by the
   in-place StaticCache `cumulative_length` update (HF marks it as a static address, so
   it should be fine), or a recompile per batch size (expected; warmup absorbs it). If
   the compiled `check` shows fewer than 16/16 tokens agreeing, that is bf16 kernel
   differences; the first 4 tokens must agree.
2. **vLLM `default_hook_device`.** Under VLLM_COMPILE the hook could stay silent
   (expected), trigger a recompile, or error. It is labelled as an exhibit; the runner
   continues if it fails.
3. **vLLM hook rows.** With `enforce_eager` the hook sees `[num_tokens(+padding),
   hidden]`. The correctness check takes row `len(prompt)-1` of a single-request
   prefill. In the benchmark the probe runs on all rows, padding included; per-request
   mapping wasn't needed for the cost measurement.
4. **vLLM `extract_hs` + V2 model runner + `include_output_tokens`** at batch 32. Each
   decode step also carries one draft row per request (spec decode with
   `num_speculative_tokens=1`). That cost is part of the feature, and the CSV notes
   say so.
5. **`flock` wait in `extract_hs`.** This assumes the connector creates `<path>.lock`
   before `generate()` returns (it does in `request_finished`, per the source).
6. **SGLang `Engine(forward_hooks=[...])` with a Python list.** ServerArgs accepts it,
   but SGLang has no end-to-end test of it. If it fails, pass
   `forward_hooks=json.dumps(spec)` or use the server CLI flag `--forward-hooks`.
7. **SGLang `correctness_return_hidden` output layout:** `hidden_states[0]` = prompt
   rows when `max_new_tokens=1`. An assert guards this.
8. **CUDA 13 wheels / driver.** `setup.sh` uses `--torch-backend=cu130`.
9. Absolute numbers depend on the GPU. Report the GPU name, which is recorded in the
   CSV, next to every number on the poster.
