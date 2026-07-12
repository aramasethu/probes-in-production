# What optimized inference hides: source-pinned exhibits

Talk-prep reference. Each exhibit shows, in the engines' own source, how an
optimization removes the clean activation access a linear probe needs. Pinned to
commits observed 2026-06-07 — **re-verify before talk day; these repos churn.**

- vLLM `main` @ `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- SGLang `main` @ `02be2e71899491b7aaf2849dce6431f61fc190b6`

## The sharpened thesis

It's not that the engines *hide* activations adversarially — it's **structural**:
the exact tricks that make inference fast (CUDA graphs, continuous batching,
paged/radix KV, tensor parallelism, quantization) each remove a property that
probing depends on (Python hook execution, `[batch, seq, hidden]` shape,
contiguous per-sequence state, a full unsharded vector, the trained dtype).

The strongest evidence is that **the maintainers say so themselves**: vLLM RFC
[#33118](https://github.com/vllm-project/vllm/issues/33118) states prior
hidden-state attempts "negatively impact hotpath performance or add too much
complexity," and that existing workarounds run "with prefix caching, auto
batching, async server, etc. disabled." That single quote is the talk in one
line: you can have the optimizations or the activations, not both — yet.

---

## vLLM v1

| # | exhibit | file / symbol | why it blocks probing |
|---|---------|---------------|------------------------|
| 1 | CUDA graph replay | `vllm/compilation/cuda_graph.py` → `CUDAGraphWrapper.__call__` (replay L360-361); driven from `vllm/v1/worker/gpu_model_runner.py` `capture_model` (L6525) | replay runs captured kernels and returns the static output tensor — the Python forward never re-executes, so `register_forward_hook` never fires during decode |
| 2 | continuous batching → flattened HS | `vllm/v1/worker/gpu_model_runner.py` `_prepare_inputs`; `logits_indices = query_start_loc[1:] - 1` (L2143); per-req state in `gpu_input_batch.py` | hidden states are `[total_tokens, hidden]`, not `[batch, seq, hidden]`; mapping a row → (seq, pos) requires `query_start_loc`/`seq_lens`/`positions`, and the runner immediately slices last-token-per-seq for the sampler |
| 3 | PagedAttention | `vllm/v1/attention/ops/paged_attn.py` `PagedAttention.split_kv_cache`/`write_to_paged_cache`; `vllm/v1/worker/block_table.py` | per-sequence state is scattered across fixed blocks via `slot_mapping`; no contiguous buffer to read |
| 4 | model forward → sampler only | `models/qwen2.py` `Qwen2Model.forward` (L389-426), `models/llama.py` (L395-433) | residual stream is a local var passed layer→layer; only the **final** post-norm hidden is returned, then reduced to last-token logits. No API for layer *k*. Internal exception: `EagleModelMixin._maybe_add_hidden_state` (`models/interfaces.py` L1291-1301) — EAGLE-3-only, pre-final-norm |
| 5 | the sanctioned exit (pooler) | `model_executor/layers/pooler/` (now a package) `abstract.py::Pooler`, `tokwise/poolers.py::TokenPooler`; `gpu_model_runner._pool` | only **final-layer**, only for **pooling/embedding models** (`is_pooling_model`), not mid-generation |
| 6 | tensor parallelism | `model_executor/layers/linear.py` `RowParallelLinear.forward` (L1537), `ColumnParallelLinear.forward` (L575) | hidden dim sharded across ranks; a single-rank activation is a fragment — needs all-gather |
| 7 | quantization | `model_executor/layers/quantization/` `fp8.py::Fp8LinearMethod.apply` (L430); `input_quant_fp8.py` | activations dynamically quantized to fp8/int with scales; a probe trained on fp16/bf16 sees a different distribution |

**vLLM's actual hidden-state support (acknowledge so the talk isn't dated):**
- `extract_hidden_states` spec-decode method — `vllm/v1/spec_decode/extract_hidden_states.py`. Supports **arbitrary layers** (`eagle_aux_hidden_state_layer_ids`) ✅, BUT: prefill/prompt tokens only (recommends `max_tokens=1`), blocking disk writes in the reference connector, requires a KV-transfer config + spec setup, returns pre-final-norm sums. Blog: vllm `2026-03-30-extract-hidden-states`.
- Open RFCs = "community is still fighting this" slide: [#33118](https://github.com/vllm-project/vllm/issues/33118) (the money quote), [#36998](https://github.com/vllm-project/vllm/issues/36998) (activation-intercept plugin, 2026), [#24288](https://github.com/vllm-project/vllm/issues/24288), [#18176](https://github.com/vllm-project/vllm/issues/18176), [#12249](https://github.com/vllm-project/vllm/issues/12249).
- Third-party drop-in: `agencyenterprise/vllm-hidden-states`.

---

## SGLang

**Headline:** SGLang is the *probe-friendlier* engine — it ships
`--enable-return-hidden-states` and a generic `--forward-hooks` system — but
the supported path returns **only the final layer**, the hooks **silently won't
fire under CUDA-graph replay** (the docs omit this!), and everything in between
is still flattened, sharded, quantized, and resolved a step late.

| # | exhibit | file / symbol | why it blocks probing |
|---|---------|---------------|------------------------|
| — | return-hidden API | `server_args.py:790` `--enable-return-hidden-states`; `io_struct.py:380` `return_hidden_states`; surfaced at `tokenizer_manager.py:1872`; example `examples/runtime/hidden_states/` | **final layer only**, no per-layer selection; forces `CaptureHiddenMode.FULL` (`model_runner.py:2571`) |
| — | forward-hooks | `model_executor/hook_manager.py` `register_forward_hooks`; `server_args.py:881` `--forward-hooks`; `docs/advanced_features/forward_hooks.md` | ordinary `nn.Module` hooks → **won't fire under CUDA graph replay**; doc has no such warning (talking point) |
| 1 | CUDA graph replay | `model_executor/cuda_graph_runner.py` `CudaGraphRunner.capture`/`replay`; `recapture_if_needed` (L1054-1083) | replayed kernels skip Python hooks; toggling `return_hidden_states` forces a full **recapture** |
| 2 | continuous batching → flattened HS | `model_executor/forward_batch_info.py` `ForwardBatch` (L210); `seq_lens`/`extend_seq_lens`/`extend_start_loc` | flat concatenated tokens, no batch dim; remap via extend/seq metadata |
| 3 | RadixAttention / prefix cache | `mem_cache/radix_cache.py` `RadixCache`/`TreeNode`; `match_prefix`/`insert` | KV shared across requests in a radix tree, non-contiguous; reached via `req_to_token` |
| 4 | model forward | `models/qwen2.py` `Qwen2Model.forward` (L400-446); intermediate captured only if `i in layers_to_capture` (L357, EAGLE3 `set_eagle3_layers_to_capture` L572) | only final post-/pre-norm hidden surfaced via logits processor; arbitrary layers = EAGLE3 internals |
| 5 | overlap scheduling | `managers/scheduler.py` `init_overlap` (L1173); `managers/overlap_utils.py` `FutureMap` | next forward launched on a separate stream before results materialize; outputs (incl. hidden states) resolved a step late — not a synchronous inline return |
| 6 | tensor parallelism | `layers/linear.py` `ColumnParallelLinear` (L292)/`RowParallelLinear` (L1340)/`QKVParallelLinear` | hidden dim sharded; single-rank activation is partial |
| 7 | quantization | `layers/quantization/` `w8a8_fp8.py`, `w8a8_int8.py`, `fp8.py` | w8a8 activations quantized with scales before matmul; not the float residual stream |

**SGLang one-liner:** "SGLang is the probe-friendly one — `--enable-return-hidden-states` and `--forward-hooks` — but you get only the final layer, the hooks won't fire under CUDA graphs, and the rest is flattened across requests, sharded across ranks, quantized, and resolved a step late."

---

## How to use this in the talk

1. Show the laptop demo + cost table (probe is free on-device, `.cpu()` = +24%).
2. "Now scale it." Open vLLM `qwen2.py` forward — the residual stream is right
   there, then thrown to the sampler. Walk exhibits 1-7.
3. Land the maintainers' own RFC quote (#33118) — they agree.
4. Contrast SGLang: it *tried* to give you the door (return-hidden, forward-hooks)
   — and even that door is final-layer-only and silently broken under CUDA graphs.
5. Takeaway: **the probe is free; activation access is the bill** — and you can
   read the size of that bill in the engines' own source.
