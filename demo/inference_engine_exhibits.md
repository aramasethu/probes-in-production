# What optimized inference hides: source-pinned exhibits

Talk-prep reference. Each exhibit shows, in the engines' own source, how an
optimization removes the clean activation access a linear probe needs. Pinned to
**release tags** (re-pinned 2026-10-19; every `file:line` below was checked
against these trees, shallow-cloned):

- vLLM **`v0.31.0`**: paths are relative to the repo root.
- SGLang **`v0.5.21`**: paths are relative to `python/sglang/srt/` unless they
  start with `docs/` or `examples/`.

Both are the versions the GPU benchmarks in `gpu/README.md` run. SGLang `main`
has a decoder-layer refactor that breaks `(hidden, residual)` hooks, so do not
move the pin without re-reading exhibit 4. Line numbers are valid only for these
tags.

**What changed since the June 2026 pin (main @ vLLM `4dcd10eb`, SGLang `02be2e71`):**
- vLLM: `extract_hidden_states` now supports **decode tokens** (per-request
  `include_output_tokens`), writes **asynchronously**, and the hidden-state RFC
  #33118 is closed. Model Runner V2 is the default, so the exhibit line numbers
  moved to `vllm/v1/worker/gpu/`.
- SGLang: `--enable-return-hidden-states` is now an alias for
  `--return-hidden-states-mode full`; there is a new `last` mode. Forward hooks
  are registered after CUDA-graph capture, and prefill CUDA graphs
  ("breakable") are on by default on CUDA. `recapture_if_needed` no longer exists.

## The sharpened thesis

It's not that the engines *hide* activations adversarially — it's **structural**:
the exact tricks that make inference fast (CUDA graphs, continuous batching,
paged/radix KV, tensor parallelism, quantization) each remove a property that
probing depends on (Python hook execution, `[batch, seq, hidden]` shape,
contiguous per-sequence state, a full unsharded vector, the trained dtype).

The clearest evidence is that **the maintainers say so themselves**: vLLM RFC
[#33118](https://github.com/vllm-project/vllm/issues/33118) states prior
hidden-state attempts "negatively impact hotpath performance or add too much
complexity," and that existing workarounds mean "many vllm features (such as
prefix caching, auto batching, async server, etc.) are usually disabled."
(Both quoted from the RFC body, checked 2026-10-19.) The RFC was **closed as
completed on 2026-03-03**. What shipped is `extract_hidden_states` (below), so
the honest version of the line is: you *can* now get activations out of vLLM,
but through speculative-decoding plumbing plus a KV-connector, with
chunked prefill off, one extra draft row per request per step, and disk or
host writes, not as a cheap in-graph read.

---

## vLLM v0.31.0

Model Runner V2 (`vllm/v1/worker/gpu/`) is used by default unless an unsupported
feature forces a fallback (`config/vllm.py:701-760`, `return True` at :760;
override with `VLLM_USE_V2_MODEL_RUNNER`, `envs.py:2092`). The V1 runner
(`vllm/v1/worker/gpu_model_runner.py`) is still in the tree as the fallback and is
cited where its structure differs.

| # | exhibit | file / symbol | why it blocks probing |
|---|---------|---------------|------------------------|
| 1 | CUDA graph replay | V2: `v1/worker/gpu/cudagraph_utils.py` `CudaGraphManager` (L142), `self.graphs[desc].replay()` (L544); capture driven from `v1/worker/gpu/model_runner.py::capture_model` (L1019). V1/compile path: `compilation/cuda_graph.py` `CUDAGraphWrapper.__call__` (L235), `entry.cudagraph.replay()` + `return entry.output` (L362-363) | replay runs captured kernels and returns the static output tensor — the Python forward never re-executes, so `register_forward_hook` never fires during decode. `enforce_eager=True` turns off **both** torch.compile and CUDA graphs (`config/vllm.py:1694-1696`) |
| 2 | continuous batching → flattened HS | V2: `v1/worker/gpu/model_runner.py::prepare_inputs` (L1318), `query_start_loc` built at L1389-1399, `sample_hidden_states = hidden_states[input_batch.logits_indices]` (L938, L1578). V1: `v1/worker/gpu_model_runner.py::_prepare_inputs` (L1951), `logits_indices = query_start_loc[1:] - 1` (L2231) | hidden states are `[total_tokens, hidden]`, not `[batch, seq, hidden]`; mapping a row → (seq, pos) requires `query_start_loc`/`seq_lens`/`positions`, and the runner immediately slices last-token-per-seq for the sampler |
| 3 | PagedAttention | `v1/attention/ops/paged_attn.py` `PagedAttention` (L15) `.split_kv_cache` (L17) / `.write_to_paged_cache` (L32); `v1/worker/gpu/block_table.py` (V2), `v1/worker/block_table.py` (V1) | per-sequence state is scattered across fixed blocks via `slot_mapping`; no contiguous buffer to read |
| 4 | model forward → sampler | `model_executor/models/qwen2.py` `Qwen2Model.forward` (L402-451), `models/llama.py` `LlamaModel.forward` (L406+) | residual stream is a local var passed layer→layer; the default return is only the **final** post-norm hidden (`qwen2.py:445-451`), then reduced to last-token logits. Mid-layer access is the `aux_hidden_states` side channel: `EagleModelMixin._maybe_add_hidden_state` (`models/interfaces.py` L1592-1602) is called at `qwen2.py:424` and `:432-434`, and appends `hidden + residual` (pre-final-norm) for ids in `aux_hidden_state_layers`. **No longer EAGLE-3-only**: `extract_hidden_states` configures the same list via `eagle_aux_hidden_state_layer_ids` |
| 5 | the sanctioned exit for pooling (pooler) | `model_executor/layers/pooler/` package: `abstract.py::Pooler` (L16), `tokwise/poolers.py::TokenPooler` (L48); V2 `v1/worker/gpu/pool/pooling_runner.py::PoolingRunner.pool` (L132); V1 `gpu_model_runner._pool` (L3401) | only **final-layer**, only for **pooling/embedding models**, not mid-generation. For mid-layer + generation, see `extract_hidden_states` below |
| 6 | tensor parallelism | `model_executor/layers/linear.py` `ColumnParallelLinear` (L433) `.forward` (L603), `RowParallelLinear` (L1626) `.forward` (L1762) | hidden dim sharded across ranks; a single-rank activation is a fragment — needs all-gather |
| 7 | quantization | `model_executor/layers/quantization/fp8.py` `Fp8LinearMethod` (L225) `.apply` (L418); `quantization/input_quant_fp8.py` | activations dynamically quantized to fp8/int with scales; a probe trained on fp16/bf16 sees a different distribution |

**vLLM's actual hidden-state support (acknowledge so the talk isn't dated):**
- **`extract_hidden_states` spec-decode method** — `v1/spec_decode/extract_hidden_states.py`
  (V2: `v1/worker/gpu/spec_decode/extract_hidden_states.py`). Docs:
  `docs/features/speculative_decoding/extract_hidden_states.md`.
  - ✅ **Arbitrary layers** via `eagle_aux_hidden_state_layer_ids`; id `k` is the
    residual stream after decoder layer `k-1`, i.e. HF `hidden_states[k]`
    (`models/interfaces.py:1592-1602`). Pre-final-norm: passing `num_hidden_layers`
    gives the last layer "_not_ normalized using the output norm" (`.md:6`).
  - ✅ **Decode tokens are supported (corrects the June pin, which said
    "prefill/prompt tokens only").** Per-request `kv_transfer_params`
    `include_output_tokens=True` saves prompt **and** generated-token hidden states
    (`.md:71`; set via `SamplingParams(extra_args={"kv_transfer_params": {...}})`,
    `v1/request.py:119-121`; consumed in
    `distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py:539-543`).
    The default is `False`, which saves prompt tokens only. The file holds
    `prompt + output − 1` tokens, because the final sampled token is never fed
    through the model (connector `:541-543`).
  - ✅ **Writes are asynchronous**, not blocking (corrects the June pin): a
    dedicated CUDA stream for D2H copies plus a `ThreadPoolExecutor` for disk writes
    (connector `:213-221`). A `LOCK_EX` flock on `<file>.lock` is held until the
    write finishes (`:285-288`, `:433`), and the reader blocks on it
    (`load_hidden_states`, `:48-79`). The path is returned in
    `output.kv_transfer_params["hidden_states_path"]` (`:559`).
  - ❌ Constraints that remain: `num_speculative_tokens` must be exactly 1
    (`v1/spec_decode/extract_hidden_states.py:35`; V2
    `v1/worker/gpu/spec_decode/extract_hidden_states.py:29-32`), which adds one draft
    row per request per step. **Chunked prefill must be disabled** (`.md:125-126`).
    It needs a `speculative_config` plus a `kv_transfer_config`
    (`ExampleHiddenStatesConnector`), and the reference connector writes
    safetensors to a shared path, not to GPU memory. Nobody has measured its
    decode cost on a GPU here yet (see `gpu/README.md`, mode `extract_hs`).
- RFC status (checked 2026-10-19 via GitHub): [#33118](https://github.com/vllm-project/vllm/issues/33118)
  **closed** 2026-03-03 (completed); [#24288](https://github.com/vllm-project/vllm/issues/24288)
  closed 2026-03-05; [#18176](https://github.com/vllm-project/vllm/issues/18176) closed 2026-05-07;
  [#12249](https://github.com/vllm-project/vllm/issues/12249) closed 2025-09-01. Still **open**:
  [#36998](https://github.com/vllm-project/vllm/issues/36998), the activation-intercept
  "observation plugin" RFC. So the "community is still fighting this" slide
  should now read "one feature shipped, one plugin RFC still open", not "five open RFCs".
- Third-party drop-in: `agencyenterprise/vllm-hidden-states` (last pushed 2026-02-06, before
  the official feature; treat as historical).
- Blog: vllm `2026-03-30-extract-hidden-states` (not re-verified this pass).

---

## SGLang v0.5.21

**Headline:** SGLang is the *probe-friendlier* engine — it ships
`--return-hidden-states-mode` (alias `--enable-return-hidden-states`) and a
generic `--forward-hooks` system — but the supported return path gives **only
the final layer, post-norm**, the hooks **are registered after CUDA-graph
capture and so never run on graph replay** (the docs still omit this), and
everything in between is flattened, sharded, quantized, and resolved a step late.

Two defaults to state precisely, because the old pin predates them:
- **Prefill CUDA graphs are on by default on CUDA**: the prefill phase's default
  backend is `breakable` ("BCG") on CUDA, `tc_piecewise` elsewhere
  (`model_executor/cuda_graph_config.py:122-135`, `:161`), and decode defaults to
  `full` (`:158`). So *both* phases skip Python hooks on replay, not just decode.
  `disable_cuda_graph=True` turns off both (`arg_groups/cuda_graph_hook.py:60-62`).
- **Hooks are registered after capture.** `finalize_cuda_graph_capture`
  (`model_executor/model_runner_components/cuda_graph_setup.py:431`) calls
  `register_forward_hooks` only after the last graph is captured (`:433-438`), and
  its own comment says "capture replay never runs Python hooks anyway". Hooks
  therefore fire only on the eager forward path. The user-facing doc says hooks
  "run on every forward pass" (`docs/docs/advanced_features/forward_hooks.mdx:17`, `:293`)
  and has no CUDA-graph warning.

| # | exhibit | file / symbol | why it blocks probing |
|---|---------|---------------|------------------------|
| — | return-hidden API | server arg `arg_groups/fields/exec_.py:71` `enable_return_hidden_states` (alias for mode `full`) and `:76` `return_hidden_states_mode` (`last` \| `full`); per-request `managers/io_struct.py:257` `return_hidden_states`; validated at `managers/tokenizer_manager.py:1347-1362`; server cap `model_executor/forward_batch_info.py:320` `get_server_return_hidden_states_mode`; stored at `layers/logits_processor.py:808` `_get_hidden_states_to_store`; examples `examples/runtime/hidden_states/` | **final layer only** (post-norm; `models/qwen2.py:440-446`), no per-layer selection. `full` returns all tokens' final-layer vectors, `last` only the last token. Layer-*k* access requires the EAGLE-3 aux path (exhibit 4) |
| — | forward-hooks | `model_executor/hook_manager.py:11` `register_forward_hooks` (spec fields `name`, `target_modules`, `hook_factory`, `config`: `:18-30`); server arg `arg_groups/fields/observability.py:231` `forward_hooks`; `docs/docs/advanced_features/forward_hooks.mdx` | ordinary `nn.Module` hooks, **registered after graph capture, never run on replay** (`cuda_graph_setup.py:431-438`); doc has no such warning (talking point) |
| 1 | CUDA graph replay | `model_executor/runner/decode_cuda_graph_runner.py` `capture` (L1026); `runner/prefill_cuda_graph_runner.py` `capture` (L1535); replay in `runner_backend/full_cuda_graph_backend.py:203` (decode) and `runner_backend/breakable_cuda_graph_backend.py:262` (prefill) | replayed kernels skip Python hooks in **both** phases. (The old `recapture_if_needed` / "toggling `return_hidden_states` forces recapture" claim is dropped: that function does not exist in v0.5.21 and I did not re-derive the new behaviour.) |
| 2 | continuous batching → flattened HS | `model_executor/forward_batch_info.py` `ForwardBatch` (L489); `seq_lens` (L502), `extend_seq_lens` (L631) | flat concatenated tokens, no batch dim; remap via extend/seq metadata |
| 3 | RadixAttention / prefix cache | `mem_cache/radix_cache.py` `TreeNode` (L249), `RadixCache` (L280), `match_prefix` (L354), `insert` (L414) | KV shared across requests in a radix tree, non-contiguous; reached via `req_to_token` |
| 4 | model forward | `models/qwen2.py` `Qwen2Model.forward` (L399-450, final `self.norm` at L440-446); intermediate layers captured only `if i in self.layers_to_capture` (L420; init L388); set by `set_eagle3_layers_to_capture` (L757-770) | only the final hidden is surfaced via the logits processor; arbitrary layers = EAGLE3 internals |
| 5 | overlap scheduling | `managers/scheduler.py` `init_overlap` (L1615); `managers/overlap_utils.py` `FutureMap` (L248) | next forward launched on a separate stream before results materialize; outputs (incl. hidden states) resolved a step late. `--disable-overlap-schedule` turns this off (`scheduler.py:489-490`) |
| 6 | tensor parallelism | `layers/linear.py` `ColumnParallelLinear` (L309), `QKVParallelLinear` (L951), `RowParallelLinear` (L1422) | hidden dim sharded; single-rank activation is partial |
| 7 | quantization | `layers/quantization/` `w8a8_fp8.py::W8A8Fp8LinearMethod` (L103), `w8a8_int8.py`, `fp8.py::Fp8LinearMethod` (L477) | w8a8 activations quantized with scales before matmul; not the float residual stream |

**SGLang one-liner:** "SGLang is the probe-friendly one — `--return-hidden-states-mode` and `--forward-hooks` — but you get only the final layer, the hooks are registered after graph capture and never run on replay (prefill graphs are now on by default too), and the rest is flattened across requests, sharded across ranks, quantized, and resolved a step late."

---

## How to use this in the talk

1. Show the laptop demo + cost table (probe is free on-device, `.cpu()` per token = +19.1% [+17.7, +21.7]; see `README.md`).
2. "Now scale it." Open vLLM `qwen2.py` forward — the residual stream is right
   there, then thrown to the sampler. Walk exhibits 1-7.
3. Land the maintainers' own RFC quote (#33118) — they agree. Then the update:
   it shipped as `extract_hidden_states`, and it *does* reach decode tokens now
   (`include_output_tokens`), but via spec-decode plumbing, async disk writes,
   chunked prefill off, one extra draft row per step. GPU numbers for that path
   come from `gpu/README.md` (not yet run).
4. Contrast SGLang: it *tried* to give you the door (return-hidden, forward-hooks)
   — and even that door is final-layer-only, and the hooks are registered after
   graph capture so they are silent under CUDA graphs (decode **and** prefill).
5. Takeaway: **the probe is free; activation access is the bill** — and you can
   read the size of that bill in the engines' own source.
