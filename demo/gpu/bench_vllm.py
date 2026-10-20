"""B. vLLM v0.31.0 (Model Runner V2 is the default on CUDA in this version).

Modes (--mode):
  default              CUDA graphs + torch.compile (vLLM defaults)       <- baseline
  eager                enforce_eager=True  (disables BOTH torch.compile and CUDA
                       graphs: config/vllm.py:1694-1696 sets
                       compilation_config.mode=NONE, cudagraph_mode=NONE)
  eager_hook_device    eager + forward hook on model.model.layers[L-1]; probe
                       on every row, result stays on device
  eager_hook_cpu       eager + same hook but rows .cpu() + numpy each forward
  default_hook_device  EXHIBIT: graphs on + the same hook. Hooks don't run on
                       CUDA-graph replay; probe_calls in the CSV shows how often
                       it actually fired. (Behaviour under VLLM_COMPILE not
                       verified without a GPU -- may also recompile or error.)
  extract_hs           speculative_config method "extract_hidden_states" with
                       eagle_aux_hidden_state_layer_ids=[L], per-request
                       include_output_tokens=True, ExampleHiddenStatesConnector
                       writing safetensors to /dev/shm. Timed region includes
                       waiting for the async writes to finish (flock), because
                       that's when the activations actually exist off-GPU.
  correctness_hook     eager + hook: p(true) at the last prompt token of each
                       check statement (one request at a time, max_tokens=1)
  correctness_extract  extract_hs: same statements, read back from the files

Sources (paths in the v0.31.0 tree):
  * extract_hidden_states config: examples/features/speculative_decoding/
    extract_hidden_states_offline.py:20-44 and
    docs/features/speculative_decoding/extract_hidden_states.md
  * num_speculative_tokens must be 1: vllm/v1/spec_decode/extract_hidden_states.py:35,
    vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py:29-32
  * include_output_tokens is PER REQUEST via SamplingParams(extra_args=
    {"kv_transfer_params": {...}}): v1/request.py:119-121, connector :539-543
    (file then holds prompt + output - 1 tokens; the last sampled token is
    never fed through the model)
  * layer id k = residual stream after decoder layer k-1, hidden+residual,
    pre-final-norm: models/interfaces.py:1592-1602 (_maybe_add_hidden_state),
    qwen2.py:422-434. So id L == HF hidden_states[L] == our hook on layers[L-1].
  * output file: <shared_storage_path>/<req_id>.safetensors, tensors
    "hidden_states" [num_tokens, num_layers, hidden] and "token_ids";
    async writes, read with load_hidden_states() which blocks on <path>.lock
    (kv_connector/v1/example_hidden_states_connector.py:48-79, 343, 415-418, 497-499)
  * docs say chunked prefill must be disabled (.md:125-126), so every mode
    here uses max_num_batched_tokens >= max_model_len.
  * apply_model: entrypoints/llm.py:603-613 -> worker_base.py:153-155
    fn(worker.get_model()); with VLLM_ENABLE_V1_MULTIPROCESSING=0 it runs in
    process (uniproc_executor.py:90-105), otherwise it needs
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 (cloudpickle, serial_utils.py:220-231).
  * Qwen2DecoderLayer returns (mlp_out, residual) with the add fused into the
    next RMSNorm (qwen2.py:291-311); the fused CUDA add-norm is IN PLACE
    (kernels/vllm_c.py:54-89), so the hook must form out[0]+out[1] immediately.

Process topology: by default ALL modes run with VLLM_ENABLE_V1_MULTIPROCESSING=0
(engine core in-process) so hook and no-hook modes are comparable. Set
PROBE_VLLM_MP=1 to run all modes with the default multiprocess engine core
instead (the hook is then installed through cloudpickle).
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys

# --- must be set before importing vllm --------------------------------------
_MP = os.environ.get("PROBE_VLLM_MP", "0") == "1"
# extract_hidden_states MUST use the multiprocess engine core. The connector
# hands finished requests' pending saves to its writer only on the engine's
# NEXT step (build_connector_meta). The multiprocess EngineCore keeps stepping
# while finished requests await connector cleanup (scheduler.has_requests());
# the in-process client only steps while LLM.generate() has unfinished outputs,
# so after generate() returns nothing ever writes the file and waiting on its
# lock deadlocks (observed on an L4, 2026-10-20). default_mp is the matching
# baseline (same topology, no extraction).
_MP_MODES = ("extract_hs", "correctness_extract", "default_mp")
if any(m in " ".join(sys.argv) for m in _MP_MODES):
    _MP = True
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if _MP else "0"
if _MP:
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import common_gpu as C  # noqa: E402

SHM_DIR = "/dev/shm/probe_vllm_hs"
MAX_MODEL_LEN = 1024                     # 256 prompt + 128 output fits
MAX_BATCHED = 8192                       # 32 x 256 prefill in one step; >= max_model_len
BENCH_MODES = ["default", "default_mp", "eager", "eager_hook_device", "eager_hook_cpu",
               "default_hook_device", "extract_hs"]


# ---------------------------------------------------------------- the hook
def install_probe_hook(model, module_idx: int, w: list, b: float, kind: str):
    """Runs inside the worker (in-process, or cloudpickled to the engine core).
    Self-contained on purpose: only torch/numpy, no references to this file."""
    import numpy as _np
    import torch as _torch

    inner = getattr(model, "module", model)           # defensive, see tests/models/
    layer = inner.model.layers[module_idx]            # multimodal/generation/test_mm_prefix_lm.py
    w_np = _np.asarray(w, dtype=_np.float32)
    st = {"calls": 0, "w": None, "last": None, "rows": None, "kind": kind}

    def hook(mod, inputs, output):
        resid = output[0] + output[1]                 # fresh tensor: residual stream after layer
        if st["w"] is None:
            st["w"] = _torch.from_numpy(w_np).to(resid.device)
        st["calls"] += 1
        if st["kind"] == "device":
            st["last"] = _torch.sigmoid(resid.float() @ st["w"] + b)   # no sync
        elif st["kind"] == "cpu":
            h = resid.float().cpu().numpy()          # <-- sync every forward
            st["last"] = 1.0 / (1.0 + _np.exp(-(h @ w_np + b)))
        elif st["kind"] == "rows":                    # correctness: keep every row
            st["rows"] = _torch.sigmoid(resid.float() @ st["w"] + b).cpu().tolist()
        return None

    layer._probe_state = st
    layer._probe_handle = layer.register_forward_hook(hook)
    return type(layer).__name__


def read_probe_state(model, module_idx: int):
    inner = getattr(model, "module", model)
    st = inner.model.layers[module_idx]._probe_state
    return {"calls": st["calls"], "rows": st["rows"]}


def reset_probe_state(model, module_idx: int):
    inner = getattr(model, "module", model)
    st = inner.model.layers[module_idx]._probe_state
    st["calls"], st["rows"] = 0, None
    return True


# ---------------------------------------------------------------- engine
def make_llm(mode: str, layer: int) -> LLM:
    kw = dict(model=C.MODEL_ID, dtype=C.DTYPE, seed=0,
              max_model_len=MAX_MODEL_LEN, max_num_batched_tokens=MAX_BATCHED,
              max_num_seqs=64, enable_prefix_caching=False,
              gpu_memory_utilization=0.85)
    if mode.startswith("eager") or mode == "correctness_hook":
        kw["enforce_eager"] = True
    if mode in ("extract_hs", "correctness_extract"):
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        os.makedirs(SHM_DIR, exist_ok=True)
        kw["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [layer]}},
        }
        kw["kv_transfer_config"] = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": SHM_DIR},
        }
    print(f"[vllm] LLM kwargs: {kw}", flush=True)
    return LLM(**kw)


def sp(n: int, extract_outputs: bool = False) -> SamplingParams:
    extra = {"kv_transfer_params": {"include_output_tokens": True}} if extract_outputs else None
    return SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True, extra_args=extra)


def wait_and_cleanup(paths: list[str]) -> None:
    """Block until each async safetensors write is done (writer holds LOCK_EX
    on <path>.lock), then delete. Same mechanism as load_hidden_states()."""
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as ehc
    for p in paths:
        with open(p + ".lock") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
        ehc.cleanup_hidden_states(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    help=",".join(BENCH_MODES + ["correctness_hook", "correctness_extract"]))
    mode = ap.parse_args().mode
    probe, prompts = C.load_probe(), C.load_prompts()
    L, mi = probe["layer"], probe["module_idx"]
    llm = make_llm(mode, L)
    topo = "multiprocess engine core" if _MP else "VLLM_ENABLE_V1_MULTIPROCESSING=0"
    ver = f"vllm {vllm.__version__}"

    hook_kind = {"eager_hook_device": "device", "eager_hook_cpu": "cpu",
                 "default_hook_device": "device", "correctness_hook": "rows"}.get(mode)
    if hook_kind:
        print("[vllm] hook installed on:",
              llm.apply_model(lambda m: install_probe_hook(m, mi, probe["w"].tolist(),
                                                           probe["b"], hook_kind)))

    # ------------------------------------------------------------ correctness
    if mode.startswith("correctness"):
        ps = []
        for ids in prompts["check_token_ids"]:
            if mode == "correctness_hook":
                llm.apply_model(lambda m: reset_probe_state(m, mi))
                llm.generate([{"prompt_token_ids": ids}], sp(1), use_tqdm=False)
                st = llm.apply_model(lambda m: read_probe_state(m, mi))[0]
                rows = st["rows"]
                assert rows is not None and len(rows) >= len(ids), (len(ids), st)
                ps.append(rows[len(ids) - 1])          # prefill rows = prompt tokens (+ padding)
            else:
                from vllm.distributed.kv_transfer.kv_connector.v1 import (
                    example_hidden_states_connector as ehc)
                o = llm.generate([{"prompt_token_ids": ids}], sp(1), use_tqdm=False)[0]
                path = o.kv_transfer_params["hidden_states_path"]
                obj = ehc.load_hidden_states(path)
                hs, tids = obj["hidden_states"], obj["token_ids"]
                assert hs.shape[0] == len(ids) and hs.shape[1] == 1, hs.shape
                assert tids.tolist() == ids
                h = hs[-1, 0].float().numpy()
                ps.append(float(C.sigmoid(h @ probe["w"] + probe["b"])))
                ehc.cleanup_hidden_states(path)
        if mode == "correctness_extract":
            # decode-token check: include_output_tokens=True -> prompt + out - 1 rows
            from vllm.distributed.kv_transfer.kv_connector.v1 import (
                example_hidden_states_connector as ehc)
            ids = prompts["bench_prompt_token_ids"][0]
            o = llm.generate([{"prompt_token_ids": ids}], sp(8, True), use_tqdm=False)[0]
            obj = ehc.load_hidden_states(o.kv_transfer_params["hidden_states_path"])
            print(f"[vllm] include_output_tokens=True, prompt {len(ids)} + 8 out -> "
                  f"hidden_states {tuple(obj['hidden_states'].shape)} (expect ({len(ids) + 7}, 1, H))")
            ehc.cleanup_hidden_states(o.kv_transfer_params["hidden_states_path"])
        path = "forward_hook(layers[%d]) out0+out1, eager" % mi if mode == "correctness_hook" \
            else "extract_hidden_states id %d -> safetensors" % L
        C.write_correctness("vllm", path, ps, prompts, probe, notes=f"{ver}; bf16; {topo}")
        return

    # ------------------------------------------------------------ benchmark
    def run(batch, n):
        extract = mode == "extract_hs"
        outs = llm.generate([{"prompt_token_ids": ids} for ids in batch],
                            sp(n, extract_outputs=extract), use_tqdm=False)
        if extract:
            wait_and_cleanup([o.kv_transfer_params["hidden_states_path"] for o in outs])
        return sum(len(o.outputs[0].token_ids) for o in outs)

    calls_per_full = None
    if hook_kind:
        # count hook firings for ONE full-length bs=1 generate (exhibit for default_hook_device)
        run(prompts["bench_prompt_token_ids"][:1], C.OUTPUT_LEN)        # warm
        llm.apply_model(lambda m: reset_probe_state(m, mi))
        run(prompts["bench_prompt_token_ids"][:1], C.OUTPUT_LEN)
        calls_per_full = llm.apply_model(lambda m: read_probe_state(m, mi))[0]["calls"]
        print(f"[vllm] hook fired {calls_per_full} times for one {C.OUTPUT_LEN}-token "
              f"request (eager would be ~{C.OUTPUT_LEN})", flush=True)

    summ = C.bench(run, prompts["bench_prompt_token_ids"], f"vllm/{mode}",
                   sync=torch.cuda.synchronize)
    notes = {
        "default": "CUDA graphs + VLLM_COMPILE (defaults)",
        "default_mp": "defaults, multiprocess engine core (baseline for extract_hs)",
        "eager": "enforce_eager=True (no compile, no CUDA graphs)",
        "eager_hook_device": f"enforce_eager + forward hook layers[{mi}], probe on device",
        "eager_hook_cpu": f"enforce_eager + forward hook layers[{mi}], .cpu() per forward",
        "default_hook_device": f"graphs ON + hook layers[{mi}]; hook fired {calls_per_full}x/request",
        "extract_hs": f"extract_hidden_states ids=[{L}], include_output_tokens, /dev/shm, "
                      "timed incl. async write completion; spec decode adds 1 draft row/req/step",
    }[mode]
    C.write_results("vllm", ver, mode, hook_kind or ("extract" if mode == "extract_hs" else "none"),
                    summ, prompts["prompt_len"], f"{notes}; bf16; prefix caching off; {topo}",
                    probe_calls=calls_per_full)


if __name__ == "__main__":
    main()
