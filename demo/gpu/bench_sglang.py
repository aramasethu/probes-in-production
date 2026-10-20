"""C. SGLang v0.5.21 (pinned: main has a decoder-layer refactor that breaks
these hooks).

Modes (--mode):
  default              CUDA graphs on (decode "full" + prefill "breakable" on
                       CUDA by default: model_executor/cuda_graph_config.py:122-162)  <- baseline
  nograph              disable_cuda_graph=True (both phases -> "disabled":
                       arg_groups/cuda_graph_hook.py:60-62)
  nograph_hook_device  nograph + disable_overlap_schedule + forward hook on
                       model.layers.{L-1}; probe on all rows, kept on device
  nograph_hook_cpu     same hook, rows .cpu() + numpy every forward
  default_hook_device  EXHIBIT: graphs on + the same hook. Hooks are registered
                       AFTER graph capture and replay never runs Python hooks
                       (model_runner_components/cuda_graph_setup.py:431-438);
                       probe_calls in the CSV shows how often it fired.
  return_hidden_last   return_hidden_states_mode="last" + per-request
                       return_hidden_states="last" with graphs ON. Returns the
                       FINAL layer, POST final norm (qwen2.py:441-446,
                       layers/logits_processor.py:808-858) -- not layer L, so
                       the trained probe can't use it. Measures what the
                       sanctioned path costs.
  correctness_hook     nograph + hook in "dump" mode: p(true) at the last
                       prompt token of each check statement (rows mapped to
                       requests via ForwardBatch.rids / extend_seq_lens_cpu)
  correctness_return_hidden  return_hidden_states=True: cosine of SGLang's
                       final post-norm vector vs HF's (shows what you get)

Hook spec format: model_executor/hook_manager.py:18-81 (name, target_modules
fnmatch patterns, hook_factory "module:fn", config); Engine(**kwargs) ->
ServerArgs (entrypoints/engine.py:260-279). The hook runs in the scheduler
subprocess (spawn), so it reports via files in /dev/shm.

Note: "hook on model.layers.14" in the original plan would read hidden_states[15];
the probe for hidden_states[L] needs the hook on model.layers.{L-1}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{HERE}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import sglang as sgl  # noqa: E402
import torch  # noqa: E402

import common_gpu as C  # noqa: E402

SHM_DIR = "/dev/shm/probe_sglang_hook"
BENCH_MODES = ["default", "nograph", "nograph_hook_device", "nograph_hook_cpu",
               "default_hook_device", "return_hidden_last"]


def hook_spec(module_idx: int, kind: str) -> list[dict]:
    return [{
        "name": f"probe_layer{module_idx}",
        "target_modules": [f"model.layers.{module_idx}"],      # exact name, no glob
        "hook_factory": "sglang_probe_hook:probe_hook_factory",
        "config": {"probe_npz": str(C.PROBE_NPZ), "mode": kind, "out_dir": SHM_DIR},
    }]


def make_engine(mode: str, module_idx: int) -> sgl.Engine:
    kw = dict(model_path=C.MODEL_ID, dtype=C.DTYPE, random_seed=0,
              disable_radix_cache=True,            # prefix caching off (repeats reuse prompts)
              context_length=1024, mem_fraction_static=0.8,
              chunked_prefill_size=8192, max_running_requests=64,
              log_level="warning")
    kind = {"nograph_hook_device": "device", "nograph_hook_cpu": "cpu",
            "default_hook_device": "device", "correctness_hook": "dump"}.get(mode)
    if mode.startswith("nograph") or mode == "correctness_hook":
        kw["disable_cuda_graph"] = True
    if kind and mode != "default_hook_device":
        kw["disable_overlap_schedule"] = True       # one forward at a time: clean row mapping
    if kind:
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        os.makedirs(SHM_DIR, exist_ok=True)
        kw["forward_hooks"] = hook_spec(module_idx, kind)
    if mode == "return_hidden_last":
        kw["return_hidden_states_mode"] = "last"
    if mode == "correctness_return_hidden":
        kw["return_hidden_states_mode"] = "full"
    print(f"[sglang] Engine kwargs: {kw}", flush=True)
    return sgl.Engine(**kw)


def sp(n: int) -> dict:
    return {"max_new_tokens": n, "temperature": 0.0, "ignore_eos": True}


def hook_calls() -> int:
    p = Path(SHM_DIR) / "calls.txt"
    return int(p.read_text()) if p.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    help=",".join(BENCH_MODES + ["correctness_hook", "correctness_return_hidden"]))
    mode = ap.parse_args().mode
    probe, prompts = C.load_probe(), C.load_prompts()
    L, mi = probe["layer"], probe["module_idx"]
    ver = f"sglang {sgl.__version__}"
    eng = make_engine(mode, mi)
    try:
        if mode == "correctness_hook":
            ps = []
            for i, ids in enumerate(prompts["check_token_ids"]):
                rid = f"chk{i}"
                eng.generate(input_ids=[ids], sampling_params=[sp(1)], rid=[rid])
                recs = [json.loads(l) for l in (Path(SHM_DIR) / "dump.jsonl").open()]
                ext = [r for r in recs if r["rid"] == rid and r["is_extend"]]
                assert ext, f"no prefill record for {rid}: {recs[-3:]}"
                assert ext[0]["n_rows"] == len(ids), (ext[0], len(ids))
                ps.append(ext[0]["p_last"])
            C.write_correctness("sglang", f"forward_hook(model.layers.{mi}) out0+out1, graphs off",
                                ps, prompts, probe, notes=f"{ver}; bf16")
            return

        if mode == "correctness_return_hidden":
            ref = json.loads(C.HF_REF_JSON.read_text()) if C.HF_REF_JSON.exists() else {}
            hf_final = ref.get("final_postnorm_hf_bf16")
            cos = []
            for i, ids in enumerate(prompts["check_token_ids"]):
                o = eng.generate(input_ids=[ids], sampling_params=[sp(1)],
                                 return_hidden_states=True)[0]
                hs = o["meta_info"]["hidden_states"]
                prompt_rows = np.asarray(hs[0], dtype=np.float32)    # [len(ids), H] prefill chunk
                assert prompt_rows.shape[0] == len(ids), prompt_rows.shape
                v = prompt_rows[-1]
                if hf_final:
                    r = np.asarray(hf_final[i], dtype=np.float32)
                    cos.append(float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r))))
            note = (f"{ver}; final layer post-norm only -> layer-{L} probe N/A; "
                    f"cos(sglang final, HF final post-norm) = "
                    + (" ".join(f"{c:.4f}" for c in cos) if cos else "(no HF ref)"))
            print("[sglang]", note)
            C.write_correctness("sglang", "return_hidden_states (final layer)",
                                [None] * len(prompts["check_token_ids"]), prompts, probe, notes=note)
            return

        rh = "last" if mode == "return_hidden_last" else False

        def run(batch, n):
            outs = eng.generate(input_ids=batch, sampling_params=[sp(n)] * len(batch),
                                return_hidden_states=rh)
            if rh:   # touch the payload so we know it came back
                assert len(outs[0]["meta_info"]["hidden_states"]) > 0
            return sum(o["meta_info"]["completion_tokens"] for o in outs)

        calls_per_full = None
        if "hook" in mode:
            one = prompts["bench_prompt_token_ids"][:1]
            run(one, C.OUTPUT_LEN)                       # warm
            run(one, 1); c0 = hook_calls()               # marker: counter flushed at its prefill
            run(one, C.OUTPUT_LEN)
            run(one, 1); c1 = hook_calls()               # marker again
            # c1 - c0 = (fires during the full request) + (the 2nd marker's prefill, if it fired)
            calls_per_full = max(c1 - c0 - 1, 0) if c1 > c0 else 0
            print(f"[sglang] hook fired {calls_per_full} times for one {C.OUTPUT_LEN}-token "
                  f"request (raw counter {c0} -> {c1}; eager would be ~{C.OUTPUT_LEN})", flush=True)

        summ = C.bench(run, prompts["bench_prompt_token_ids"], f"sglang/{mode}",
                       sync=lambda: None)          # generate() returns finished results
        if "hook" in mode:
            print(f"[sglang] hook counter at end of run: {hook_calls()}")
        notes = {
            "default": "CUDA graphs on (decode full + prefill breakable), overlap schedule on",
            "nograph": "disable_cuda_graph=True",
            "nograph_hook_device": f"graphs off + disable_overlap_schedule + hook model.layers.{mi}, probe on device",
            "nograph_hook_cpu": f"graphs off + disable_overlap_schedule + hook model.layers.{mi}, .cpu() per forward",
            "default_hook_device": f"graphs ON + hook model.layers.{mi}; fired ~{calls_per_full}x/request",
            "return_hidden_last": "return_hidden_states_mode=last, graphs on; FINAL layer post-norm only",
        }[mode]
        kind = ("device" if "device" in mode else "cpu" if "cpu" in mode
                else "final_layer" if rh else "none")
        C.write_results("sglang", ver, mode, kind, summ, prompts["prompt_len"],
                        f"{notes}; bf16; radix cache off", probe_calls=calls_per_full)
    finally:
        eng.shutdown()


if __name__ == "__main__":
    main()
