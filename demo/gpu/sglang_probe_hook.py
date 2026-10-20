"""Forward-hook factory for SGLang's --forward-hooks / Engine(forward_hooks=...).

SGLang v0.5.21 contract (python/sglang/srt/model_executor/hook_manager.py):
  spec = {"name", "target_modules": [fnmatch patterns], "hook_factory":
          "module:fn", "config": {...}}
  hook = fn(config); module.register_forward_hook(hook)
Hooks are registered AFTER CUDA-graph capture
(model_runner_components/cuda_graph_setup.py:431-438), so they only fire on
the eager path: with graphs on (default = full decode graph + "breakable"
prefill graph on CUDA) they essentially never fire. The hook runs in the
scheduler SUBPROCESS, so it reports back through files (in /dev/shm).

Qwen2DecoderLayer.forward(positions, hidden_states, forward_batch, residual)
returns (hidden_states, residual) with the residual-add fused into the next
layer's RMSNorm (qwen2.py:287-315), so the residual stream after layer i is
out[0] + out[1]. fused_add_rmsnorm later mutates both IN PLACE, so we sum
immediately (a fresh tensor) and never keep references to out[0]/out[1].
Hook target for probe layer L is "model.layers.{L-1}" (exact name: fnmatch's
'*' also matches dots).

config keys:
  probe_npz   path to probe.npz (w, b)
  mode        "device": probe on all rows, result kept on device (no sync)
              "cpu"   : probe rows .cpu() + numpy every call (forces a sync)
              "dump"  : per-request last-row p(true) -> <out_dir>/dump.jsonl
                        (correctness only; slow)
  out_dir     where calls.txt (fire counter, flushed at each prefill) /
              dump.jsonl go
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


def probe_hook_factory(config):
    z = np.load(config["probe_npz"])
    w_np = z["w"].astype(np.float32)
    b = float(z["b"])
    mode = config.get("mode", "device")
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    state = {"calls": 0, "w": None, "last": None}

    def _flush_calls():
        with open(os.path.join(out_dir, "calls.txt"), "w") as f:
            f.write(str(state["calls"]))

    def hook(module, inputs, output):
        resid = output[0] + output[1]                      # [num_tokens, hidden]
        if state["w"] is None or state["w"].device != resid.device:
            state["w"] = torch.from_numpy(w_np).to(resid.device)
        state["calls"] += 1
        # flush the fire counter on every prefill (rare: once per generate call)
        # so the parent can bracket a request with 1-token marker requests
        if inputs[2].forward_mode.is_extend():
            _flush_calls()
        if mode == "device":
            state["last"] = torch.sigmoid(resid.float() @ state["w"] + b)   # stays on GPU
        elif mode == "cpu":
            h = resid.float().cpu().numpy()                # <-- sync every forward
            state["last"] = 1.0 / (1.0 + np.exp(-(h @ w_np + b)))
        elif mode == "dump":
            fb = inputs[2]                                 # ForwardBatch
            p = torch.sigmoid(resid.float() @ state["w"] + b).cpu().tolist()
            rids = list(fb.rids or [])
            ext = bool(fb.forward_mode.is_extend())
            if ext:
                lens = list(fb.extend_seq_lens_cpu)
            else:                                          # decode: 1 row / request
                lens = [1] * len(rids)
            # ForwardMode is an IntEnum: str() is "1", so record .name explicitly
            fm = getattr(fb.forward_mode, "name", str(fb.forward_mode))
            recs, start = [], 0
            for rid, n in zip(rids, lens):
                recs.append({"rid": rid, "forward_mode": fm, "is_extend": ext,
                             "n_rows": int(n), "p_last": p[start + n - 1]})
                start += n
            with open(os.path.join(out_dir, "dump.jsonl"), "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            _flush_calls()
        return None                                        # leave output untouched

    return hook
