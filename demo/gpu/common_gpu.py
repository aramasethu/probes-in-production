"""Shared bits for the GPU benchmarks. Deliberately dependency-light (numpy +
stdlib; torch only where needed) so it imports in all three venvs (hf, vllm,
sglang) without pulling in sklearn/joblib.

Benchmark protocol (identical for every engine and mode):
  * the same N_PROMPTS chat prompts, each exactly PROMPT_LEN tokens, passed as
    token ids (no tokenizer differences between engines); batch size b uses
    prompts[:b] submitted together
  * greedy, ignore EOS, exactly OUTPUT_LEN new tokens per request
  * prefix caching OFF everywhere (repeats reuse the same prompts; a prefix
    cache would turn every repeat after the first into a cache hit and make
    TTFT meaningless)
  * per repeat:  t_full = wall time of generate(batch, OUTPUT_LEN)
                 t_1    = wall time of generate(batch, 1)     -> TTFT (batch)
    tok/s = b * OUTPUT_LEN / t_full ;  TPOT = (t_full - t_1) / (OUTPUT_LEN - 1)
    This "two-run" TTFT/TPOT works identically for HF, vLLM and SGLang offline
    APIs, so the three engines are measured the same way.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBE_NPZ = HERE / "probe.npz"
PROMPTS_JSON = HERE / "prompts.json"
RESULTS_DIR = Path(os.environ.get("PROBE_RESULTS_DIR", HERE / "results"))
RESULTS_CSV = RESULTS_DIR / "results.csv"
CORRECTNESS_CSV = RESULTS_DIR / "correctness.csv"
HF_REF_JSON = RESULTS_DIR / "hf_reference.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = "bfloat16"            # every engine runs the model in bf16
OUTPUT_LEN = int(os.environ.get("PROBE_OUTPUT_LEN", 128))
BATCH_SIZES = [int(x) for x in os.environ.get("PROBE_BATCH_SIZES", "1,8,32").split(",")]
REPEATS = int(os.environ.get("PROBE_REPEATS", 5))
WARMUP = int(os.environ.get("PROBE_WARMUP", 2))

CSV_FIELDS = [
    "timestamp", "engine", "engine_version", "mode", "probe", "batch_size",
    "prompt_len", "output_len", "repeats",
    "tok_s_median", "tok_s_p25", "tok_s_p75", "tok_s_min", "tok_s_max",
    "tpot_ms_median", "tpot_ms_p25", "tpot_ms_p75",
    "ttft_ms_median", "ttft_ms_p25", "ttft_ms_p75",
    "e2e_s_median", "probe_calls", "gpu", "torch_version", "notes",
]
CORR_FIELDS = [
    "timestamp", "engine", "path", "idx", "statement", "label",
    "p_laptop_fp32", "p_hf_fp32", "p_hf_bf16", "p_engine",
    "absdiff_vs_hf_bf16", "same_side_as_hf_bf16", "notes",
]


# ----------------------------------------------------------------- inputs
def load_probe() -> dict:
    """w [hidden] fp32, b float, layer L (hidden_states index), module index
    L-1 (the decoder layer whose OUTPUT is hidden_states[L])."""
    z = np.load(PROBE_NPZ)
    L = int(z["layer"])
    return {
        "w": z["w"].astype(np.float32),
        "b": float(z["b"]),
        "layer": L,
        "module_idx": L - 1,
        "ref_laptop": z["ref_p_true_laptop_fp32"].tolist(),
    }


def load_prompts() -> dict:
    d = json.loads(PROMPTS_JSON.read_text())
    assert d["model_id"] == MODEL_ID
    return d


# ----------------------------------------------------------------- timing
def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def summarize(t_full: list[float], t_one: list[float], bs: int) -> dict:
    tok_s = [bs * OUTPUT_LEN / t for t in t_full]
    tpot = [(f - o) / (OUTPUT_LEN - 1) * 1e3 for f, o in zip(t_full, t_one)]
    ttft = [o * 1e3 for o in t_one]
    return {
        "tok_s_median": statistics.median(tok_s), "tok_s_p25": pct(tok_s, 25),
        "tok_s_p75": pct(tok_s, 75), "tok_s_min": min(tok_s), "tok_s_max": max(tok_s),
        "tpot_ms_median": statistics.median(tpot), "tpot_ms_p25": pct(tpot, 25),
        "tpot_ms_p75": pct(tpot, 75),
        "ttft_ms_median": statistics.median(ttft), "ttft_ms_p25": pct(ttft, 25),
        "ttft_ms_p75": pct(ttft, 75),
        "e2e_s_median": statistics.median(t_full),
    }


def bench(run, prompts: list[list[int]], label: str, sync=lambda: None) -> list[dict]:
    """run(batch_token_ids, max_new_tokens) -> total generated tokens.
    Must block until generation is complete (call `sync` for async engines).
    Returns one summary dict per batch size."""
    out = []
    for bs in BATCH_SIZES:
        batch = prompts[:bs]
        assert len(batch) == bs, f"need {bs} prompts, have {len(prompts)}"
        for _ in range(WARMUP):
            run(batch, OUTPUT_LEN)
            run(batch, 1)
        t_full, t_one = [], []
        for _ in range(REPEATS):
            sync(); t0 = time.perf_counter()
            n = run(batch, OUTPUT_LEN); sync()
            t_full.append(time.perf_counter() - t0)
            assert n == bs * OUTPUT_LEN, f"{label}: expected {bs * OUTPUT_LEN} tokens, got {n}"
            sync(); t0 = time.perf_counter()
            run(batch, 1); sync()
            t_one.append(time.perf_counter() - t0)
        s = summarize(t_full, t_one, bs)
        s["batch_size"] = bs
        print(f"  [{label}] bs={bs:3d}  {s['tok_s_median']:9.1f} tok/s  "
              f"(IQR {s['tok_s_p25']:.1f}-{s['tok_s_p75']:.1f})  "
              f"TPOT {s['tpot_ms_median']:7.2f} ms  TTFT {s['ttft_ms_median']:8.2f} ms",
              flush=True)
        out.append(s)
    return out


# ----------------------------------------------------------------- output
def gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        return platform.processor() or "unknown"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def write_results(engine: str, engine_version: str, mode: str, probe: str,
                  summaries: list[dict], prompt_len: int, notes: str = "",
                  probe_calls: int | None = None) -> None:
    import torch
    rows = []
    for s in summaries:
        r = dict(s)
        r.update(timestamp=_now(), engine=engine, engine_version=engine_version,
                 mode=mode, probe=probe, prompt_len=prompt_len,
                 output_len=OUTPUT_LEN, repeats=REPEATS, gpu=gpu_name(),
                 torch_version=torch.__version__, notes=notes,
                 probe_calls="" if probe_calls is None else probe_calls)
        rows.append(r)
    _append(RESULTS_CSV, CSV_FIELDS, rows)
    print(f"  -> appended {len(rows)} rows to {RESULTS_CSV}")


def write_correctness(engine: str, path: str, p_engine: list[float | None],
                      prompts: dict, probe: dict, notes: str = "") -> None:
    ref = json.loads(HF_REF_JSON.read_text()) if HF_REF_JSON.exists() else {}
    rows = []
    for i, (s, y) in enumerate(zip(prompts["check_statements"], prompts["check_labels"])):
        p = p_engine[i]
        r16 = ref.get("p_hf_bf16", [None] * len(p_engine))[i]
        r32 = ref.get("p_hf_fp32", [None] * len(p_engine))[i]
        rows.append(dict(
            timestamp=_now(), engine=engine, path=path, idx=i, statement=s, label=y,
            p_laptop_fp32=probe["ref_laptop"][i], p_hf_fp32=r32, p_hf_bf16=r16,
            p_engine=p,
            absdiff_vs_hf_bf16="" if (p is None or r16 is None) else abs(p - r16),
            same_side_as_hf_bf16="" if (p is None or r16 is None) else int((p >= .5) == (r16 >= .5)),
            notes=notes))
    _append(CORRECTNESS_CSV, CORR_FIELDS, rows)
    shown = " ".join("NA" if p is None else f"{p:.3f}" for p in p_engine)
    ref16 = " ".join(f"{x:.3f}" for x in ref.get("p_hf_bf16", [])) or "(run hf first)"
    print(f"  [correctness:{engine}/{path}] p(true) = {shown}\n"
          f"  [correctness:hf/bf16 ref ] p(true) = {ref16}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
