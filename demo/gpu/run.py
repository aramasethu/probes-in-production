"""One runner for all engines. Each (engine, mode) runs in a FRESH Python
process (engines can't be re-initialised with different graph/hook settings
in one process, and some env vars must be set before `import vllm`), using
that engine's venv from setup.sh. Every mode appends rows to
results/results.csv; correctness rows go to results/correctness.csv.

    python run.py hf          # A. HF eager vs torch.compile, probe in-graph vs .cpu()
    python run.py vllm        # B. vLLM v0.31.0
    python run.py sglang      # C. SGLang v0.5.21
    python run.py all         # hf, then vllm, then sglang, then plot
    python run.py vllm --modes default,extract_hs     # subset
    python run.py plot

Run `hf` first: its correctness mode writes results/hf_reference.json, which
the vLLM / SGLang correctness checks compare against.
Knobs (env): PROBE_BATCH_SIZES=1,8,32  PROBE_OUTPUT_LEN=128  PROBE_REPEATS=5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENVS = HERE / ".venvs"

ENGINES = {
    "hf": ("bench_hf.py", ["correctness", "eager_none", "eager_device", "eager_cpu",
                           "compile_none", "compile_ingraph", "compile_cpu"]),
    "vllm": ("bench_vllm.py", ["default", "eager", "eager_hook_device", "eager_hook_cpu",
                               "extract_hs", "correctness_hook", "correctness_extract"]),
    "sglang": ("bench_sglang.py", ["default", "nograph", "nograph_hook_device",
                                   "nograph_hook_cpu", "return_hidden_last",
                                   "correctness_hook", "correctness_return_hidden"]),
}


def python_for(engine: str) -> str:
    p = VENVS / engine / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def run_engine(engine: str, modes: list[str] | None) -> list[tuple[str, int, float]]:
    script, default_modes = ENGINES[engine]
    py = python_for(engine)
    status = []
    for mode in modes or default_modes:
        print(f"\n{'=' * 72}\n[run] {engine} / {mode}   ({py})\n{'=' * 72}", flush=True)
        t0 = time.time()
        log = HERE / "results" / "logs" / f"{engine}_{mode}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            proc = subprocess.Popen([py, "-u", str(HERE / script), "--mode", mode],
                                    cwd=HERE, env=os.environ.copy(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:          # tee to console + log
                sys.stdout.write(line)
                f.write(line)
            rc = proc.wait()
        status.append((mode, rc, time.time() - t0))
        if rc != 0:
            print(f"[run] !! {engine}/{mode} exited {rc}; see {log}. Continuing.", flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=list(ENGINES) + ["all", "plot"])
    ap.add_argument("--modes", default=None, help="comma-separated subset")
    a = ap.parse_args()
    modes = a.modes.split(",") if a.modes else None
    summary = {}
    for eng in (list(ENGINES) if a.engine == "all" else [] if a.engine == "plot" else [a.engine]):
        summary[eng] = run_engine(eng, modes)
    if a.engine in ("all", "plot"):
        subprocess.call([python_for("hf"), str(HERE / "plot.py")], cwd=HERE)
    for eng, st in summary.items():
        print(f"\n[run] {eng} summary:")
        for mode, rc, dt in st:
            print(f"   {'OK  ' if rc == 0 else 'FAIL'}  {mode:28s} {dt:7.1f}s")


if __name__ == "__main__":
    main()
