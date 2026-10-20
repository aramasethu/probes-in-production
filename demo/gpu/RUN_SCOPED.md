# Scoped GPU run — instructions for Claude Code

You are Claude Code running in the user's **laptop terminal** in `/Users/aishwarya/work/probes`.
Your job: run a **fixed, pre-agreed** set of GPU experiments on the user's RunPod pod over
SSH, bring the results back, and report. The numbers will back two upstream proposals
(SGLang and vLLM): *serving a linear probe today requires giving up CUDA graphs; the probe
itself is free when computed in-graph.*

Read `demo/gpu/README.md` once for background (layer mapping, what each mode does, what
was only verified by reading source). Everything you need to run is below.

## Scope rules (strict)

- Run **only** the modes listed in Step 2, at batch sizes 1 and 32, 3 repeats. Do not add
  modes, batch sizes, models or engines.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, bf16, trained probe at **layer 15** (from
  `demo/gpu/probe.npz`). Do not change the model or the probe.
- Code: branch `gpu-probe-benchmarks` of `https://github.com/aramasethu/probes-in-production`
  (public). Do not commit, push, open PRs or issues, or post anything to GitHub.
- If a run fails, you may make a **minimal fix on the pod copy** and rerun only that mode.
  Save every fix: `git -C /workspace/probes-in-production diff > demo/gpu/results/fixes.patch`
  on the pod. The user decides what gets committed.
- **Ask the user before:** spending more than ~30 minutes total on debugging, anything that
  costs extra money (new pods, bigger GPUs), deleting anything outside `/dev/shm` scratch,
  or stopping/terminating the pod. Never handle RunPod credentials or API keys.
- Long commands run inside `tmux` on the pod so a dropped SSH connection doesn't kill them.
  Poll with `tail` at sensible intervals (every few minutes); don't hammer the pod.

## Input you need from the user

The pod's **"SSH over exposed TCP"** command from RunPod → Connect, e.g.
`ssh root@203.0.113.7 -p 22045 -i ~/.ssh/id_ed25519`. (Not the `ssh.runpod.io` one; it
doesn't support rsync.) Define, for your own use:

```bash
POD_HOST=root@<ip>; POD_PORT=<port>; POD_KEY=~/.ssh/id_ed25519
P="ssh -o StrictHostKeyChecking=accept-new -p $POD_PORT -i $POD_KEY $POD_HOST"
```

## Step 0 — Preflight (≈1 min, before installing anything)

```bash
$P 'nvidia-smi; ldd --version | head -1; df -h /dev/shm / /workspace 2>/dev/null; nproc; free -g'
```

Pass criteria — **stop and tell the user if any fail** (they're paying per minute):

| check | need | why |
|---|---|---|
| `nvidia-smi` GPU | NVIDIA L4 (or what the user rented) | numbers are labelled with it |
| `CUDA Version` | ≥ 13.0 | vLLM 0.31 / SGLang 0.5.21 ship CUDA 13 builds |
| glibc | ≥ 2.34 (Ubuntu 22.04+) | sglang 0.5.21 only has manylinux_2_34 wheels |
| free disk on `/workspace` or `/` | ≥ 40 GB | three venvs + model |
| `/dev/shm` | ≥ 1 GB | SGLang IPC; vLLM `extract_hs` writes there |

## Step 1 — Clone and set up (≈10–15 min)

```bash
$P 'command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux git)'
$P 'mkdir -p /workspace && cd /workspace && git clone -b gpu-probe-benchmarks https://github.com/aramasethu/probes-in-production.git && git -C probes-in-production log --oneline -1'
$P 'tmux new -d -s setup "cd /workspace/probes-in-production/demo/gpu && bash setup.sh 2>&1 | tee setup.log"'
```

Poll `$P 'tail -3 /workspace/probes-in-production/demo/gpu/setup.log'` until it prints
`>> setup done`. Record the commit hash from the clone. If `setup.sh` exits on the glibc or
CUDA check, stop and tell the user.

## Step 2 — The runs (≈45 min)

All runs use the same environment; `run.py` launches every mode in a fresh process with the
right venv and appends to `results/results.csv` / `results/correctness.csv`; per-mode logs
go to `results/logs/`. **Run HF first** (its `correctness` writes the reference the others
compare against).

```bash
R='cd /workspace/probes-in-production/demo/gpu && export PROBE_BATCH_SIZES=1,32 PROBE_REPEATS=3'
$P "tmux new -d -s hf     \"$R && .venvs/hf/bin/python run.py hf     --modes correctness,compile_none,compile_ingraph 2>&1 | tee results_hf.log\""
# when hf finishes:
$P "tmux new -d -s sglang \"$R && .venvs/hf/bin/python run.py sglang --modes default,default_hook_device,nograph,nograph_hook_device,return_hidden_last,correctness_hook 2>&1 | tee results_sglang.log\""
# when sglang finishes:
$P "tmux new -d -s vllm   \"$R && .venvs/hf/bin/python run.py vllm   --modes default,default_mp,default_hook_device,eager,eager_hook_device,extract_hs,correctness_hook,correctness_extract 2>&1 | tee results_vllm.log\""
```

Run the three engines **one after another**, never at the same time (one GPU). `run.py` ends
with an OK/FAIL summary per mode.

### What each run establishes, and the sanity check to apply

| engine | mode | establishes | sanity check |
|---|---|---|---|
| HF | `correctness` | reference p(true) for 8 statements | `hf bf16` values within ~0.01 of `laptop fp32` |
| HF | `compile_none` | baseline: CUDA-graphed decode, no probe | `[check] … 16/16` (or ≥ first 4 tokens agree) |
| HF | `compile_ingraph` | **probe inside the CUDA graph costs ≈0%** | tok/s within a few % of `compile_none` |
| SGLang | `default` | baseline (CUDA graphs on) | — |
| SGLang | `default_hook_device` | **the gap: hook ≈ never fires with graphs on** | log line `hook fired N times`, expect N ≈ 0 |
| SGLang | `nograph` | cost of disabling CUDA graphs | slower than `default` |
| SGLang | `nograph_hook_device` | **only way to serve a probe today**; hook+probe add ≈0 over `nograph` | `hook fired` ≈ 128 (1 prefill + 127 decode) |
| SGLang | `return_hidden_last` | cost of the sanctioned export path (final layer only) | — |
| SGLang | `correctness_hook` | the hook reproduces the HF probe | p(true) within ~0.02 of HF bf16, same side of 0.5 |
| vLLM | `default` | baseline | — |
| vLLM | `default_hook_device` | the gap, as SGLang (may also **error** — that's evidence too; record it) | `hook fired N times`, expect ≈ 0 |
| vLLM | `eager` | cost of `enforce_eager` | slower than `default` |
| vLLM | `eager_hook_device` | only live way today; probe adds ≈0 over `eager` | `hook fired` ≈ 128 |
| vLLM | `default_mp` | baseline for `extract_hs` (multiprocess engine core, which extraction needs) | — |
| vLLM | `extract_hs` | **cost of vLLM's sanctioned mid-layer path** (layer 15, decode tokens, to `/dev/shm`) | finishes; rows appear in CSV |
| vLLM | `correctness_hook` / `correctness_extract` | both paths reproduce the HF probe | as SGLang; extract also prints a shape line `(263, 1, 1536)` |

If a sanity check fails, say so in the report — don't "fix" numbers.

### Known risks and first-try fixes (all unverified on GPU)

| symptom | first try |
|---|---|
| SGLang rejects `forward_hooks=[...]` (list) | in `bench_sglang.py::make_engine`, pass `json.dumps(hook_spec(...))` instead |
| CUDA OOM on L4 | lower `gpu_memory_utilization` 0.85→0.7 (`bench_vllm.py`) / `mem_fraction_static` 0.8→0.7 (`bench_sglang.py`) |
| HF `compile_*` crashes or recompiles every step | capture the log tail, report; don't swap compile modes |
| HF `[check]` < 16/16 but first 4 agree | fine (bf16 kernel differences); note it |
| `/dev/shm` too small | change `SHM_DIR` in `bench_vllm.py` / `bench_sglang.py` to `/workspace/shm/...` |
| vLLM `default_hook_device` errors | expected possibility; keep the error text for the report, move on |

Rerun a single mode with the same `R` prefix and `run.py <engine> --modes <mode>`.

## Step 3 — Plot and bring results back

```bash
$P 'cd /workspace/probes-in-production/demo/gpu && .venvs/hf/bin/python run.py plot && (git -C /workspace/probes-in-production diff > results/fixes.patch) && cp setup.log results_*.log results/'
rsync -av -e "ssh -p $POD_PORT -i $POD_KEY" $POD_HOST:/workspace/probes-in-production/demo/gpu/results/ demo/gpu/results/
ls demo/gpu/results
```

Check locally that `results.csv`, `correctness.csv`, `overhead.png`, `overhead.md`,
`freeze_*.txt` and `logs/` arrived. Then **tell the user to stop the pod** (they do it in the
RunPod console; billing runs until then).

## Step 4 — Report to the user

Keep it short. Include:

1. Pod: GPU name, CUDA version, branch commit hash, any `fixes.patch` contents (one line each).
2. Per engine, median tok/s and % vs baseline at bs=1 and bs=32 (from `overhead.md`), and
   the `hook fired` counts for the graphs-on vs graphs-off hook modes.
3. Correctness: max |p_engine − p_hf_bf16| per path and whether every statement landed on the
   same side of 0.5.
4. Any mode that failed or any sanity check that didn't hold, with the log line.
5. Open the chart: `demo/gpu/results/overhead.png`.

Do not draft or post the SGLang/vLLM issues — the user does that after reviewing the numbers.
