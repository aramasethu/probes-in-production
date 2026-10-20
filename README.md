# Probes in production

Can you serve an interpretability probe (a small classifier that reads a model's
hidden states) in open serving engines like vLLM and SGLang, without slowing them
down? This repo has the experiments behind my PyTorch Conference NA 2026 poster.

## Short answer

- **The probe itself is free.** On the GPU, or inside the CUDA graph, it adds ~0%.
- **Getting the hidden states out is the cost.** `.cpu()` every token: +19% (laptop).
  Turning CUDA graphs off so a hook can run: +3% (vLLM) to +11% (SGLang) on an L4.
  vLLM's `extract_hidden_states`: +15% at batch 32.
- **With CUDA graphs on (the default), hooks never run** in vLLM or SGLang: 0 of 128 steps.
- **Same answers everywhere.** Every engine path that runs the probe matches HF on 8/8 statements.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, truth probe at layer 15.

## Results (NVIDIA L4)

Extra time per token vs. the engine's default (CUDA graphs on). Qwen2.5-1.5B, bf16,
256-token prompts, 128 new tokens, median of 3.

| how the probe is served | runs every token? | batch 1 | batch 32 |
|---|---|---|---|
| inside the CUDA graph (HF, compiled) | yes | 0.0% | 0.0% |
| vLLM: hook, CUDA graphs on | **no (0 of 128)** | 0.0% | +0.1% |
| vLLM: hook, CUDA graphs off | yes | +2.6% | +2.6% |
| vLLM: `extract_hidden_states` (to disk) | after the request | +2.0% | +14.6% |
| SGLang: hook, CUDA graphs on | **no (0 of 128)** | +0.1% | 0.0% |
| SGLang: hook, CUDA graphs off* | yes | +10.6% | +8.3% |
| SGLang: `return_hidden_states` | final layer only | +0.1% | +0.2% |

\*also turns off SGLang's overlap scheduler. `extract_hidden_states` is compared with
vLLM's default on the same multiprocess engine. Charts: `python demo/gpu/plot_poster.py`.

## What's here

| folder | what |
|---|---|
| [`demo/`](demo/) | laptop pipeline: build the data, train the truth probe, measure its cost on Apple Silicon. See [demo/README.md](demo/README.md) |
| [`demo/gpu/`](demo/gpu/) | GPU benchmarks for HF, vLLM 0.31.0 and SGLang 0.5.21. See [demo/gpu/README.md](demo/gpu/README.md) |
| [`poster/`](poster/) | poster source |

## Run it

Laptop:

```bash
cd demo && ../.venv/bin/python 01_make_data.py && ../.venv/bin/python 02_extract.py && ../.venv/bin/python 03_sweep.py && ../.venv/bin/python 05_cost.py
```

GPU (one NVIDIA GPU, CUDA 13, Ubuntu 22.04+):

```bash
cd demo/gpu && bash setup.sh && .venvs/hf/bin/python run.py all
```

## Upstream bugs, issues and PRs

| what | link | status |
|---|---|---|
| SGLang: forward hooks never run with CUDA graphs on | [sgl-project/sglang#45317](https://github.com/sgl-project/sglang/issues/45317) | open |
| SGLang: docs fix for the above | [sgl-project/sglang#45318](https://github.com/sgl-project/sglang/pull/45318) | draft PR |
| SGLang: warn at startup when hooks are set and CUDA graphs are on | TODO | not opened |
| SGLang: probe score inside the CUDA graph (feature proposal) | TODO | not opened |
| vLLM: forward hooks never run with CUDA graphs on | TODO | not opened |
| vLLM: `extract_hidden_states` never writes its file with `VLLM_ENABLE_V1_MULTIPROCESSING=0` | TODO | to check, then open |
| vLLM: probe score inside the CUDA graph (comment on RFC [#59542](https://github.com/vllm-project/vllm/issues/59542)) | TODO | not opened |

Related work by others: SGLang [#43449](https://github.com/sgl-project/sglang/pull/43449)
(mid-layer hidden states without a draft model), [#43450](https://github.com/sgl-project/sglang/pull/43450);
vLLM RFC [#36998](https://github.com/vllm-project/vllm/issues/36998).
