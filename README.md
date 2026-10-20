# Probes in production

Can you serve an interpretability probe (a small classifier that reads a model's
hidden states) in open serving engines like vLLM and SGLang, without slowing them
down? This repo has the experiments behind my PyTorch Conference NA 2026 poster.

## Short answer

- **The probe itself is free.** On the GPU, or inside the CUDA graph, it adds ~0%.
- **Getting the hidden states out is the cost.** `.cpu()` every token: +19% (laptop).
  Turning CUDA graphs off so a hook can run: +3% (vLLM) to +11% (SGLang) on an L4.
- **With CUDA graphs on (the default), hooks never run** in vLLM or SGLang: 0 of 128 steps.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, truth probe at layer 15.

## What's here

| folder | what |
|---|---|
| [`demo/`](demo/) | laptop pipeline: build the data, train the truth probe, measure its cost on Apple Silicon. See [demo/README.md](demo/README.md) |
| [`demo/gpu/`](demo/gpu/) | GPU benchmarks for HF, vLLM 0.31.0 and SGLang 0.5.21. See [demo/gpu/README.md](demo/gpu/README.md) |
| `demo/gpu/results/` | raw GPU numbers from an NVIDIA L4 *(TODO: add after the run finishes)* |
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
