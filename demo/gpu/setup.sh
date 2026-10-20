#!/usr/bin/env bash
# One-time setup on the rented GPU box (Linux x86_64, NVIDIA L4/A100/H100).
# Three isolated venvs because vLLM and SGLang pin different transformers /
# flashinfer versions; all three use the same torch (2.13.0, what both engines pin).
#
#   bash setup.sh            # ~5-10 min, mostly downloads
set -euo pipefail
cd "$(dirname "$0")"

PY=3.12
TORCH_BACKEND=${TORCH_BACKEND:-cu130}   # vLLM 0.31 / SGLang 0.5.21 deps are cu13 builds

nvidia-smi
echo ">> need an NVIDIA driver that supports CUDA 13 (driver >= 580; 'CUDA Version: 13.x' above)."
echo ">> if it says 12.x, try TORCH_BACKEND=cu129 bash setup.sh (unverified for sglang's flashinfer[cu13])."

# sglang 0.5.21 only ships manylinux_2_34 wheels -> glibc >= 2.34 (Ubuntu 22.04+).
# (Checked 2026-10-19 with `uv pip compile --python-platform x86_64-manylinux_2_34`;
#  all three venvs resolve there, sglang fails on manylinux_2_28.)
GLIBC=$(ldd --version | sed -n 1p | grep -oE '[0-9]+\.[0-9]+$')
echo ">> glibc $GLIBC"
if [ "$(printf '%s\n2.34\n' "$GLIBC" | sort -V | head -1)" != "2.34" ]; then
  echo "!! glibc < 2.34: sglang==0.5.21 has no wheel for this image. Use an Ubuntu 22.04+ template."; exit 1
fi
df -h /dev/shm / | sed 's/^/>> /'

# need a uv that knows --torch-backend (some pod images ship an older uv, e.g. 0.9.0)
if ! command -v uv >/dev/null || ! uv pip install --help 2>/dev/null | grep -c -- --torch-backend >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

mkdir -p .venvs

# A. HF transformers baseline (transformers 5.9.0 = what the laptop probe was trained with;
#    StaticCache/compile code path smoke-tested against it on CPU)
uv venv .venvs/hf --python $PY
uv pip install --python .venvs/hf/bin/python --torch-backend=$TORCH_BACKEND \
    torch==2.13.0 transformers==5.9.0 accelerate numpy pandas matplotlib

# B. vLLM (pinned; brings torch 2.13.0, transformers>=5.10.4,<5.18)
uv venv .venvs/vllm --python $PY
uv pip install --python .venvs/vllm/bin/python --torch-backend=$TORCH_BACKEND \
    vllm==0.31.0 numpy

# C. SGLang (pinned: main has a decoder-layer refactor that breaks the hooks)
uv venv .venvs/sglang --python $PY
uv pip install --python .venvs/sglang/bin/python --torch-backend=$TORCH_BACKEND \
    sglang==0.5.21 numpy

# model weights once into the shared HF cache (~3 GB)
.venvs/hf/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-1.5B-Instruct'))"

# record exact versions next to the results
mkdir -p results
for e in hf vllm sglang; do
  echo "== $e"; .venvs/$e/bin/python -c "import torch,sys; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
  uv pip freeze --python .venvs/$e/bin/python > results/freeze_$e.txt
done
.venvs/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
.venvs/sglang/bin/python -c "import sglang; print('sglang', sglang.__version__)"
echo ">> setup done. next: .venvs/hf/bin/python run.py hf"
