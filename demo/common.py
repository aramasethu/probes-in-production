"""Shared config + model loading for the truthfulness linear-probe demo.

Small instruct model on Apple Silicon (mps). We read the residual stream
(hidden states) at the last token of a declarative statement, then train a
linear probe to recover whether the statement is true or false.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "statements.jsonl"
ACTS_PATH = HERE / "activations.npz"
PROBE_PATH = HERE / "best_probe.joblib"
PLOT_PATH = HERE / "layer_sweep.png"

SEED = 0


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer():
    """Load the instruct model in float32 with left padding.

    Left padding matters: we read the last-token hidden state with index -1,
    and left padding guarantees index -1 is the real final token for every
    sequence in a batch.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    print(f"[common] loading {MODEL_ID} on {device} ...")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"[common] loaded. hidden_size={model.config.hidden_size}, "
          f"num_hidden_layers={n_layers} "
          f"(=> {n_layers + 1} hidden-state tensors incl. embeddings)")
    return model, tok, device
