"""Run every statement through the model and capture hidden states.

One forward pass per batch with `output_hidden_states=True`. For each
statement we keep the residual-stream vector at the LAST token (the model has
"read" the whole statement by then), at EVERY layer. Saved aligned with labels.

Output: activations.npz
    X     float32 [n_statements, n_hidden_states, hidden_size]
    y     int     [n_statements]
    kind  str     [n_statements]
"""
from __future__ import annotations

import json

import numpy as np
import torch
from tqdm import tqdm

from common import ACTS_PATH, DATA_PATH, load_model_and_tokenizer

BATCH_SIZE = 16


def main() -> None:
    rows = [json.loads(line) for line in open(DATA_PATH)]
    statements = [r["statement"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    kinds = np.array([r["kind"] for r in rows])

    model, tok, device = load_model_and_tokenizer()

    compute_dtype = None
    all_vecs: list[np.ndarray] = []
    for i in tqdm(range(0, len(statements), BATCH_SIZE), desc="extract"):
        batch = statements[i:i + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc)
        # tuple of (n_layers+1) tensors, each [B, seq, hidden].
        # left padding => index -1 is the true last token for all sequences.
        if compute_dtype is None:
            compute_dtype = out.hidden_states[0].dtype   # model's native dtype
        hs = torch.stack(out.hidden_states, dim=1)   # [B, L, seq, hidden]
        last = hs[:, :, -1, :]                        # [B, L, hidden]
        all_vecs.append(last.float().cpu().numpy())

    X = np.concatenate(all_vecs, axis=0)             # [n, L, hidden]
    np.savez_compressed(ACTS_PATH, X=X, y=labels, kind=kinds)

    print(f"[extract] saved {ACTS_PATH}")
    print(f"[extract] X shape {X.shape}  (statements, hidden_states, hidden_dim)")
    print(f"[extract] y shape {labels.shape}  balance {labels.sum()}/{len(labels)}")

    # --- data volume the extraction implies ------------------------------
    # One residual-stream read is a `hidden`-dim vector per (token, layer).
    # This is the demand-side cost that dominates once you move activations
    # off-device: bytes/token/layer x layers x tokens x requests.
    n, n_layers, hidden = X.shape                    # n_layers = num_layers + 1
    compute_bytes = torch.finfo(compute_dtype).bits // 8   # native, e.g. bf16=2
    stored_bytes = X.dtype.itemsize                        # what we saved, fp32=4

    bptl_compute = hidden * compute_bytes            # bytes / token / layer (native)
    bptl_stored = hidden * stored_bytes              # bytes / token / layer (as saved)
    bpt_all_layers = bptl_compute * n_layers         # bytes / token, all layers

    print(f"[volume] hidden={hidden}  layers(incl. embed)={n_layers}  "
          f"compute_dtype={compute_dtype}({compute_bytes}B)  stored=fp32({stored_bytes}B)")
    print(f"[volume] bytes/token/layer : {bptl_compute:,} B native  "
          f"({bptl_compute/1024:.1f} KiB)   |  {bptl_stored:,} B as-saved fp32")
    print(f"[volume] bytes/token, all {n_layers} layers : {bpt_all_layers:,} B "
          f"({bpt_all_layers/1024:.1f} KiB) native")


if __name__ == "__main__":
    main()
