"""Run ON THE LAPTOP (needs the repo's .venv + cached model). Writes the two
small files every GPU script loads, so the GPU box needs neither sklearn nor
best_probe.joblib:

  probe.npz       folded probe  w [1536] fp32, b scalar, layer (read from the
                  joblib; 14 originally, 15 after the 2026-10-17 re-sweep), plus the
                  laptop (MPS/CPU fp32) reference p(true) for CHECK_STATEMENTS
  prompts.json    fixed benchmark prompts as token ids (all exactly PROMPT_LEN
                  tokens, chat-formatted, all distinct) + the correctness
                  statements as token ids

    cd demo/gpu && ../../.venv/bin/python export_probe.py

Layer mapping (verified here, see check_layer_mapping()):
  hidden_states[L] (what the probe was trained on, 02_extract.py)
    == output of model.model.layers[L-1]   (0-indexed decoder module)
    == the residual stream after L decoder layers.
  Engines that fuse residual-add into the next RMSNorm (vLLM, SGLang) return
  (hidden, residual) from each decoder layer; the residual stream after
  layers[L-1] is hidden + residual of its outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # demo/ for common.py

from common import MODEL_ID, PROBE_PATH, load_model_and_tokenizer  # noqa: E402

PROMPT_LEN = 256          # tokens per benchmark prompt (exact)
N_PROMPTS = 32            # distinct prompts; batch b uses prompts[:b]

# Correctness statements: fresh ones from 04_live.py (not in training) plus a
# few training-set kinds. Probed exactly like training: raw text, no chat
# template, last prompt token.
CHECK_STATEMENTS = [
    ("The capital of the United States is Washington.", 1),
    ("The capital of the United States is Toronto.", 0),
    ("The capital of Mexico is Mexico City.", 1),
    ("The capital of Switzerland is Zurich.", 0),
    ("The chemical symbol for nickel is Ni.", 1),
    ("The chemical symbol for water is H.", 0),
    ("One hundred is greater than ninety.", 1),
    ("Three is greater than fifty.", 0),
]


def fold_probe(bundle):
    """Same as demo/05_cost.py::fold_probe -- StandardScaler+LR -> w, b."""
    pipe = bundle["probe"]
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coef = lr.coef_[0]
    w = coef / sc.scale_
    b = float(lr.intercept_[0] - np.sum(sc.mean_ * coef / sc.scale_))
    return w.astype(np.float32), b, pipe


def check_layer_mapping(model, tok, device, layer: int) -> None:
    """hidden_states[layer] must equal the output of model.model.layers[layer-1]."""
    grab = {}
    h = model.model.layers[layer - 1].register_forward_hook(
        lambda m, i, o: grab.__setitem__("o", o[0] if isinstance(o, tuple) else o))
    enc = tok("The capital of France is Paris.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    h.remove()
    hs = out.hidden_states[layer]
    err = (hs - grab["o"]).abs().max().item()
    print(f"[map] len(hidden_states)={len(out.hidden_states)}  "
          f"max|hidden_states[{layer}] - layers[{layer - 1}].out| = {err:.2e}")
    assert err == 0.0, "layer mapping assumption broken"


def build_prompts(tok, statements: list[str]) -> list[list[int]]:
    """N_PROMPTS distinct chat prompts, each exactly PROMPT_LEN tokens.

    user content = a run of statements (different start offset per prompt),
    trimmed at the token level so the chat-template head/tail stay intact.
    """
    sentinel = "XQXQXQ"
    full = tok.apply_chat_template([{"role": "user", "content": sentinel}],
                                   add_generation_prompt=True, tokenize=False)
    head_txt, tail_txt = full.split(sentinel)
    instr = ("For each statement below, say whether it is true or false and "
             "explain your reasoning in detail.\n\n")
    head = tok(head_txt + instr, add_special_tokens=False)["input_ids"]
    tail = tok(tail_txt, add_special_tokens=False)["input_ids"]
    budget = PROMPT_LEN - len(head) - len(tail)
    assert budget > 64, budget
    prompts = []
    n = len(statements)
    for i in range(N_PROMPTS):
        body, j = [], 0
        while len(body) < budget:
            s = statements[(i * 7 + j) % n]
            body += tok(f"- {s}\n", add_special_tokens=False)["input_ids"]
            j += 1
        ids = head + body[:budget] + tail
        assert len(ids) == PROMPT_LEN
        prompts.append(ids)
    assert len({tuple(p) for p in prompts}) == N_PROMPTS
    return prompts


def main() -> None:
    bundle = joblib.load(PROBE_PATH)
    layer = int(bundle["layer"])
    w, b, pipe = fold_probe(bundle)

    model, tok, device = load_model_and_tokenizer()
    check_layer_mapping(model, tok, device, layer)

    # laptop fp32 reference p(true), folded vs sklearn
    check_ids, ref_fold, ref_sk = [], [], []
    w_t = torch.from_numpy(w).to(device)
    for text, _ in CHECK_STATEMENTS:
        enc = tok(text, return_tensors="pt").to(device)
        check_ids.append(enc["input_ids"][0].tolist())
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer][0, -1]
        ref_fold.append(torch.sigmoid(hs.float() @ w_t + b).item())
        ref_sk.append(float(pipe.predict_proba(hs.float().cpu().numpy()[None])[0, 1]))
    print("[ref] laptop fp32 p(true):")
    for (t, y), pf, ps in zip(CHECK_STATEMENTS, ref_fold, ref_sk):
        print(f"   y={y} p_fold={pf:.4f} p_sklearn={ps:.4f}  {t}")
    assert max(abs(a - c) for a, c in zip(ref_fold, ref_sk)) < 1e-4

    np.savez(HERE / "probe.npz",
             w=w, b=np.float32(b), layer=np.int64(layer),
             model_id=np.array(MODEL_ID),
             test_acc=np.float32(bundle["test_acc"]),
             ref_p_true_laptop_fp32=np.array(ref_fold, dtype=np.float32))

    rows = [json.loads(l) for l in (HERE.parent / "statements.jsonl").open()]
    prompts = build_prompts(tok, [r["statement"] for r in rows])
    (HERE / "prompts.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "prompt_len": PROMPT_LEN,
        "bench_prompt_token_ids": prompts,
        "check_statements": [t for t, _ in CHECK_STATEMENTS],
        "check_labels": [y for _, y in CHECK_STATEMENTS],
        "check_token_ids": check_ids,
        "example_prompt_text": tok.decode(prompts[0]),
    }, indent=None))
    print(f"[export] wrote probe.npz (w {w.shape}, b={b:.4f}, layer={layer}) "
          f"and prompts.json ({N_PROMPTS} x {PROMPT_LEN} tokens)")


if __name__ == "__main__":
    main()
