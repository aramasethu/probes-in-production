"""Close the loop: generate -> read hidden state -> probe -> live truth signal.

Two modes:

  probe   (default)  Score fresh statements (not in training) with the best
                     probe and compare to ground truth.

  generate           Ask the model to complete a factual statement, then probe
                     the FULL generated statement's last-token hidden state for
                     truth -- the safety signal rides along the same forward
                     pass that produced the text.

Usage:
    python 04_live.py
    python 04_live.py generate
    python 04_live.py "The capital of Japan is Lisbon."
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import torch

from common import PROBE_PATH, load_model_and_tokenizer

# fresh held-out statements (deliberately NOT in the generated dataset)
FRESH = [
    ("The capital of the United States is Washington.", 1),
    ("The capital of the United States is Toronto.", 0),
    ("The capital of Mexico is Mexico City.", 1),
    ("The capital of Switzerland is Zurich.", 0),       # it's Bern
    ("The chemical symbol for water is H.", 0),
    ("The chemical symbol for nickel is Ni.", 1),
    ("Eight plus nine equals seventeen.", 1),
    ("Eight plus nine equals twenty.", 0),
    ("One hundred is greater than ninety.", 1),
    ("Three is greater than fifty.", 0),
]


def hidden_at_layer(model, tok, device, text: str, layer: int) -> np.ndarray:
    """Last-token residual-stream vector at `layer` for `text`."""
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    return out.hidden_states[layer][0, -1, :].float().cpu().numpy()[None, :]


def verdict(probe, vec) -> tuple[str, float]:
    p_true = float(probe.predict_proba(vec)[0, 1])
    return ("TRUE " if p_true >= 0.5 else "FALSE"), p_true


def run_probe_mode(model, tok, device, probe, layer, extra_texts):
    items = [(t, None) for t in extra_texts] if extra_texts else FRESH
    print(f"\n=== probing fresh statements at layer {layer} ===\n")
    correct = 0
    scored = 0
    for text, gold in items:
        vec = hidden_at_layer(model, tok, device, text, layer)
        say, p = verdict(probe, vec)
        tag = ""
        if gold is not None:
            ok = int(p >= 0.5) == gold
            correct += ok
            scored += 1
            tag = "  ✓" if ok else "  ✗ (gold=%s)" % ("TRUE" if gold else "FALSE")
        print(f"  probe says {say}  p(true)={p:.2f}{tag}   {text}")
    if scored:
        print(f"\n  probe accuracy on fresh set: {correct}/{scored} "
              f"= {correct / scored:.2f}")


def run_generate_mode(model, tok, device, probe, layer):
    """Let the model generate, then probe what it produced."""
    prompts = [
        "Complete this factual statement in one short sentence: "
        "The capital of Australia is",
        "Complete this factual statement in one short sentence: "
        "The chemical symbol for gold is",
        "Complete this factual statement in one short sentence: "
        "Twelve plus thirteen equals",
    ]
    print(f"\n=== generate -> probe (layer {layer}) ===\n")
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt",
                                      return_dict=True).to(device)
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, n_in:], skip_special_tokens=True).strip()
        # probe the generated statement on its own forward pass
        vec = hidden_at_layer(model, tok, device, text, layer)
        say, p = verdict(probe, vec)
        print(f"  prompt   : {prompt.split(': ', 1)[1]} ...")
        print(f"  generated: {text!r}")
        print(f"  -> probe says {say}  p(true)={p:.2f}\n")


def main() -> None:
    bundle = joblib.load(PROBE_PATH)
    probe, layer = bundle["probe"], bundle["layer"]
    print(f"[live] loaded probe trained at layer {layer} "
          f"(held-out acc {bundle['test_acc']:.3f})")

    model, tok, device = load_model_and_tokenizer()

    args = sys.argv[1:]
    if args and args[0] == "generate":
        run_generate_mode(model, tok, device, probe, layer)
    else:
        run_probe_mode(model, tok, device, probe, layer, extra_texts=args)


if __name__ == "__main__":
    main()
