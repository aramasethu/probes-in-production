"""Close the loop: generate -> read hidden state -> probe -> live truth signal.

Two modes:

  probe   (default)  Score fresh statements (not in training) with the best
                     probe and compare to ground truth.

  generate           Ask the model to complete a factual statement and read the
                     layer-L hidden state out of the SAME generate() call
                     (output_hidden_states=True), at the last generated content
                     token. No extra model call: the safety signal rides along
                     the forward passes that produced the text. For comparison
                     it also prints a clearly-labelled second-forward-pass score
                     (the generated text re-encoded on its own, out of context).

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


def same_pass_hidden(gen_out, gen_ids, special_ids, layer):
    """Layer-`layer` residual at the last generated content token, taken from
    the hidden states generate() already computed.

    gen_out.hidden_states has one entry per decode step; step t holds the
    hidden states of the token FED at step t (step 0 = the prompt, step t>=1 =
    generated token t-1). The final generated token (usually <|im_end|>) is
    never fed back, so it has no hidden state -- the last content token
    before it (typically the closing ".") does, which is also the position the
    probe was trained on (last token of a statement).
    """
    n_steps = len(gen_out.hidden_states)
    j = n_steps - 2                          # last generated token that was fed
    while j >= 0 and int(gen_ids[j]) in special_ids:
        j -= 1
    if j < 0:
        raise ValueError("no generated content token has a hidden state")
    h = gen_out.hidden_states[j + 1][layer][0, -1, :]
    return h.float().cpu().numpy()[None, :], j


def run_generate_mode(model, tok, device, probe, layer):
    """Generate, and probe the hidden state from that same generate call."""
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
                                 pad_token_id=tok.eos_token_id,
                                 output_hidden_states=True,
                                 return_dict_in_generate=True)
        gen_ids = gen.sequences[0, n_in:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        # closed loop: hidden state from the SAME generate call, no extra pass
        vec, j = same_pass_hidden(gen, gen_ids, set(tok.all_special_ids), layer)
        say, p = verdict(probe, vec)
        # comparison only: OLD behaviour, a SECOND forward pass that re-encodes
        # the generated text alone (no chat template / prompt in context)
        say2, p2 = verdict(probe, hidden_at_layer(model, tok, device, text, layer))

        print(f"  prompt   : {prompt.split(': ', 1)[1]} ...")
        print(f"  generated: {text!r}")
        print(f"  -> probe says {say}  p(true)={p:.2f}   [same generate() "
              f"pass, token {tok.decode(gen_ids[j])!r}, no extra forward]")
        print(f"     (comparison: {say2}  p(true)={p2:.2f}   [2nd forward "
              f"pass on the generated text alone, out of context])\n")


def main() -> None:
    bundle = joblib.load(PROBE_PATH)
    probe, layer = bundle["probe"], bundle["layer"]
    print(f"[live] loaded probe trained at layer {layer} "
          f"(held-out acc {bundle['test_acc']:.3f} "
          f"+/- {bundle.get('test_acc_se', float('nan')):.3f} over "
          f"{bundle.get('n_splits', 1)} entity-grouped splits)")

    model, tok, device = load_model_and_tokenizer()

    args = sys.argv[1:]
    if args and args[0] == "generate":
        run_generate_mode(model, tok, device, probe, layer)
    else:
        run_probe_mode(model, tok, device, probe, layer, extra_texts=args)


if __name__ == "__main__":
    main()
