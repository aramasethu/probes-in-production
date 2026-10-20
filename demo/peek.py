"""Peek at hidden states: what the model emits, and what `h` looks like."""
from common import load_model_and_tokenizer
import torch

model, tok, device = load_model_and_tokenizer()

text = "The capital of France is Paris."
enc = tok(text, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)

hs = out.hidden_states                     # tuple: one tensor per layer (+embeddings)
print(f"\ntext: {text!r}")
print(f"tokens: {tok.convert_ids_to_tokens(enc['input_ids'][0])}")
print(f"\nnum hidden-state tensors: {len(hs)}  (embeddings + {len(hs)-1} layers)")
print(f"each tensor shape: {tuple(hs[0].shape)}  == [batch, seq_len, hidden]")

# h = one layer, last token -> a single 1536-dim vector
L = 14
h = hs[L][0, -1, :]                         # [hidden]
print(f"\nh = hidden_states[{L}][0, -1, :]   shape {tuple(h.shape)}")
print(f"first 10 numbers of h:\n{h[:10].cpu().numpy()}")
print(f"\nnorm |h| = {h.norm():.2f},  min {h.min():.2f},  max {h.max():.2f}")
