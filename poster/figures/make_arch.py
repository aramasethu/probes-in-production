"""Poster figure: one decode step of Qwen2.5-1.5B-Instruct in an optimized engine,
with an in-graph probe. Same model the probe was trained and measured on.

Qwen2.5-1.5B: 28 decoder layers (all full attention, GQA 12 q / 2 kv heads,
head_dim 128), hidden 1536, bf16.
  residual stream at one layer : 1536 x 2 B = 3 KiB / token
  all 29 hidden states         : 29 x 3 KiB = 87 KiB / token
  KV cache                     : 28 layers x 2 (K,V) x 2 heads x 128 x 2 B = 28 KiB / token
Probe tap = HF hidden_states[15] = output of layers[14] (after 15 layers).

    python make_arch.py   ->  qwen25_decode_step.png / .pdf
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

LAYER, PROBE, RED, INK, GREY = "#2b8a8a", "#2f9e44", "#c92a2a", "#222222", "#868e96"
KV = "#ee4c2c"
REQ = ["#4c6ef5", "#f59f00", "#ae3ec9"]
N_LAYERS, TAP = 28, 15          # tap after the 15th layer = hidden_states[15]

fig, ax = plt.subplots(figsize=(18, 7.2))
ax.set_xlim(0, 180)
ax.set_ylim(0, 76)
ax.axis("off")


def box(x, y, w, h, fc, ec=INK, lw=1.5, ls="-", r=0.8, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z))


def arrow(x0, y0, x1, y1, c=INK, lw=2.2, ls="-", z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 color=c, lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=12, c=INK, w="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=c, weight=w, ha=ha, va=va, zorder=6, **kw)


# ---- input: flattened continuous batch ----------------------------------
text(11, 64, "Continuous batch", 13, w="bold")
y = 57
for r, c in enumerate(REQ):
    for j in range([4, 2, 3][r]):
        ax.add_patch(Rectangle((2 + j * 4.2, y - r * 5), 3.6, 3.6, fc=c, ec="white", zorder=3))
    text(20.5, y + 1.8 - r * 5, f"req {r + 1}", 10, GREY, ha="left")
text(11, 39.5, "flattened to\n[num_tokens, 1536]", 11, GREY)
arrow(22, 45, 31, 45)

# ---- CUDA graph with 28 layers ------------------------------------------
gx0, gx1 = 32, 132
box(gx0, 22, gx1 - gx0, 46, "#f8f9fa", ec=GREY, lw=1.8, ls=(0, (5, 4)), r=1.5, z=1)
text(gx0 + 1.5, 65.5, "CUDA graph (one replay per decode step):  Python hooks never fire here  (measured: 0 of 128 steps)",
     11.5, GREY, ha="left", style="italic")

lw_, gap, ly, lh = 2.6, 0.75, 37, 16
xs = [gx0 + 4.5 + i * (lw_ + gap) for i in range(N_LAYERS)]
for i, x in enumerate(xs):
    ax.add_patch(Rectangle((x, ly), lw_, lh, fc=LAYER, ec="white", lw=0.8, zorder=3))
# residual stream through the layers
ax.plot([gx0 + 1, gx1 - 1], [45, 45], color=INK, lw=4, alpha=0.55, zorder=4, solid_capstyle="round")
text((gx0 + gx1) / 2, 57.2,
     "Qwen2.5-1.5B: 28 decoder layers (attention + MLP)   ·   residual stream  h + r  (3 KiB / token / layer)",
     11.5)

# KV cache under the layers
for x in xs:
    ax.plot([x + lw_ / 2, x + lw_ / 2], [ly, 29.2], color=KV, lw=1, alpha=0.6, zorder=2)
box(gx0 + 2, 26, gx1 - gx0 - 4, 3.2, "#fde3dc", ec=KV, lw=1.2, r=0.6)
text((gx0 + gx1) / 2, 27.6, "Paged KV cache: 28 KiB per token (28 layers, 2 KV heads)  ·  prefix-cached across requests", 10.5, KV)

# ---- probe tap ----------------------------------------------------------
tx = xs[TAP - 1] + lw_ + gap / 2
ax.add_patch(plt.Circle((tx, 45), 0.9, color=PROBE, zorder=5))
arrow(tx, 44, tx, 15.2, c=PROBE, lw=2.6, z=5)
box(tx - 14, 6, 28, 9, "#e6f4ea", ec=PROBE, lw=2.2, r=1)
text(tx, 12.4, "Probe = \"layer 29\"", 12.5, PROBE, w="bold")
text(tx, 8.8, "p = sigmoid(w·(h+r) + b)", 11, PROBE)
text(tx + 1.2, 19.5, "tap after layer 15", 10.5, PROBE, ha="left", w="bold")

# crossed-out export path (routed above the graph)
ax.plot([tx, tx], [46, 71], color=RED, lw=1.8, ls=(0, (4, 3)), zorder=1)
arrow(tx, 71, 172, 71, c=RED, lw=1.8, ls=(0, (4, 3)), z=1)
ax.plot([161, 167], [68.3, 73.7], color=RED, lw=3, zorder=6)
ax.plot([161, 167], [73.7, 68.3], color=RED, lw=3, zorder=6)
text(tx + 2, 73.4, "instead, copy activations out: 87 KiB/token (all 29 hidden states) + a device sync every step",
     10.5, RED, ha="left")

# ---- output path --------------------------------------------------------
ox = 137
arrow(gx1 - 1, 41, ox, 41)
steps = [("final norm", 37), ("last-token slice", 32), ("lm_head → sampler", 27)]
for s, yy in steps:
    box(ox, yy - 1.8, 26, 3.6, "white", r=0.6)
    text(ox + 13, yy, s, 11)
for (_, a), (_, b) in zip(steps, steps[1:]):
    arrow(ox + 13, a - 1.9, ox + 13, b + 1.9, lw=1.6)
arrow(ox + 13, 25.1, ox + 13, 18.6, lw=1.6)
box(ox - 4, 9.5, 34, 9, "white", ec=INK, lw=1.8, r=1)
text(ox + 13, 16, "token  +  1 float / token", 12, w="bold")
text(ox + 13, 12.2, "scheduler: continue / escalate / abort", 10.5, GREY)
arrow(tx + 14, 10.5, ox - 4, 12, c=PROBE, lw=2.4)
text((tx + 14 + ox - 4) / 2, 3.2, "score sliced like logits; masked to accepted tokens under speculative decoding", 10, PROBE)

fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(f"qwen25_decode_step.{ext}", dpi=160)
print("ok")
