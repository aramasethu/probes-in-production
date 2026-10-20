"""Poster figure: one decode step of Qwen2.5-1.5B, zoomed in on layer 15, showing
where the probe goes and the two routes that fail.

    python make_decode_step.py   ->  decode_step_probe.png / .pdf

Numbers: Qwen2.5-1.5B (28 layers, hidden 1536, bf16, 1.54B params = 3.1 GB),
probe tap = HF hidden_states[15] = after layer 15. Measured on an NVIDIA L4:
hooks ran 0 of 128 steps with CUDA graphs on; laptop .cpu() per token +19%.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK, GREY, LIGHT = "#222222", "#6b6f76", "#f4f5f7"
LAYER, PROBE, RED, KV = "#2b8a8a", "#2f9e44", "#c92a2a", "#ee4c2c"

fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(0, 180)
ax.set_ylim(0, 80)
ax.axis("off")


def box(x, y, w, h, fc="white", ec=INK, lw=1.6, ls="-", r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(x0, y0, x1, y1, c=INK, lw=2.2, ls="-", z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 color=c, lw=lw, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def text(x, y, s, size=13, c=INK, w="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=c, weight=w, ha=ha, va=va, zorder=7, **kw)


Y = 45  # residual stream height

# ---- input token + embedding --------------------------------------------
box(1, Y - 4, 13, 8)
text(7.5, Y + 1.3, "last token", 12)
text(7.5, Y - 1.6, '"Paris"', 12, GREY)
arrow(14, Y, 18, Y)
box(18, Y - 4, 11, 8, fc=LIGHT)
text(23.5, Y, "embed", 12)

# ---- CUDA graph box ----------------------------------------------------
gx0, gx1 = 31, 140
box(gx0, 1, gx1 - gx0, 65, fc="#fbfbfc", ec=GREY, lw=2, ls=(0, (6, 4)), r=1.5, z=1)
text(gx0 + 2, 63, "CUDA graph: the whole step is replayed on the GPU, no Python",
     13, GREY, ha="left", style="italic")

# residual stream line
ax.plot([29, gx1 + 2], [Y, Y], color=INK, lw=4, alpha=0.5, zorder=3, solid_capstyle="round")
text(gx0 + 2, 59.5, "residual stream  h  (1536 numbers) flows left to right", 11.5, GREY, ha="left")

# layers 1-14 as thin bars
bw, bg = 1.6, 0.55
x = gx0 + 3
for i in range(14):
    ax.add_patch(Rectangle((x, Y - 8), bw, 16, fc=LAYER, ec="white", lw=0.6, zorder=4))
    x += bw + bg
text(gx0 + 3 + 7 * (bw + bg), Y - 10.5, "layers 1–14", 11.5, LAYER)
l15_x0 = x + 1.5

# layer 15 expanded
parts = [("norm", 7), ("attention", 11), ("+", 3.5), ("norm", 7), ("MLP", 9), ("+", 3.5)]
total = sum(w for _, w in parts) + 0.9 * (len(parts) - 1)
box(l15_x0 - 1, Y - 9, total + 2, 18, fc="#e3f2f1", ec=LAYER, lw=2, r=1, z=3)
text(l15_x0 + total / 2, Y + 11.2, "layer 15", 13, LAYER, w="bold")
px = l15_x0
attn_x = None
for name, w in parts:
    is_add = name == "+"
    box(px, Y - 3.5, w, 7, fc="white" if not is_add else "#fff8e1", ec=LAYER, lw=1.4, r=0.6, z=5)
    text(px + w / 2, Y, name, 12 if not is_add else 15, w="bold" if is_add else "normal")
    if name == "attention":
        attn_x = px + w / 2
    px += w + 0.9
tap_x = px + 0.4                    # just after the last "+"
l15_x1 = px - 0.9

# KV cache under attention
box(attn_x - 9, 19, 18, 6, fc="#fde3dc", ec=KV, lw=1.4, r=0.6)
text(attn_x, 22, "KV cache", 12, KV)
arrow(attn_x - 1.5, Y - 3.6, attn_x - 1.5, 25.2, c=KV, lw=1.6)
arrow(attn_x + 1.5, 25.2, attn_x + 1.5, Y - 3.6, c=KV, lw=1.6)
text(attn_x + 3, 30, "read old K,V\nwrite new", 9.5, KV, ha="left")

# layers 16-28 as thin bars
x = l15_x1 + 2.5
for i in range(13):
    ax.add_patch(Rectangle((x, Y - 8), bw, 16, fc=LAYER, ec="white", lw=0.6, zorder=4))
    x += bw + bg
text(l15_x1 + 2.5 + 6.5 * (bw + bg), Y - 10.5, "layers 16–28", 11.5, LAYER)

# memory-bound note
text(gx0 + 2, 4, "each step reads all 3.1 GB of weights:\ndecode is memory-bound", 11.5, GREY, ha="left")

# ---- probe (inside the graph) ------------------------------------------
ax.add_patch(plt.Circle((tap_x, Y), 1.1, color=PROBE, zorder=6))
arrow(tap_x, Y - 1.2, tap_x, 13.2, c=PROBE, lw=3)
box(tap_x - 22, 3, 44, 10, fc="#e6f4ea", ec=PROBE, lw=2.4, r=1.2, z=5)
text(tap_x, 10.2, "probe inside the graph", 13, PROBE, w="bold")
text(tap_x, 6.2, "p = sigmoid(w·h + b)  ·  cost 0.0% (measured)", 11, PROBE)

# ---- output -------------------------------------------------------------
ox = 145
arrow(gx1 + 2, Y, ox, Y)
for k, (s, yy) in enumerate([("final norm", Y + 5), ("LM head", Y), ("sample", Y - 5)]):
    box(ox, yy - 2, 16, 4, r=0.5)
    text(ox + 8, yy, s, 11.5)
arrow(ox + 16, Y - 5, ox + 19, Y - 5)
box(ox + 19, Y - 12, 15, 14, ec=INK, lw=2, r=1)
text(ox + 26.5, Y - 1.5, "next token", 12, w="bold")
text(ox + 26.5, Y - 5.2, "+", 13, w="bold")
text(ox + 26.5, Y - 8.8, "1 float", 12, PROBE, w="bold")
arrow(tap_x + 22, 8, ox + 26.5, Y - 12.2, c=PROBE, lw=2.4, rad=0.25)

# ---- the two routes that fail ------------------------------------------
# 1) copy to CPU (left) and 2) Python hook (right), both straight up from the tap
for (bx, title, sub) in [
        (tap_x - 47, "copy to CPU  (.cpu() per token)", "GPU waits every step: +19% (laptop)"),
        (tap_x + 3, "Python hook", "skipped on graph replay: ran 0 of 128 steps")]:
    box(bx, 68.5, 44, 9, fc="#fdecec", ec=RED, lw=1.8, r=1)
    text(bx + 22, 74.8, title, 12.5, RED, w="bold")
    text(bx + 22, 71, sub, 11, RED)
    ax.plot([tap_x, bx + 22], [Y + 1.2, 68.4], color=RED, lw=2, ls=(0, (4, 3)), zorder=3)

fig.tight_layout(pad=0.3)
for ext in ("png", "pdf"):
    fig.savefig(f"decode_step_probe.{ext}", dpi=200, facecolor="white")
print("ok")
