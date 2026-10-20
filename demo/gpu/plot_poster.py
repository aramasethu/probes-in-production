"""Two simple poster charts from results/results.csv.

    python plot_poster.py                  # -> results/hooks_run.png, results/probe_cost.png
    python plot_poster.py --csv x.csv --out-dir ../../poster/figures

1. hooks_run.png  : does a forward hook run? (calls per 128-step request,
                    CUDA graphs on vs off, vLLM and SGLang)
2. probe_cost.png : what a probe that actually runs every token costs
                    (extra time per token vs the engine's default, bs 1 and 32)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.edgecolor": GRID, "figure.facecolor": "white",
})


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[(r["engine"], r["mode"], int(r["batch_size"]))] = r   # last run wins
    return rows


def tok_s(rows, e, m, bs):
    return float(rows[(e, m, bs)]["tok_s_median"])


def extra_time(rows, e, mode, base, bs):
    """% extra time per token = base tok/s / mode tok/s - 1."""
    return 100 * (tok_s(rows, e, base, bs) / tok_s(rows, e, mode, bs) - 1)


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def hooks_run(rows, out):
    engines = [("vLLM", "vllm", "default_hook_device", "eager_hook_device"),
               ("SGLang", "sglang", "default_hook_device", "nograph_hook_device")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    w = 0.36
    for i, (name, e, on, off) in enumerate(engines):
        n_on = int(rows[(e, on, 1)]["probe_calls"] or 0)
        n_off = int(rows[(e, off, 1)]["probe_calls"] or 0)
        for dx, n, c in ((-w / 2, n_on, ORANGE), (w / 2, n_off, BLUE)):
            ax.bar(i + dx, max(n, 1.2), w - 0.04, color=c, edgecolor="white")
            ax.text(i + dx, max(n, 1.2) + 3, f"{n}", ha="center", va="bottom",
                    fontsize=20, fontweight="bold", color=INK)
    ax.set_xticks(range(len(engines)), [n for n, *_ in engines], fontsize=19)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 64, 128])
    ax.set_ylabel("steps where the hook ran (of 128)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["CUDA graphs on (default)", "CUDA graphs off"],
              loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2,
              frameon=False, fontsize=15)
    fig.suptitle("With CUDA graphs on, hooks never run", x=0.02, ha="left",
                 fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def probe_cost(rows, out):
    items = [  # label, engine, mode, baseline
        ("Inside the CUDA graph\n(HF, compiled)", "hf", "compile_ingraph", "compile_none"),
        ("vLLM: hook,\nCUDA graphs off", "vllm", "eager_hook_device", "default"),
        ("vLLM:\nextract_hidden_states", "vllm", "extract_hs", "default_mp"),
        ("SGLang: hook,\nCUDA graphs off*", "sglang", "nograph_hook_device", "default"),
    ]
    vals = [(lab, extra_time(rows, e, m, b, 1), extra_time(rows, e, m, b, 32))
            for lab, e, m, b in items if (e, m, 1) in rows and (e, b, 1) in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    h = 0.36
    hi = max(max(v1, v32) for _, v1, v32 in vals)
    for i, (lab, v1, v32) in enumerate(vals):
        for dy, v, c in ((-h / 2, v1, BLUE), (h / 2, v32, ORANGE)):
            v = max(v, 0.0)
            ax.barh(i + dy, v, h - 0.04, color=c, edgecolor="white")
            ax.text(v + hi * 0.015, i + dy, f"+{v:.1f}%" if v >= 0.05 else "0%",
                    va="center", fontsize=16, color=INK)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, hi * 1.22)
    ax.set_xlabel("extra time per token vs. the engine's default (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["batch 1", "batch 32"], loc="lower right", frameon=False,
              fontsize=15)
    fig.text(0.01, 0.01, "Qwen2.5-1.5B, NVIDIA L4, bf16.  *also turns off SGLang's "
             "overlap scheduler.", fontsize=12, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "results" / "results.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args()
    rows = load(a.csv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hooks_run(rows, out / "hooks_run.png")
    probe_cost(rows, out / "probe_cost.png")
    print(f"[plot_poster] wrote {out / 'hooks_run.png'} and {out / 'probe_cost.png'}")


if __name__ == "__main__":
    main()
