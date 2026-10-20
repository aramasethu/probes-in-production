"""Poster chart: decode throughput change vs each engine's fast baseline.

    python plot.py                      # reads results/results.csv
    python plot.py --csv other.csv --out results/overhead

Writes <out>.png (300 dpi), <out>.pdf and <out>.md (the same numbers as a
table, for the poster text / speaker notes). Only needs pandas + matplotlib.

One panel per engine, one bar group per mode, one bar per batch size.
Bars are % change in output tok/s relative to the engine's baseline at the
same batch size (negative = slower). Whiskers span the p25..p75 tok/s of the
mode divided by the baseline median (the baseline's own spread is in the CSV).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent

# (engine, mode) -> (baseline mode, poster label). Order = plotting order.
SPEC = {
    "hf": [
        ("eager_device", "eager_none", "eager + probe on device"),
        ("eager_cpu", "eager_none", "eager + probe via .cpu()"),
        ("compile_ingraph", "compile_none", "compiled + probe in-graph"),
        ("compile_cpu", "compile_none", "compiled + probe via .cpu()"),
        ("eager_none", "compile_none", "eager (no probe) vs compiled"),
    ],
    "vllm": [
        ("eager", "default", "enforce_eager (no probe)"),
        ("eager_hook_device", "default", "eager + hook, probe on device"),
        ("eager_hook_cpu", "default", "eager + hook, probe via .cpu()"),
        ("default_hook_device", "default", "graphs on + hook (hook mostly silent)"),
        ("extract_hs", "default", "extract_hidden_states (to /dev/shm)"),
    ],
    "sglang": [
        ("nograph", "default", "CUDA graphs off (no probe)"),
        ("nograph_hook_device", "default", "graphs off + hook, probe on device"),
        ("nograph_hook_cpu", "default", "graphs off + hook, probe via .cpu()"),
        ("default_hook_device", "default", "graphs on + hook (hook mostly silent)"),
        ("return_hidden_last", "default", "return_hidden_states (final layer)"),
    ],
}
TITLES = {
    "hf": "HF transformers (baseline: same loop without probe)",
    "vllm": "vLLM (baseline: default, CUDA graphs on)",
    "sglang": "SGLang (baseline: default, CUDA graphs on)",
}
# categorical slots 1-3 of the reference palette (validated all-pairs, light)
BS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def overhead_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engine"] = df["engine"].str.replace("-smoke", "", regex=False)
    # last run wins if a mode was re-run
    df = df.sort_values("timestamp").drop_duplicates(["engine", "mode", "batch_size"], keep="last")
    key = df.set_index(["engine", "mode", "batch_size"])
    rows = []
    for eng, specs in SPEC.items():
        for mode, base, label in specs:
            for bs in sorted(df.loc[df.engine == eng, "batch_size"].unique()):
                try:
                    m, b = key.loc[(eng, mode, bs)], key.loc[(eng, base, bs)]
                except KeyError:
                    continue
                bm = b["tok_s_median"]
                rows.append(dict(
                    engine=eng, mode=mode, label=label, baseline=base, batch_size=int(bs),
                    tok_s=m["tok_s_median"], base_tok_s=bm,
                    pct=100 * (m["tok_s_median"] / bm - 1),
                    pct_lo=100 * (m["tok_s_p25"] / bm - 1),
                    pct_hi=100 * (m["tok_s_p75"] / bm - 1),
                    tpot_ms=m["tpot_ms_median"], base_tpot_ms=b["tpot_ms_median"],
                    ttft_ms=m["ttft_ms_median"], base_ttft_ms=b["ttft_ms_median"],
                    gpu=m["gpu"]))
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, out: Path) -> None:
    engines = [e for e in SPEC if (t.engine == e).any()]
    bss = sorted(t.batch_size.unique())
    plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans",
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    heights = [len(t[t.engine == e]["mode"].unique()) for e in engines]
    fig, axes = plt.subplots(len(engines), 1, figsize=(13, 1.4 + 1.25 * sum(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    lim_lo = min(-5, t.pct_lo.min() * 1.2)
    lim_hi = max(5, t.pct_hi.max() * 1.15)
    for ax, eng in zip(axes[:, 0], engines):
        sub = t[t.engine == eng]
        labels = [l for _, _, l in SPEC[eng] if l in set(sub.label)]
        gh = 0.8 / len(bss)
        for gi, lab in enumerate(labels):
            for bi, bs in enumerate(bss):
                r = sub[(sub.label == lab) & (sub.batch_size == bs)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = gi + (bi - (len(bss) - 1) / 2) * gh
                ax.barh(y, r.pct, height=gh - 0.04, color=BS_COLORS[bi % 3],
                        edgecolor="white", linewidth=1)
                ax.plot([r.pct_lo, r.pct_hi], [y, y], color=INK, lw=1.2, alpha=.6)
                ha = "left" if r.pct >= 0 else "right"
                pad = 0.012 * (lim_hi - lim_lo)          # label clears bar AND whisker
                xl = max(r.pct, r.pct_hi) + pad if r.pct >= 0 else min(r.pct, r.pct_lo) - pad
                ax.text(xl, y, f"{r.pct:+.1f}%",
                        va="center", ha=ha, fontsize=12, color=INK2)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_xlim(lim_lo, lim_hi)
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(TITLES[eng], loc="left", fontsize=16, fontweight="bold", pad=8)
    axes[-1, 0].set_xlabel("decode throughput vs baseline (% change in output tok/s; whisker = IQR)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BS_COLORS[i % 3]) for i in range(len(bss))]
    fig.legend(handles, [f"batch {b}" for b in bss], loc="upper left", ncol=len(bss),
               bbox_to_anchor=(0.005, 0.975), frameon=False, fontsize=14)
    gpu = t.gpu.iloc[0] if len(t) else ""
    fig.suptitle(f"Probe cost in production engines — Qwen2.5-1.5B-Instruct, bf16, {gpu}",
                 x=0.01, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, facecolor="white")
    print(f"[plot] wrote {out.with_suffix('.png')} and .pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "results" / "results.csv"))
    ap.add_argument("--out", default=str(HERE / "results" / "overhead"))
    a = ap.parse_args()
    t = overhead_table(pd.read_csv(a.csv))
    if t.empty:
        raise SystemExit("no (mode, baseline) pairs found in the CSV yet")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(t, out)
    cols = ["engine", "label", "batch_size", "tok_s", "base_tok_s", "pct",
            "tpot_ms", "base_tpot_ms", "ttft_ms", "base_ttft_ms"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t[cols].round(2).iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(t[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
