"""Train one linear probe per layer; plot accuracy vs. depth.

This is the actual result. Logistic regression maps a hidden vector ->
true/false. We fit one probe per layer on a train split, score on a held-out
test split, and plot accuracy vs. layer. The signal should rise from chance in
early layers and peak in the middle/late layers -- that curve shows the
truth signal is localized in the network, which is the whole point of probing.

The best layer's probe (+ its scaler) is refit on ALL data and saved for the
live demo.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ACTS_PATH, PLOT_PATH, PROBE_PATH, SEED


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def main() -> None:
    d = np.load(ACTS_PATH, allow_pickle=True)
    X, y, kind = d["X"], d["y"], d["kind"]   # X: [n, L, hidden]
    n, n_layers, _ = X.shape
    print(f"[sweep] {n} statements, {n_layers} hidden-state layers")

    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED,
                              stratify=y)
    chance = max(y[te].mean(), 1 - y[te].mean())
    print(f"[sweep] train {len(tr)} / test {len(te)}  |  chance = {chance:.3f}")

    accs = np.zeros(n_layers)
    for L in range(n_layers):
        clf = make_probe()
        clf.fit(X[tr, L, :], y[tr])
        accs[L] = clf.score(X[te, L, :], y[te])

    best = int(np.argmax(accs))
    print("\n[sweep] test accuracy by layer:")
    for L in range(n_layers):
        bar = "#" * int(round(accs[L] * 40))
        mark = "  <-- best" if L == best else ""
        print(f"    layer {L:2d}: {accs[L]:.3f} {bar}{mark}")
    print(f"\n[sweep] best layer = {best}  acc = {accs[best]:.3f}  "
          f"(chance {chance:.3f})")

    # per-category accuracy at the best layer (on the test split)
    best_clf = make_probe()
    best_clf.fit(X[tr, best, :], y[tr])
    pred = best_clf.predict(X[te, best, :])
    print(f"\n[sweep] per-category accuracy at best layer {best} (test split):")
    for k in sorted(set(kind[te])):
        m = kind[te] == k
        print(f"    {k:10s}: {(pred[m] == y[te][m]).mean():.3f}  (n={m.sum()})")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(n_layers), accs, "o-", color="#2b6cb0", lw=2)
    ax.axhline(chance, ls="--", color="gray", label=f"chance ({chance:.2f})")
    ax.axvline(best, ls=":", color="#e53e3e", alpha=0.7,
               label=f"best layer {best} ({accs[best]:.2f})")
    ax.set_xlabel("hidden-state layer (0 = embeddings)")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_title("Truthfulness linear probe: accuracy vs. layer depth")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    print(f"[sweep] saved plot -> {PLOT_PATH}")

    # refit best layer on ALL data and persist for live demo
    final = make_probe()
    final.fit(X[:, best, :], y)
    joblib.dump({"probe": final, "layer": best, "test_acc": float(accs[best])},
                PROBE_PATH)
    print(f"[sweep] saved best-layer probe (layer {best}) -> {PROBE_PATH}")


if __name__ == "__main__":
    main()
