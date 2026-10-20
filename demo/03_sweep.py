"""Train one linear probe per layer; plot accuracy vs. depth, with error bars.

This is the actual result. Logistic regression maps a hidden vector ->
true/false. We fit one probe per layer and plot held-out accuracy vs. layer.

Evaluation protocol (why it looks like this):

  * Entity-grouped splits. Each row has a `group` (country / element / number
    pair, see 01_make_data.py). Matched true/false twins and same-entity rows
    share a group, and GroupShuffleSplit keeps whole groups on one side, so the
    probe can never score a statement whose twin it trained on.
  * Repeated over N_SEEDS different group splits; we report mean +/- SE across
    splits per layer and plot it as a band. One split of ~94 rows has roughly
    +/-0.03-0.05 noise on its own, which is the size of most layer-to-layer
    differences on the plateau.
  * No layer picking on test. Within each split's TRAIN set, a GroupKFold
    validation picks the layer; that layer is then scored once on the split's
    test set. The mean of those scores is the honest "probe at the chosen
    layer" number (nested estimate).
  * The probe saved for the live demo uses the layer chosen most often by the
    inner validation, refit on ALL data.
"""
from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ACTS_PATH, DATA_PATH, PLOT_PATH, PROBE_PATH, SEED

N_SEEDS = 10        # outer group splits
TEST_FRAC = 0.3     # fraction of GROUPS held out per split
INNER_FOLDS = 5     # GroupKFold folds inside train, for layer selection
PLATEAU_TOL = 0.02  # plateau = layers whose mean is within this of the best mean


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def fit_score(X, y, tr, te, L):
    clf = make_probe()
    clf.fit(X[tr, L, :], y[tr])
    return clf.predict(X[te, L, :])


def se(a, axis=0):
    a = np.asarray(a, dtype=float)
    return a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


def main() -> None:
    d = np.load(ACTS_PATH, allow_pickle=True)
    X, y, kind = d["X"], d["y"], d["kind"]   # X: [n, L, hidden]
    n, n_layers, _ = X.shape

    # 02_extract.py saves rows in statements.jsonl order; groups come from there
    rows = [json.loads(line) for line in open(DATA_PATH)]
    assert len(rows) == n and all(r["label"] == yy for r, yy in zip(rows, y)), \
        "statements.jsonl and activations.npz are out of sync; re-run 02"
    groups = np.array([r["group"] for r in rows])
    print(f"[sweep] {n} statements, {len(set(groups))} entity groups, "
          f"{n_layers} hidden-state layers")
    print(f"[sweep] {N_SEEDS} group-shuffle splits (test = {TEST_FRAC:.0%} of "
          f"groups), layer chosen by {INNER_FOLDS}-fold GroupKFold inside train")

    gss = GroupShuffleSplit(n_splits=N_SEEDS, test_size=TEST_FRAC,
                            random_state=SEED)
    outer = np.zeros((N_SEEDS, n_layers))     # test acc per split x layer
    inner = np.zeros((N_SEEDS, n_layers))     # inner-val acc per split x layer
    chosen = np.zeros(N_SEEDS, dtype=int)
    nested = np.zeros(N_SEEDS)                # test acc at the inner-chosen layer
    test_sizes, chances, test_idx = [], [], []
    pooled_kind, pooled_ok = [], []           # per-category, at chosen layer

    for s, (tr, te) in enumerate(gss.split(X[:, 0, :], y, groups)):
        assert not set(groups[tr]) & set(groups[te])
        test_sizes.append(len(te))
        test_idx.append(te)
        chances.append(max(y[te].mean(), 1 - y[te].mean()))
        preds = {}
        for L in range(n_layers):
            preds[L] = fit_score(X, y, tr, te, L)
            outer[s, L] = (preds[L] == y[te]).mean()
            accs = []
            for itr, iva in GroupKFold(INNER_FOLDS).split(tr, groups=groups[tr]):
                p = fit_score(X, y, tr[itr], tr[iva], L)
                accs.append((p == y[tr[iva]]).mean())
            inner[s, L] = np.mean(accs)
        chosen[s] = int(np.argmax(inner[s]))   # ties -> earliest layer
        nested[s] = outer[s, chosen[s]]
        pooled_kind.append(kind[te])
        pooled_ok.append(preds[chosen[s]] == y[te])
        print(f"    split {s}: test n={len(te):3d}  inner-chosen layer "
              f"{chosen[s]:2d}  -> test acc {nested[s]:.3f}")

    mean, err = outer.mean(0), se(outer)
    chance = float(np.mean(chances))
    best_mean = int(np.argmax(mean))
    # plateau: contiguous span from the first layer whose mean comes within
    # PLATEAU_TOL of the best mean (layer-to-layer differences inside it are
    # the same size as the split-to-split noise)
    near = mean >= mean[best_mean] - PLATEAU_TOL
    start = int(np.argmax(near))
    end = start
    while end + 1 < n_layers and near[end + 1]:
        end += 1
    plateau = list(range(start, end + 1))
    votes = Counter(chosen.tolist())
    deploy = min(votes, key=lambda L: (-votes[L], L))   # mode, ties -> earliest

    print(f"\n[sweep] test accuracy by layer (mean +/- SE over {N_SEEDS} splits, "
          f"test n ~{int(np.mean(test_sizes))}):")
    for L in range(n_layers):
        bar = "#" * int(round(mean[L] * 40))
        mark = "  <- plateau" if L in plateau else ""
        print(f"    layer {L:2d}: {mean[L]:.3f} +/- {err[L]:.3f} {bar}{mark}")
    print(f"\n[sweep] highest mean: layer {best_mean} = {mean[best_mean]:.3f} "
          f"+/- {err[best_mean]:.3f}")
    print(f"[sweep] plateau: layers {start}-{end} all within {PLATEAU_TOL} of "
          f"it (range {mean[start:end + 1].min():.3f}-"
          f"{mean[start:end + 1].max():.3f}); no single peak layer is "
          f"distinguishable from its neighbours")
    print(f"[sweep] inner-validation layer choices: "
          f"{dict(sorted(votes.items()))}")
    print(f"[sweep] NESTED estimate (layer chosen in train, scored on test): "
          f"{nested.mean():.3f} +/- {se(nested):.3f}  (chance {chance:.3f})")
    print(f"[sweep] deploy layer (most-chosen by inner validation) = {deploy}: "
          f"test acc {mean[deploy]:.3f} +/- {err[deploy]:.3f}")
    print("[sweep] note: SE across overlapping resampled splits understates "
          "true variance; read it as a lower bound on the noise.")

    # per-category accuracy at each split's chosen layer, pooled over splits
    pk, po = np.concatenate(pooled_kind), np.concatenate(pooled_ok)
    print(f"\n[sweep] per-category accuracy at the inner-chosen layer, pooled "
          f"over {N_SEEDS} splits:")
    print(f"    {'category':10s}  acc    +/-SE   n_pred  n_unique  n_groups")
    per_cat = {}
    for k in sorted(set(kind)):
        m = pk == k
        acc = po[m].mean()
        uniq = len({i for te in test_idx for i in te if kind[i] == k})
        n_groups = len(set(groups[kind == k]))
        # binomial SE on pooled predictions (statements repeat across splits,
        # so this is optimistic; n_unique is the real sample size)
        s_ = np.sqrt(acc * (1 - acc) / uniq)
        per_cat[k] = {"acc": float(acc), "n_pred": int(m.sum()),
                      "n_unique": uniq}
        print(f"    {k:10s}  {acc:.3f}  {s_:.3f}  {m.sum():6d}  {uniq:8d}  "
              f"{n_groups:8d}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for s in range(N_SEEDS):
        ax.plot(layers, outer[s], color=blue, lw=0.8, alpha=0.15)
    ax.fill_between(layers, mean - err, mean + err, color=blue, alpha=0.25,
                    lw=0, label="±1 SE across splits")
    ax.plot(layers, mean, "o-", color=blue, lw=2, ms=4,
            label=f"mean test accuracy ({N_SEEDS} entity-grouped splits)")
    ax.axhline(chance, ls="--", color=muted, lw=1, label=f"chance ({chance:.2f})")
    ax.axvspan(plateau[0] - 0.5, plateau[-1] + 0.5, color=muted, alpha=0.07,
               lw=0)
    ax.axvline(deploy, ls=":", color=ink, lw=1, alpha=0.7,
               label=f"layer chosen by inner validation ({deploy})")
    ax.text(0.99, 0.04,
            f"nested estimate {nested.mean():.2f} ± {se(nested):.2f}\n"
            f"shaded: plateau, layers {start}-{end} "
            f"(within {PLATEAU_TOL} of best mean)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=muted)
    ax.set_xlabel("hidden-state layer (0 = embeddings)", color=ink)
    ax.set_ylabel("held-out probe accuracy", color=ink)
    ax.set_title("Truthfulness linear probe: accuracy vs. layer depth",
                 color=ink)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.tick_params(colors=muted)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(muted)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130, facecolor=fig.get_facecolor())
    print(f"\n[sweep] saved plot -> {PLOT_PATH}")

    # refit deploy layer on ALL data and persist for live demo
    final = make_probe()
    final.fit(X[:, deploy, :], y)
    joblib.dump({"probe": final, "layer": deploy,
                 "test_acc": float(mean[deploy]),
                 "test_acc_se": float(err[deploy]),
                 "nested_acc": float(nested.mean()),
                 "nested_acc_se": float(se(nested)),
                 "n_splits": N_SEEDS,
                 "per_category": per_cat}, PROBE_PATH)
    print(f"[sweep] saved probe (layer {deploy}, refit on all data) -> "
          f"{PROBE_PATH}")


if __name__ == "__main__":
    main()
