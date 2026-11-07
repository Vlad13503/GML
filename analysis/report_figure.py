"""Builds report/figures/order_gap.pdf and .png from the core suite. Run from the repo root."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORE = Path("results/5e68c177dfb8")
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"allset": "#2a78d6", "gcn": "#eb6834", "deepsets": "#1baf7a"}
NAMES = {"deepsets": "DeepSets", "gcn": "GCN", "allset": "AllSet", "allset_cpa": "AllSet+CPA", "allset_size": "AllSet+size"}

runs = pd.read_csv(CORE / "runs.csv")
runs["size"] = runs["size"].astype(str)
test = runs[runs.split == "test"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1, 1.15]})

# Same-size gap: order shift minus random split, AskUbuntu.
models = list(NAMES)
for offset, size, face in ((-.14, "4", "white"), (.14, "5", INK)):
    means, sds = [], []
    q = test[(test.dataset == "tags-ask-ubuntu") & (test["size"] == size)]
    for model in models:
        m = q[q.model == model]
        gap = m[m.protocol == "order"].set_index("seed").auprc - m[m.protocol == "random"].set_index("seed").auprc
        means.append(gap.mean())
        sds.append(gap.std())
    left.errorbar(np.arange(len(models)) + offset, means, yerr=sds, fmt="o", ms=4.5, mfc=face, mec=INK, ecolor=MUTED,
                  elinewidth=.9, capsize=0, label=f"Size {size} (gap {int(size) - 3})")
left.axhline(0, color=MUTED, lw=.8)
left.set_xticks(range(len(models)), [NAMES[m] for m in models], rotation=20, ha="right")
left.set_ylabel("AP: order shift - random split")
left.set_title("AskUbuntu, same hyperedge size", color=INK, fontsize=8.5, loc="left")
left.grid(axis="y", color=GRID, lw=.6)
left.set_axisbelow(True)
left.legend(frameon=False, loc="upper left", fontsize=7, ncol=1, handletextpad=.1, labelspacing=.25)

# Performance by unseen size, Enron order shift.
sizes = ["4", "5", "6", "7", "8"]
x = np.array([int(s) for s in sizes])
for model, color in COLORS.items():
    q = test[(test.dataset == "email-Enron") & (test.protocol == "order") & (test.model == model) & (test["size"].isin(sizes))]
    stat = q.groupby("size").auprc.agg(["mean", "std"]).loc[sizes]
    right.plot(x, stat["mean"], color=color, lw=2, marker="o", ms=4, mec="white", mew=.8, label=NAMES[model])
    right.fill_between(x, stat["mean"] - stat["std"], stat["mean"] + stat["std"], color=color, alpha=.12, lw=0)
    right.annotate(NAMES[model], (8, stat["mean"].iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
right.set_xticks(x)
right.set_xlim(3.8, 9.0)
right.set_xlabel("Test hyperedge size (trained on sizes 2-3)")
right.set_ylabel("Test AP")
right.set_title("Enron, order shift (band: seed SD)", color=INK, fontsize=8.5, loc="left")
right.grid(axis="y", color=GRID, lw=.6)
right.set_axisbelow(True)
right.legend(frameon=False, loc="upper center", ncol=3, fontsize=7, handlelength=1.4, columnspacing=.9, bbox_to_anchor=(.42, 1.0))
right.set_ylim(.55, .93)

fig.tight_layout(pad=.4, w_pad=1.5)
fig.savefig(OUT / "order_gap.pdf")
fig.savefig(OUT / "order_gap.png", dpi=200)
