"""Read-only follow-up checks. Run from the GML folder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200, "display.max_rows", 200)
pd.options.display.float_format = "{:.4f}".format
CORE, HARD, EASY = (Path("results") / s for s in ("5e68c177dfb8", "26d38b1e9b65", "10651eeb61ac"))


def pair(df, a, b, **f):
    q = df
    for k, v in f.items():
        q = q[q[k] == v]
    return q[q.model == a].set_index("seed").auprc - q[q.model == b].set_index("seed").auprc


print("##### 1. Hard negatives: CPA minus AllSet, val vs test, and epoch of selection")
h = pd.read_csv(HARD / "runs.csv")
h["size"] = h["size"].astype(str)
for ds in h.dataset.unique():
    for split in ("val", "test"):
        d = pair(h, "allset_cpa", "allset", dataset=ds, split=split, size="all")
        print(ds, split, round(d.mean(), 4), round(d.std(), 4), np.round(d.values, 4).tolist())
c = pd.read_csv(HARD / "convergence.csv")
s = c[(c.dataset == "tags-ask-ubuntu") & (c.learning_rate == .001)].pivot_table(index="epoch", columns="model", values="val_auprc")
print(s.iloc[[0, 5, 10, 15, 20, 25, 30, -1]].to_string())

print("\n##### 2. Paired model gaps, test AP overall, core")
r = pd.read_csv(CORE / "runs.csv")
r["size"] = r["size"].astype(str)
rows = []
for ds in r.dataset.unique():
    for p in r.protocol.unique():
        for a, b in (("gcn", "allset"), ("allset", "deepsets"), ("gcn", "deepsets")):
            d = pair(r, a, b, dataset=ds, protocol=p, split="test", size="all")
            rows.append({"dataset": ds, "protocol": p, "pair": f"{a}-{b}", "delta": d.mean(), "sd": d.std(), "pos": int((d > 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))

print("\n##### 3. Fixed threshold FPR drift, order k=3, AskUbuntu: size minus val, per seed")
rows = []
for m in ("deepsets", "gcn", "allset", "allset_cpa", "allset_size"):
    for seed in (0, 1, 2):
        base = CORE / "tags-ask-ubuntu" / "order-k3" / f"seed-{seed}" / m
        te, va = np.load(base / "test_predictions.npz"), np.load(base / "val_predictions.npz")
        vs, vl = va["logits"], va["labels"]
        o = np.argsort(-vs)
        tp = np.cumsum(vl[o])
        prec, rec = tp / np.arange(1, len(vs) + 1), tp / vl.sum()
        thr = vs[o][(2 * prec * rec / np.maximum(prec + rec, 1e-9)).argmax()]
        vfpr = (vs[vl == 0] >= thr).mean()
        row = {"model": m, "seed": seed, "val": vfpr}
        for sz in (4, 5):
            mask = te["sizes"] == sz
            row[f"s{sz}"] = (te["logits"][mask][te["labels"][mask] == 0] >= thr).mean() - vfpr
        rows.append(row)
f = pd.DataFrame(rows)
print(f.groupby("model").agg(val=("val", "mean"), s4=("s4", "mean"), s4sd=("s4", "std"), s5=("s5", "mean"), s5sd=("s5", "std"),
                             s5pos=("s5", lambda x: int((x > 0).sum()))).to_string())

print("\n##### 4. Probe MAE at each stage, test, linear probe, order")
p = pd.read_csv(CORE / "cardinality_probes.csv")
q = p[(p.protocol == "order") & (p.split == "test") & (p.method == "linear")]
print(q.pivot_table(index=["dataset", "stage"], columns="model", values="log_size_mae").round(3).to_string())
q = p[(p.protocol == "order") & (p.split == "test") & (p.method == "norm_only")]
print(q.pivot_table(index=["dataset", "stage"], columns="model", values="log_size_mae").round(3).to_string())
print("seeds per model at core:", r.groupby("model").seed.nunique().to_dict())
