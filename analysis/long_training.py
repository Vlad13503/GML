"""Compare the 600-epoch AskUbuntu order-shift runs with the 150-epoch runs. Run from the repo root."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200, "display.max_rows", 200)
pd.options.display.float_format = "{:.4f}".format

R = Path("results")
LONG = {"allset": "51ca6ceec20c", "allset_cpa": "7d63d1aa8d89", "allset_cpa_encoder": "284cd40b976d", "gcn": "e5b43490c297"}
SHORT = {"allset": "382679f494ad", "allset_cpa": "382679f494ad", "allset_cpa_encoder": "382679f494ad", "gcn": "5e68c177dfb8"}


def fpr_drift(base):
    te, va = np.load(base / "test_predictions.npz"), np.load(base / "val_predictions.npz")
    vs, vl = va["logits"], va["labels"]
    o = np.argsort(-vs)
    tp = np.cumsum(vl[o])
    prec, rec = tp / np.arange(1, len(vs) + 1), tp / vl.sum()
    thr = vs[o][(2 * prec * rec / np.maximum(prec + rec, 1e-9)).argmax()]
    mask = te["sizes"] == 5
    return (te["logits"][mask][te["labels"][mask] == 0] >= thr).mean() - (vs[vl == 0] >= thr).mean()


rows = []
for model in LONG:
    for horizon, suite in ((600, LONG[model]), (150, SHORT[model])):
        for seed in (0, 1, 2):
            base = R / suite / "tags-ask-ubuntu" / "order-k3" / f"seed-{seed}" / model
            r = json.loads((base / "result.json").read_text())
            e = r["evaluation"]
            row = {"model": model, "horizon": horizon, "seed": seed, "sha": r["split_sha256"][:10],
                   "best_epoch": r["training"]["best_epoch"], "lr": r["training"]["selected_learning_rate"],
                   "val_ap": e["val"]["overall"]["auprc"], "test_ap": e["test"]["overall"]["auprc"], "test_auroc": e["test"]["overall"]["auroc"],
                   "s4": e["test"]["by_size"]["4"]["auprc"], "s5": e["test"]["by_size"]["5"]["auprc"], "fpr_drift_s5": fpr_drift(base)}
            rows.append(row)
df = pd.DataFrame(rows)

print("##### Same splits per seed across suites?")
print(df.groupby("seed").sha.nunique().to_dict(), "(1 means identical)")

print("\n##### Per run")
print(df.sort_values(["model", "seed", "horizon"]).drop(columns="sha").to_string(index=False))

print("\n##### Mean +/- SD by horizon")
print(df.groupby(["model", "horizon"]).agg(best=("best_epoch", "mean"), bmax=("best_epoch", "max"), val=("val_ap", "mean"), test=("test_ap", "mean"), test_sd=("test_ap", "std"),
                                            auroc=("test_auroc", "mean"), s4=("s4", "mean"), s5=("s5", "mean"), fpr=("fpr_drift_s5", "mean"), fpr_sd=("fpr_drift_s5", "std")).to_string())

print("\n##### Gain from longer training, paired by seed")
for model in LONG:
    a = df[(df.model == model) & (df.horizon == 600)].set_index("seed")
    b = df[(df.model == model) & (df.horizon == 150)].set_index("seed")
    d = a.test_ap - b.test_ap
    print(model, "test", round(d.mean(), 4), round(d.std(), 4), np.round(d.values, 4).tolist(), "| val", round((a.val_ap - b.val_ap).mean(), 4))

print("\n##### Paired comparisons at 600 epochs (150-epoch value in brackets)")
for x, y in (("allset_cpa", "allset"), ("allset_cpa_encoder", "allset"), ("gcn", "allset"), ("allset_cpa_encoder", "allset_cpa")):
    out = []
    for h in (600, 150):
        q = df[df.horizon == h]
        for col in ("test_ap", "val_ap"):
            d = q[q.model == x].set_index("seed")[col] - q[q.model == y].set_index("seed")[col]
            out.append(f"{h} {col}: {d.mean():+.4f} +/- {d.std():.4f} ({int((d > 0).sum())}/3)")
    print(f"{x} - {y}\n   " + "\n   ".join(out))

print("\n##### Training curve, LR=0.001, val AP, mean over seeds")
frames = []
for model, suite in LONG.items():
    c = pd.read_csv(R / suite / "convergence.csv")
    frames.append(c[c.learning_rate == .001])
c = pd.concat(frames)
pv = c.pivot_table(index="epoch", columns="model", values="val_auprc")
print(pv.loc[[e for e in pv.index if e % 50 == 0]].to_string())

print("\n##### Interventions at 600 epochs")
for model in ("allset_cpa",):
    s = pd.read_csv(R / LONG[model] / "size_interventions.csv")
    print(s[["seed", "mean_absolute_logit_change", "original_auprc", "neutralized_auprc"]].to_string(index=False))
    p = pd.read_csv(R / LONG[model] / "cardinality_probes.csv")
    q = p[(p.split == "test") & (p.method.isin(["linear", "norm_only"])) & (p.stage.isin(["modified", "post_norm", "representation"]))]
    print(q.pivot_table(index="stage", columns="method", values="log_size_mae").round(3).to_string())
