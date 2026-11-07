"""Read-only analysis of finished suites. Run from the GML folder."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 40)
pd.options.display.float_format = "{:.4f}".format

CORE, ABL, K4, EASY, HARD = (Path("results") / s for s in (
    "5e68c177dfb8", "382679f494ad", "f571005e3814", "10651eeb61ac", "26d38b1e9b65"))
MODELS = ["deepsets", "gcn", "allset", "allset_cpa", "allset_size"]


def runs(root):
    df = pd.read_csv(root / "runs.csv")
    df["size"] = df["size"].astype(str)
    return df


core = runs(CORE)
k4 = runs(K4)
abl = runs(ABL)


def head(text):
    print(f"\n\n##### {text}")


head("A. Test bins per size: positives per seed and sparse flag (random, order)")
t = core[(core.split == "test") & (core["size"] != "all") & (core.model == "allset")]
print(t.groupby(["dataset", "protocol", "size"]).agg(pos=("positives", "mean"), sparse=("small_bin", "max")).reset_index()
      .pivot_table(index=["dataset", "size"], columns="protocol", values="pos").to_string())

head("B. Matched-size AP: same size, order vs random vs interpolation (mean over seeds)")
sizes = {"email-Enron": ["4", "5", "6", "7", "8"], "tags-ask-ubuntu": ["4", "5"]}
rows = []
for ds, ss in sizes.items():
    for s in ss:
        for m in MODELS:
            sel = core[(core.dataset == ds) & (core.model == m) & (core["size"] == s) & (core.split == "test")]
            row = {"dataset": ds, "size": s, "model": m}
            for p in ("random", "order", "interpolation"):
                v = sel[sel.protocol == p].auprc
                row[p] = v.mean() if len(v) else np.nan
                row[p + "_sd"] = v.std() if len(v) else np.nan
            v = sel[sel.protocol == "order"].set_index("seed").auprc - sel[sel.protocol == "random"].set_index("seed").auprc
            row["gap"] = v.mean()
            row["gap_sd"] = v.std()
            row["gap_neg_seeds"] = int((v < 0).sum())
            rows.append(row)
print(pd.DataFrame(rows).drop(columns=["interpolation_sd"]).to_string(index=False))

head("B2. Same as B for AUROC gap (order - random)")
rows = []
for ds, ss in sizes.items():
    for s in ss:
        for m in MODELS:
            sel = core[(core.dataset == ds) & (core.model == m) & (core["size"] == s) & (core.split == "test")]
            v = sel[sel.protocol == "order"].set_index("seed").auroc - sel[sel.protocol == "random"].set_index("seed").auroc
            rows.append({"dataset": ds, "size": s, "model": m, "gap": v.mean(), "sd": v.std(), "neg": int((v < 0).sum())})
print(pd.DataFrame(rows).pivot_table(index=["dataset", "size"], columns="model", values="gap")[MODELS].to_string())

head("C. Sizes seen in training, random protocol, for context (test AP by size)")
sel = core[(core.protocol == "random") & (core.split == "test") & (core["size"].isin(["2", "3", "4", "5", "6", "7", "8"]))]
print(sel.pivot_table(index=["dataset", "size"], columns="model", values="auprc")[MODELS].to_string())

head("D. k=4 vs k=3 at size 5 and above (order protocol, test AP)")
rows = []
for ds, ss in (("email-Enron", ["5", "6", "7", "8"]), ("tags-ask-ubuntu", ["5"])):
    for s in ss:
        for m in MODELS:
            a = core[(core.dataset == ds) & (core.model == m) & (core["size"] == s) & (core.split == "test") & (core.protocol == "order")].set_index("seed").auprc
            b = k4[(k4.dataset == ds) & (k4.model == m) & (k4["size"] == s) & (k4.split == "test")].set_index("seed").auprc
            d = b - a
            rows.append({"dataset": ds, "size": s, "model": m, "k3": a.mean(), "k4": b.mean(), "k4_minus_k3": d.mean(), "sd": d.std(), "pos_seeds": int((d > 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))

head("E. Best epoch and learning rate: how often the 150-epoch cap binds")
rows = []
for name, root in (("core", CORE), ("abl", ABL), ("k4", K4), ("easy", EASY), ("hard", HARD)):
    for path in root.glob("*/*/seed-*/*/result.json"):
        r = json.loads(path.read_text())
        rows.append({"suite": name, "dataset": r["dataset"], "protocol": r["protocol"], "model": r["model"], "seed": r["seed"],
                     "best": r["training"]["best_epoch"], "lr": r["training"]["selected_learning_rate"],
                     "secs": r["training"]["training_seconds_all_trials"], "params": r["training"]["parameters"]})
tr = pd.DataFrame(rows)
tr["capped"] = tr.best >= 140
print(tr.groupby(["dataset", "model"]).agg(runs=("best", "size"), capped=("capped", "sum"), mean_best=("best", "mean"), lr001=("lr", lambda x: (x == .001).sum()),
                                            secs=("secs", "median"), params=("params", "first")).to_string())
print("\nAskUbuntu core, by protocol: capped/total")
print(tr[(tr.suite == "core") & (tr.dataset == "tags-ask-ubuntu")].groupby(["protocol", "model"]).capped.agg(["sum", "size"]).unstack(0).to_string())

head("F. CPA placement: val vs test paired deltas per seed (ablation)")
rows = []
for ds in ("email-Enron", "tags-ask-ubuntu"):
    for m in ("allset_cpa", "allset_cpa_encoder", "allset_cpa_both"):
        for split in ("val", "test"):
            sel = abl[(abl.dataset == ds) & (abl["size"] == "all") & (abl.split == split)]
            d = sel[sel.model == m].set_index("seed").auprc - sel[sel.model == "allset"].set_index("seed").auprc
            rows.append({"dataset": ds, "variant": m, "split": split, "delta_AP": d.mean(), "sd": d.std(), "pos_seeds": int((d > 0).sum()), "values": np.round(d.values, 4).tolist()})
print(pd.DataFrame(rows).to_string(index=False))

head("F2. CPA placement training speed, AskUbuntu, seed mean of LR=0.001 val AP by epoch")
conv = pd.read_csv(ABL / "convergence.csv")
sel = conv[(conv.dataset == "tags-ask-ubuntu") & (conv.learning_rate == .001)]
print(sel.pivot_table(index="epoch", columns="model", values="val_auprc").to_string())
print(tr[(tr.suite == "abl") & (tr.dataset == "tags-ask-ubuntu")].groupby("model").agg(best=("best", "mean"), capped=("capped", "sum"), lr=("lr", "mean")).to_string())

head("F3. Convergence, Enron, LR=0.001, val AP by epoch")
sel = conv[(conv.dataset == "email-Enron") & (conv.learning_rate == .001)]
print(sel.pivot_table(index="epoch", columns="model", values="val_auprc").iloc[::2].to_string())

head("G. Negative-sampling: CPA minus AllSet within each suite, per seed")
rows = []
for frac, root in ((0., EASY), (.5, CORE), (1., HARD)):
    df = runs(root)
    for ds in ("email-Enron", "tags-ask-ubuntu"):
        sel = df[(df.dataset == ds) & (df.protocol == "order") & (df.split == "test")]
        for s in sizes[ds] + ["all"]:
            q = sel[sel["size"] == s]
            d = q[q.model == "allset_cpa"].set_index("seed").auprc - q[q.model == "allset"].set_index("seed").auprc
            rows.append({"frac": frac, "dataset": ds, "size": s, "allset": q[q.model == "allset"].auprc.mean(), "cpa": q[q.model == "allset_cpa"].auprc.mean(),
                         "delta": d.mean(), "sd": d.std(), "pos": int((d > 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))

head("H. Size interventions per size (order)")
sr = pd.read_csv(CORE / "size_response.csv")
print(sr[(sr.protocol == "order") & (sr.model.isin(["allset_cpa", "allset_size"])) & (sr["size"] <= 8)].to_string(index=False))

head("I. Cardinality probe per stage (order, test, best-known method rows)")
pr = pd.read_csv(CORE / "cardinality_probes.csv")
print(pr.method.unique(), pr.stage.unique(), pr.split.unique())
sel = pr[(pr.protocol == "order") & (pr.split == "test")]
print(sel.pivot_table(index=["dataset", "stage", "method"], columns="model", values="log_size_r2").round(3).to_string())

head("J. Correlation across dataset, protocol, seed: probe quality vs AP, within dataset")
stage = "decoder" if "decoder" in pr.stage.unique() else pr.stage.unique()[-1]
print("stage", stage)
q = pr[(pr.split == "test") & (pr.stage == stage) & (pr.method == pr.method.unique()[0])].merge(
    core[(core.split == "test") & (core["size"] == "all")], on=["dataset", "protocol", "model", "seed"])
for (ds, p), g in q.groupby(["dataset", "protocol"]):
    print(ds, p, "spearman(probe MAE, AP) =", round(g.log_size_mae.corr(g.auprc, method="spearman"), 3), "n =", len(g))

head("K. Fixed threshold (chosen on validation, best F1) applied by test size, from saved predictions")
sample = np.load(next(CORE.glob("tags-ask-ubuntu/order-k3/seed-0/allset/test_predictions.npz")), allow_pickle=True)
print("npz keys", sample.files)
rows = []
for ds in ("email-Enron", "tags-ask-ubuntu"):
    for m in MODELS:
        for seed in (0, 1, 2):
            base = CORE / ds / "order-k3" / f"seed-{seed}" / m
            te, va = np.load(base / "test_predictions.npz", allow_pickle=True), np.load(base / "val_predictions.npz", allow_pickle=True)
            keys = te.files
            sc = next(k for k in keys if k in ("scores", "logits", "score", "logit", "predictions"))
            lb = next(k for k in keys if k in ("labels", "label", "y"))
            sz = next(k for k in keys if k in ("sizes", "size"))
            vs, vl = va[sc], va[lb]
            order = np.argsort(-vs)
            tp = np.cumsum(vl[order])
            prec = tp / np.arange(1, len(vs) + 1)
            rec = tp / vl.sum()
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            thr = vs[order][f1.argmax()]
            for s in sizes[ds]:
                mask = te[sz] == int(s)
                neg = te[lb][mask] == 0
                pos = te[lb][mask] == 1
                rows.append({"dataset": ds, "model": m, "seed": seed, "size": s, "FPR": (te[sc][mask][neg] >= thr).mean(), "TPR": (te[sc][mask][pos] >= thr).mean(),
                             "neg_score": te[sc][mask][neg].mean(), "pos_score": te[sc][mask][pos].mean()})
            for s, split, data in (("val", "val", va),):
                neg, pos = data[lb] == 0, data[lb] == 1
                rows.append({"dataset": ds, "model": m, "seed": seed, "size": "val", "FPR": (data[sc][neg] >= thr).mean(), "TPR": (data[sc][pos] >= thr).mean(),
                             "neg_score": data[sc][neg].mean(), "pos_score": data[sc][pos].mean()})
fx = pd.DataFrame(rows)
print(fx.pivot_table(index=["dataset", "model"], columns="size", values="FPR").round(3).to_string())
print(fx.pivot_table(index=["dataset", "model"], columns="size", values="TPR").round(3).to_string())
print(fx.pivot_table(index=["dataset", "model"], columns="size", values="neg_score").round(3).to_string())
