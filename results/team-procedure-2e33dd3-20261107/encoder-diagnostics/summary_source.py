"""Summarize the fresh frozen-checkpoint diagnostics without training or selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from current_results import ROOT, current_campaign, require_current_path


def main():
    campaign = current_campaign()
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    out = require_current_path(ROOT / registry["encoder_diagnostics"])
    artifact = json.loads((out / "results.json").read_text())
    assert artifact["status"] == "VERIFIED_CURRENT" and len(artifact["records"]) == 12
    rows = list(csv.DictReader((out / "local_traces.csv").open()))
    datasets = ["email-Enron", "tags-ask-ubuntu"]
    models = {"allset_cpa_encoder": "Encoder CPA", "allset_cpa_both": "Both"}
    conditions = ["normal", "disabled", "constant3", "shuffled"]
    summaries, local = [], []
    for dataset in datasets:
        for model in models:
            runs = sorted([r for r in artifact["records"] if r["dataset"] == dataset and r["model"] == model], key=lambda r: r["seed"])
            assert [r["seed"] for r in runs] == [0, 1, 2]
            for condition in conditions:
                # Shuffles are averaged inside each seed first, never treated as nine seeds.
                values = {}
                for metric in ("auprc", "auroc", "delta_auprc", "delta_auroc", "mean_absolute_logit_change", "embedding_relative_l2"):
                    per_seed = []
                    for run in runs:
                        selected = [v for k, v in run["conditions"].items() if k.startswith(condition)]
                        def value(v):
                            if metric in ("auprc", "auroc"):
                                return v["evaluation"]["overall"][metric]
                            if metric == "embedding_relative_l2":
                                return v["embedding"]["mean_relative_l2"]
                            return v[metric]
                        per_seed.append(float(np.mean([value(v) for v in selected])))
                    values[metric] = {"mean": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed, ddof=1)), "per_seed": per_seed}
                summaries.append({"dataset": dataset, "model": model, "condition": condition, **values})
            for layer in (1, 2):
                for control in ("query_residual", "zero_query_residual"):
                    for multiplier in (1, 2, 3, 4, 8, "actual"):
                        for stage in ("modified", "pre_norm", "post_norm", "representation"):
                            filtered = [r for r in rows if (r["dataset"],r["model"],r["layer"],r["control"],r["multiplier"],r["stage"])
                                        == (dataset,model,str(layer),control,str(multiplier),stage)]
                            record = {"dataset": dataset, "model": model, "layer": layer, "control": control, "multiplier": str(multiplier), "stage": stage}
                            for metric in ("mean_norm", "mean_l2", "mean_relative_l2", "mean_cosine"):
                                vals = []
                                for seed in (0, 1, 2):
                                    selected = [r for r in filtered if int(r["seed"]) == seed]
                                    vals.append(float(np.average([float(r[metric]) for r in selected], weights=[int(r["count"]) for r in selected])))
                                record[metric] = {"mean": float(np.mean(vals)), "seed_sd": float(np.std(vals,ddof=1)), "per_seed": vals}
                            local.append(record)
    context_counts = []
    for dataset in datasets:
        for seed in (0, 1, 2):
            folder = out / dataset / f"seed-{seed}" / "allset_cpa_encoder"
            with np.load(folder / "normal-validation.npz") as a:
                sizes = a["encoder_multipliers"]
            fractions = []
            for rep in (0, 1, 2):
                with np.load(folder / f"shuffled{rep}-validation.npz") as a:
                    changed = a["encoder_multipliers"]
                fractions.append(float(np.mean(changed[sizes>1] != sizes[sizes>1])))
            context_counts.append({"dataset": dataset, "seed": seed,
                "size_counts_including_synthetic": {str(k): int(v) for k,v in zip(*np.unique(sizes,return_counts=True))},
                "shuffle_changed_real_edge_fraction": fractions})
    result = {"status": "VERIFIED_CURRENT", "test_evaluated": False, "downstream": summaries,
              "local": local, "context_counts": context_counts,
              "aggregation": "equal weight per seed; local edge-weighted means inside each seed; shuffle replicates averaged inside each seed"}
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Fresh encoder diagnostics (validation only)", "",
        "Frozen encoder-only and combined CPA checkpoints from the verified current campaign; both datasets; seeds 0,1,2. All 12 normal and restored validation logit replays are bitwise exact. Checkpoint files and in-memory weights remained unchanged. No retraining, test scoring, condition selection or legacy results.", "",
        "## Downstream interventions", "", "Only encoder node-to-hyperedge multipliers change. Synthetic singleton self-messages retain m=1; readout and classifier are unchanged. Shuffling permutes actual sizes 2/3 among real context edges, preserving counts; one permutation is reused across layers. Three deterministic shuffles are averaged within each seed before computing across-seed SD.", "",
        "| Dataset | Model | Condition | Validation AP | AP change (pp) | Validation AUROC | Mean absolute logit change |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['dataset']} | {models[r['model']]} | {r['condition']} | {r['auprc']['mean']:.4f} | {100*r['delta_auprc']['mean']:+.3f} +/- {100*r['delta_auprc']['seed_sd']:.3f} | {r['auroc']['mean']:.4f} | {r['mean_absolute_logit_change']['mean']:.4f} |")
    lines += ["", "## Local normalization check", "",
        "Normal upstream inputs and raw attention are fixed separately in each encoder layer. Compare m=3 with m=1 in the same checkpoint and real edge. The zero-query control removes only the query added to the pooled vector; the query used to compute attention is preserved. Local reconstructions exactly match source forward for actual sizes. Values below are mean relative L2 vector changes, not AP changes; first average real edges inside each seed, then average seeds.", "",
        "| Dataset | Model | Layer | Residual | After first LayerNorm | Final pool representation |",
        "|---|---|---:|---|---:|---:|"]
    for dataset in datasets:
        for model in models:
            for layer in (1,2):
                for control in ("query_residual", "zero_query_residual"):
                    selected = [r for r in local if (r["dataset"],r["model"],r["layer"],r["control"],r["multiplier"]) == (dataset,model,layer,control,"3")]
                    vals = [next(r["mean_relative_l2"]["mean"] for r in selected if r["stage"] == stage) for stage in ("post_norm", "representation")]
                    lines.append(f"| {dataset} | {models[model]} | {layer} | {control} | {vals[0]:.6g} | {vals[1]:.6g} |")
    lines += ["", "## Interpretation and limits", "",
        "The query residual allows multiplicative scaling to change direction and survive normalization. Removing only this residual almost eliminates local scale sensitivity; remaining differences are compatible with finite LayerNorm epsilon and floating-point effects. This refutes an assumption that LayerNorm necessarily erases CPA in this implementation.", "",
        "On AskUbuntu, disabling scaling is highly disruptive, but constant m=3 and shuffled real-edge sizes leave AP nearly unchanged. Thus dependence on encoder scaling does not establish dependence on assigning each edge its exact cardinality. Global scale, optimization under that scale, and real-edge versus synthetic-self-message differences remain possible explanations. Shuffling changes roughly 45% of real AskUbuntu edge multipliers, so this is a substantive intervention rather than an almost unchanged assignment.", "",
        "On Enron, shuffling sizes and disabling scaling reduce mean validation AP for encoder-only CPA, suggesting sensitivity to the size assignment despite mixed held-out test improvements. Computational dependence does not imply a predictive advantage over a separately trained baseline.", "",
        "The local zero-query experiment is a block-level control, not an end-to-end retrained architecture. Disabling scaling on CPA-trained weights is not trained vanilla AST. Interventions may move representations away from their training distribution. Validation was used for checkpoint selection; these are mechanism diagnostics, not independent generalization estimates. Neither shuffling nor constant scaling has been retrained as a model, and no condition is selected for test evaluation.", "",
        "![Local normalization sensitivity](normalization.png)"]
    (out / "SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})
    fig, axes = plt.subplots(1,2,figsize=(11,4.7))
    for ax, dataset in zip(axes,datasets):
        for layer,color in ((1,"#167c80"),(2,"#8755a0")):
            for control,style in (("query_residual","-"),("zero_query_residual","--")):
                selected = [r for r in local if r["dataset"]==dataset and r["model"]=="allset_cpa_encoder" and r["layer"]==layer and r["control"]==control and r["stage"]=="post_norm" and r["multiplier"] in ["2","3","4","8"]]
                selected.sort(key=lambda r:int(r["multiplier"]))
                ax.plot([int(r["multiplier"]) for r in selected],[r["mean_relative_l2"]["mean"] for r in selected],"o"+style,color=color,
                        label=f"Layer {layer}, {'query retained' if control=='query_residual' else 'residual removed'}")
                for r in selected:
                    ax.scatter([int(r["multiplier"])]*3,r["mean_relative_l2"]["per_seed"],s=12,alpha=.35,color=color)
        ax.set_yscale("log"); ax.set_xticks([2,3,4,8]); ax.set_xlabel("Multiplier m (reference m=1)")
        ax.set_ylabel("Relative L2 change after first LayerNorm")
        ax.set_title(dataset); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Frozen encoder-only CPA: local scaling survives through the query residual\nAttention held fixed; real context edges only; small dots show three seed means",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ("png","pdf"):
        fig.savefig(out/f"normalization.{ext}",dpi=180)
    plt.close(fig)
    (out/"summary_source.py").write_bytes(Path(__file__).read_bytes())
    print(f"Saved verified summaries and figure to {out}")


if __name__ == "__main__":
    main()
