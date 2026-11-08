"""Four-way matched comparison: vanilla, encoder, readout, and joint CPA."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hnn_project.train import write_json


def compare(suite_id):
    roots = {"vanilla": Path("results/ast-test") / suite_id,
             "encoder": Path("results/encoder-cpa") / suite_id,
             "readout": Path("results/readout-cpa") / suite_id,
             "both": Path("results/both-cpa") / suite_id}
    output = Path("results/cpa-placement-both") / suite_id
    output.mkdir(parents=True, exist_ok=True)
    source, runs = [], []
    for dataset in ("email-Enron", "tags-ask-ubuntu"):
        for seed in (0, 1, 2):
            folders = {model: root / dataset / f"seed-{seed}" for model, root in roots.items()}
            records = {model: json.loads((folder / ("test_result.json" if model == "vanilla" else "result.json")).read_text())
                       for model, folder in folders.items()}
            identities = [records[model]["identity"] for model in ("encoder", "readout", "both")]
            for key in ("split_sha256", "baseline_checkpoint_sha256", "initial_state_sha256"):
                assert len({r[key] for r in identities}) == 1
            assert identities[0]["split_sha256"] == records["vanilla"]["split_sha256"]
            normalized_configs = [{k: v for k, v in r["config"].items() if k != "models"} for r in identities]
            assert normalized_configs[0] == normalized_configs[1] == normalized_configs[2]
            with np.load(folders["vanilla"] / "test_predictions.npz") as reference:
                for model in ("encoder", "readout", "both"):
                    with np.load(folders[model] / "test_predictions.npz") as arrays:
                        np.testing.assert_array_equal(reference["labels"], arrays["labels"])
                        np.testing.assert_array_equal(reference["sizes"], arrays["sizes"])
            evaluation = {model: r["evaluation"] if model == "vanilla" else r["evaluation"]["test"]
                          for model, r in records.items()}
            source.append({"dataset": dataset, "seed": seed, "evaluation": evaluation})
            for metric in ("auprc", "auroc"):
                values = {model: e["overall"][metric] for model, e in evaluation.items()}
                runs.append({"dataset": dataset, "seed": seed, "metric": metric, **values,
                    "both_minus_vanilla": values["both"] - values["vanilla"],
                    "both_minus_encoder": values["both"] - values["encoder"],
                    "both_minus_readout": values["both"] - values["readout"]})
    summary = []
    for dataset in ("email-Enron", "tags-ask-ubuntu"):
        for metric in ("auprc", "auroc"):
            selected = [r for r in runs if r["dataset"] == dataset and r["metric"] == metric]
            entry = {"dataset": dataset, "metric": metric}
            for key in ("vanilla", "encoder", "readout", "both", "both_minus_vanilla", "both_minus_encoder", "both_minus_readout"):
                values = [r[key] for r in selected]
                entry[key] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)), "per_seed": values}
            summary.append(entry)
    write_json(output / "comparison.json", {"records": summary, "matched_inputs_verified": True})
    with (output / "paired_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, dataset in zip(axes, ("email-Enron", "tags-ask-ubuntu")):
        selected = [r for r in source if r["dataset"] == dataset]
        sizes = sorted({int(s) for r in selected for s in r["evaluation"]["vanilla"]["by_size"]})
        for model, label, color in (("vanilla", "Vanilla AST", "#2878b5"), ("encoder", "Encoder CPA", "#d66b21"),
                                    ("readout", "Readout CPA", "#38864a"), ("both", "Encoder + readout CPA", "#9250a2")):
            means, sds, small = [], [], []
            for size in sizes:
                bins = [r["evaluation"][model]["by_size"][str(size)] for r in selected]
                means.append(np.mean([b["auprc"] for b in bins]))
                sds.append(np.std([b["auprc"] for b in bins], ddof=1))
                small.append(any(b["small_bin"] for b in bins))
            x, means, sds, small = np.array(sizes)-3, np.array(means), np.array(sds), np.array(small)
            ax.plot(x, means, color=color, label=label)
            ax.fill_between(x, np.clip(means-sds, 0, 1), np.clip(means+sds, 0, 1), color=color, alpha=.08)
            ax.scatter(x[~small], means[~small], color=color, s=17)
            ax.scatter(x[small], means[small], edgecolors=color, facecolors="none", s=23)
        ax.set(title=dataset, xlabel="Order gap (cardinality - 3)", ylabel="Test AUPRC", ylim=(0, 1.02))
        if dataset == "tags-ask-ubuntu":
            ax.set_xticks([1, 2])
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Matched AST settings; mean and sample SD across three seeds\n"
                 "Open markers: fewer than 20 positives in at least one seed", fontsize=10)
    fig.savefig(output / "test_auprc_by_order_gap.png", dpi=180)
    plt.close(fig)
    lines = ["# Four-way CPA placement comparison", "",
        "All variants use the same frozen AST settings, contexts, candidates, shared initial weights and seeds. "
        "Every model is trained end to end from initialization; checkpoints use validation AP only. "
        "Both means CPA at encoder v2e and candidate readout, before their respective residual/LayerNorm blocks. "
        "Hyperedge-to-node pooling remains vanilla. Mean +/- sample SD across three seeds; not confidence intervals.", "",
        "| Dataset | Metric | Vanilla AST | Encoder CPA | Readout CPA | Both |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        cells = [f"{row[k]['mean']:.4f} +/- {row[k]['seed_sd']:.4f}" for k in ("vanilla", "encoder", "readout", "both")]
        lines.append(f"| {row['dataset']} | {row['metric'].upper()} | " + " | ".join(cells) + " |")
    lines += ["", "| Dataset | Metric | Both - vanilla | Both - encoder | Both - readout |", "|---|---|---:|---:|---:|"]
    for row in summary:
        cells = [f"{row[k]['mean']:+.4f} +/- {row[k]['seed_sd']:.4f}"
                 for k in ("both_minus_vanilla", "both_minus_encoder", "both_minus_readout")]
        lines.append(f"| {row['dataset']} | {row['metric'].upper()} | " + " | ".join(cells) + " |")
    lines += ["", "Differences are paired within seeds. This is a fixed-setting ablation, "
              "not a comparison of independently optimized architectures. Test positives have cardinality >3, "
              "with closed-node filtering and 1:1 size-matched negatives. No settings were changed using test scores."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output / "comparison_source.py").write_bytes(Path(__file__).read_bytes())
    print((output / "SUMMARY.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", default="7d0356f3d97c")
    compare(parser.parse_args().suite_id)
