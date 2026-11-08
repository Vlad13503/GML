"""Validate and summarize a frozen multiplier sweep, aggregating edges within seeds first."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hnn_project.train import write_json


def summarize(root):
    groups = defaultdict(list)
    control_max = defaultdict(float)
    expected_max_error = 0.
    count = 0
    with (root / "per_edge.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            key = (row["dataset"], row["model"], int(row["seed"]), int(row["layer"]),
                   row["condition"], int(row["multiplier"]), row["stage"])
            metrics = {name: float(row[name]) for name in ("norm", "l2", "relative_l2", "cosine")}
            if not all(np.isfinite(v) for k, v in metrics.items() if k != "cosine"):
                raise ValueError("Non-finite norm or distance")
            groups[(*key, "all")].append(metrics)
            groups[(*key, row["actual_size"])].append(metrics)
            if row["model"] == "allset" or row["condition"] == "scaling_disabled":
                control_max["invariant_relative_l2"] = max(control_max["invariant_relative_l2"], metrics["relative_l2"])
            if row["condition"] == "residual_zero" and row["stage"] in ("post_norm", "representation"):
                control_max[row["stage"] + "_zero_residual_relative_l2"] = max(
                    control_max[row["stage"] + "_zero_residual_relative_l2"], metrics["relative_l2"])
            if row["stage"] == "attention" or int(row["multiplier"]) == 1:
                if metrics["l2"] != 0:
                    raise ValueError("Raw aggregate or reference changed")
            if row["model"] == "allset_cpa_encoder" and row["condition"] != "scaling_disabled" and row["stage"] == "modified":
                expected_max_error = max(expected_max_error, abs(metrics["relative_l2"] - (int(row["multiplier"]) - 1)))
    if control_max["invariant_relative_l2"] != 0 or expected_max_error > 1e-4:
        raise ValueError("Invariance or scaling formula control failed")
    per_seed = []
    aggregate = defaultdict(list)
    for key, values in groups.items():
        dataset, model, seed, layer, condition, multiplier, stage, size = key
        row = dict(zip(("dataset", "model", "seed", "layer", "condition", "multiplier", "stage", "actual_size"), key))
        row["n_edges"] = len(values)
        for metric in ("norm", "l2", "relative_l2", "cosine"):
            finite = [v[metric] for v in values if np.isfinite(v[metric])]
            row[metric] = float(np.mean(finite)) if finite else None
        per_seed.append(row)
        aggregate[(dataset, model, layer, condition, multiplier, stage, size)].append(row)
    summary = []
    for key, seeds in aggregate.items():
        row = dict(zip(("dataset", "model", "layer", "condition", "multiplier", "stage", "actual_size"), key))
        row["seeds"] = [r["seed"] for r in seeds]
        for metric in ("norm", "l2", "relative_l2", "cosine"):
            values = [r[metric] for r in seeds if r[metric] is not None]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    with (root / "per_seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    write_json(root / "summary.json", {"metric_rows": count, "control_maxima": dict(control_max),
        "scaling_formula_max_absolute_error": expected_max_error,
        "aggregation": "Edge means within each seed, then mean/sample SD of three seed means",
        "records": summary})

    def curve(dataset, model, condition, stage, metric):
        rows = sorted([r for r in summary if r["dataset"] == dataset and r["model"] == model
                       and r["condition"] == condition and r["stage"] == stage and r["actual_size"] == "all"],
                      key=lambda r: r["multiplier"])
        # The trained models have one encoder layer; prevent accidentally pooling layers.
        assert len({r["layer"] for r in rows}) == 1
        return (np.array([r["multiplier"] for r in rows]),
                np.array([r[metric + "_mean"] for r in rows]),
                np.array([r[metric + "_seed_sd"] for r in rows]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row_id, dataset in enumerate(("email-Enron", "tags-ask-ubuntu")):
        for col, series in enumerate((
                [("allset_cpa_encoder", "actual", "modified", "m a", "#666666"),
                 ("allset_cpa_encoder", "actual", "pre_norm", "q + m a", "#2878b5")],
                [("allset_cpa_encoder", "actual", "post_norm", "After LN1", "#2878b5"),
                 ("allset_cpa_encoder", "actual", "representation", "Full block", "#d66b21"),
                 ("allset_cpa_encoder", "residual_zero", "representation", "Full block: residual q=0", "#38864a")],
                [("allset_cpa_encoder", "actual", "representation", "Full block", "#d66b21"),
                 ("allset_cpa_encoder", "residual_zero", "representation", "Residual q=0", "#38864a"),
                 ("allset", "actual", "representation", "Vanilla (m ignored)", "#666666")])):
            ax = axes[row_id, col]
            metric = "cosine" if col == 2 else "relative_l2"
            for model, condition, stage, label, color in series:
                x, means, sd = curve(dataset, model, condition, stage, metric)
                ax.plot(x, means, "o-", color=color, label=label, markersize=3)
                low, high = means - sd, means + sd
                if metric == "cosine":
                    low, high = np.clip(low, -1, 1), np.clip(high, -1, 1)
                else:
                    low = np.maximum(0, low)
                ax.fill_between(x, low, high, color=color, alpha=.12)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
            ax.set(xlabel="Artificial absolute multiplier m", title=dataset,
                   ylabel="Cosine to m=1" if col == 2 else "Relative L2 to m=1")
            if col == 2:
                ax.set_ylim(-.1, 1.04)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Fixed attention aggregate; context sizes 2/3 balanced within each seed\n"
                 "Mean and sample SD of three seed means; independent y scales across stages", fontsize=11)
    fig.savefig(root / "normalization_response.png", dpi=180)
    plt.close(fig)
    lines = ["# Artificial cardinality multiplier diagnostic", "",
        "Frozen vanilla and encoder-CPA checkpoints; 128 real context edges per checkpoint "
        "(64 each of sizes 2 and 3), across both datasets and all three seeds. "
        "The same edge IDs are used in paired models. Synthetic singleton self-messages are excluded.", "",
        "Each attention aggregate is captured once; only the multiplier and the stated residual control change. "
        "This is a local pooling-block intervention, not a prediction or generalization test.", "",
        "| Dataset | m | Full-block relative L2 | Residual-zero relative L2 | Full-block cosine |",
        "|---|---:|---:|---:|---:|"]
    for dataset in ("email-Enron", "tags-ask-ubuntu"):
        for m in (2, 3, 8, 32):
            actual = next(r for r in summary if r["dataset"] == dataset and r["model"] == "allset_cpa_encoder"
                          and r["condition"] == "actual" and r["stage"] == "representation"
                          and r["actual_size"] == "all" and r["multiplier"] == m)
            control = next(r for r in summary if r["dataset"] == dataset and r["model"] == "allset_cpa_encoder"
                           and r["condition"] == "residual_zero" and r["stage"] == "representation"
                           and r["actual_size"] == "all" and r["multiplier"] == m)
            lines.append(f"| {dataset} | {m} | {actual['relative_l2_mean']:.4f} +/- {actual['relative_l2_seed_sd']:.4f} | "
                         f"{control['relative_l2_mean']:.2e} | {actual['cosine_mean']:.4f} |")
    lines += ["", "Distances use m=1 within the same edge, checkpoint, layer and control condition. "
              "Reported variation is sample SD across seed means, not an uncertainty estimate based on independent edges.", "",
              "Both invariant controls have exactly zero L2 change. The modified aggregate follows the expected "
              "relative distance m-1. Finite epsilon leaves very small normalized differences in the zero-residual control.", "",
              "With the learned query residual present, multiplier-dependent directional differences survive both LayerNorms. "
              "Response to m flattens at larger values. This rules out complete erasure in these sampled blocks, "
              "but does not establish globally decodable cardinality or explain the task-score gains. "
              "Stage-wise relative distances are normalized against their own references and are not percentages of information retained."]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"metric_rows": count, "control_maxima": dict(control_max),
                      "scaling_formula_max_absolute_error": expected_max_error}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
