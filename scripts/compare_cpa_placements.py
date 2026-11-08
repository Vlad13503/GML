"""Compare frozen vanilla, encoder-only CPA, and readout-only CPA test results."""
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
    vanilla = Path("results/ast-test") / suite_id
    encoder = Path("results/encoder-cpa") / suite_id
    readout = Path("results/readout-cpa") / suite_id
    output = Path("results/cpa-placement") / suite_id
    output.mkdir(parents=True, exist_ok=True)
    records = []
    source = []
    for dataset in ("email-Enron", "tags-ask-ubuntu"):
        for seed in (0, 1, 2):
            base_dir, enc_dir, read_dir = [root / dataset / f"seed-{seed}" for root in (vanilla, encoder, readout)]
            base = json.loads((base_dir / "test_result.json").read_text())
            enc = json.loads((enc_dir / "result.json").read_text())
            read = json.loads((read_dir / "result.json").read_text())
            assert base["split_sha256"] == enc["identity"]["split_sha256"] == read["identity"]["split_sha256"]
            for key in ("baseline_checkpoint_sha256", "initial_state_sha256"):
                assert enc["identity"][key] == read["identity"][key]
            assert {k: v for k, v in enc["identity"]["config"].items() if k != "models"} == {
                k: v for k, v in read["identity"]["config"].items() if k != "models"}
            with np.load(base_dir / "test_predictions.npz") as base_arrays:
                for candidate_dir in (enc_dir, read_dir):
                    with np.load(candidate_dir / "test_predictions.npz") as arrays:
                        np.testing.assert_array_equal(base_arrays["labels"], arrays["labels"])
                        np.testing.assert_array_equal(base_arrays["sizes"], arrays["sizes"])
            evaluation = {"vanilla": base["evaluation"], "encoder": enc["evaluation"]["test"],
                          "readout": read["evaluation"]["test"]}
            source.append({"dataset": dataset, "seed": seed, "evaluation": evaluation})
            for metric in ("auprc", "auroc"):
                values = {model: e["overall"][metric] for model, e in evaluation.items()}
                records.append({"dataset": dataset, "seed": seed, "metric": metric, **values,
                    "readout_minus_vanilla": values["readout"] - values["vanilla"],
                    "readout_minus_encoder": values["readout"] - values["encoder"]})
    summary = []
    for dataset in ("email-Enron", "tags-ask-ubuntu"):
        for metric in ("auprc", "auroc"):
            rows = [r for r in records if r["dataset"] == dataset and r["metric"] == metric]
            entry = {"dataset": dataset, "metric": metric}
            for model in ("vanilla", "encoder", "readout", "readout_minus_vanilla", "readout_minus_encoder"):
                values = [r[model] for r in rows]
                entry[model] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)),
                                "per_seed": values}
            summary.append(entry)
    write_json(output / "comparison.json", {"records": summary, "matched_inputs_verified": True})
    with (output / "paired_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, dataset in zip(axes, ("email-Enron", "tags-ask-ubuntu")):
        selected = [r for r in source if r["dataset"] == dataset]
        sizes = sorted({int(s) for r in selected for s in r["evaluation"]["vanilla"]["by_size"]})
        for model, label, color in (("vanilla", "Vanilla AST", "#2878b5"),
                                    ("encoder", "Encoder CPA", "#d66b21"),
                                    ("readout", "Readout CPA", "#38864a")):
            means, sds, small = [], [], []
            for size in sizes:
                bins = [r["evaluation"][model]["by_size"][str(size)] for r in selected]
                means.append(np.mean([b["auprc"] for b in bins]))
                sds.append(np.std([b["auprc"] for b in bins], ddof=1))
                small.append(any(b["small_bin"] for b in bins))
            x = np.array(sizes) - 3
            means, sds, small = np.array(means), np.array(sds), np.array(small)
            ax.plot(x, means, color=color, label=label)
            ax.fill_between(x, np.clip(means-sds, 0, 1), np.clip(means+sds, 0, 1), color=color, alpha=.1)
            ax.scatter(x[~small], means[~small], color=color, s=18)
            ax.scatter(x[small], means[small], edgecolors=color, facecolors="none", s=23)
        ax.set(title=dataset, xlabel="Order gap (cardinality - 3)", ylabel="Test AUPRC", ylim=(0, 1.02))
        if dataset == "tags-ask-ubuntu":
            ax.set_xticks([1, 2])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Matched AST settings; mean and sample SD across three seeds\n"
                 "Open markers: fewer than 20 positives in at least one seed", fontsize=10)
    fig.savefig(output / "test_auprc_by_order_gap.png", dpi=180)
    plt.close(fig)
    lines = ["# CPA placement comparison", "",
        "Same frozen AST settings, exact candidates, initial shared weights and paired seeds. "
        "Both CPA variants train end to end from initialization and independently select checkpoints using validation AP.", "",
        "| Dataset | Metric | Vanilla AST | Encoder CPA | Readout CPA | Readout - vanilla | Readout - encoder |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        cells = [f"{row[k]['mean']:.4f} +/- {row[k]['seed_sd']:.4f}" for k in
                 ("vanilla", "encoder", "readout", "readout_minus_vanilla", "readout_minus_encoder")]
        lines.append(f"| {row['dataset']} | {row['metric'].upper()} | " + " | ".join(cells) + " |")
    lines += ["", "Differences are paired within seeds. Variation is sample SD across three seeds, "
              "not confidence intervals. Test positives have cardinality >3 with 1:1 size-matched negatives. "
              "No variant-specific tuning or test-based configuration selection was performed.", "",
              "These are placement comparisons under vanilla-selected settings, not separately optimized architectures. "
              "Per-cardinality small-bin limits remain important, especially for Enron."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output / "comparison_source.py").write_bytes(Path(__file__).read_bytes())
    print((output / "SUMMARY.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", default="7d0356f3d97c")
    compare(parser.parse_args().suite_id)
