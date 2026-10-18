from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from .train import write_json


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path):
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("result.json"))]
    if not results:
        raise ValueError(f"No completed runs under {root}")
    rows, groups, paired = [], defaultdict(list), defaultdict(dict)
    for result in results:
        meta = {k: result[k] for k in ("dataset", "protocol", "k", "model", "seed")}
        for split in ("val", "test"):
            data = result["evaluation"][split]
            for size, metrics in [("all", data["overall"]), *data["by_size"].items()]:
                row = {**meta, "split": split, "size": size, "n": metrics["n"],
                       "positives": metrics["positives"], "small_bin": metrics["positives"] < 20,
                       "auroc": metrics["auroc"],
                       "auprc": metrics["auprc"], "parameters": result["training"]["parameters"],
                       "training_seconds": result["training"]["training_seconds_all_trials"]}
                rows.append(row)
                key = tuple(row[k] for k in ("dataset", "protocol", "k", "model", "split", "size"))
                groups[key].append(row)
                if split == "test":
                    pair_key = (meta["dataset"], meta["protocol"], meta["k"], meta["seed"], size)
                    paired[pair_key][meta["model"]] = row
    summary = []
    for key, records in groups.items():
        row = dict(zip(("dataset", "protocol", "k", "model", "split", "size"), key))
        row["seeds"] = len(records)
        row["mean_positives"] = float(np.mean([r["positives"] for r in records]))
        row["small_bin_any_seed"] = any(r["small_bin"] for r in records)
        for metric in ("auroc", "auprc", "training_seconds", "parameters"):
            values = [r[metric] for r in records if r[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    deltas = []
    for key, models in paired.items():
        if "allset" in models and "allset_cpa" in models:
            deltas.append({**dict(zip(("dataset", "protocol", "k", "seed", "size"), key)),
                           "delta_auprc": models["allset_cpa"]["auprc"] - models["allset"]["auprc"],
                           "delta_auroc": models["allset_cpa"]["auroc"] - models["allset"]["auroc"]})
    _csv(root / "runs.csv", rows)
    _csv(root / "summary.csv", summary)
    _csv(root / "paired_cpa_deltas.csv", deltas)
    plots = root / "figures"
    plots.mkdir(exist_ok=True)
    for dataset, protocol, k in sorted({(r["dataset"], r["protocol"], r["k"]) for r in results}):
        selected = [r for r in summary if r["dataset"] == dataset and r["protocol"] == protocol
                    and r["k"] == k and r["split"] == "test" and r["size"] != "all"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for model in sorted({r["model"] for r in selected}):
            points = sorted([r for r in selected if r["model"] == model], key=lambda x: int(x["size"]))
            sizes = np.array([int(r["size"]) for r in points])
            x = sizes - k if protocol == "order" else sizes
            for ax, metric in zip(axes, ("auprc", "auroc")):
                means = np.array([r[f"{metric}_mean"] for r in points])
                std = np.array([r[f"{metric}_std"] or 0. for r in points])
                line, = ax.plot(x, means, "-", label=model)
                small = np.array([r["small_bin_any_seed"] for r in points])
                ax.scatter(x[~small], means[~small], s=17, color=line.get_color())
                ax.scatter(x[small], means[small], s=20, facecolors="none", edgecolors=line.get_color())
                ax.fill_between(x, np.maximum(0, means - std), np.minimum(1, means + std), alpha=.12)
                ax.set(xlabel="Order gap |e| - k" if protocol == "order" else "Hyperedge cardinality",
                       ylabel=metric.upper(), ylim=(0, 1.02))
                ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{dataset}: {protocol}, k={k}; bands: across-seed SD\nOpen markers: fewer than 20 positives in at least one seed", fontsize=10)
        fig.savefig(plots / f"{dataset}-{protocol}-k{k}.png", dpi=170)
        plt.close(fig)
    replication, probe_rows, intervention_rows = defaultdict(list), [], []
    for path in root.rglob("interpretability.json"):
        metadata = json.loads((path.parent / "result.json").read_text()) if (path.parent / "result.json").exists() else None
        if metadata is None:
            continue
        content = json.loads(path.read_text())
        meta = {k: metadata[k] for k in ("dataset", "protocol", "k", "model", "seed")}
        probes = content["probes"]
        for stage, methods in probes.get("stages", {}).items():
            for method, partitions in methods.items():
                for split, metrics in partitions.items():
                    probe_rows.append({**meta, "stage": stage, "method": method,
                                       "split": split, **metrics})
        intervention = content["size_intervention"]
        intervention_rows.append({**meta,
            "mean_absolute_logit_change": intervention["mean_absolute_logit_change"],
            "original_auprc": intervention["original"]["overall"]["auprc"],
            "neutralized_auprc": intervention["neutralized"]["overall"]["auprc"]})
        for row in content["replication"]["records"]:
            key = (metadata["dataset"], metadata["protocol"], metadata["k"], metadata["model"],
                   row["stage"], row["factor"])
            replication[key].append(row["relative_l2_from_base"])
    replication_rows = [{**dict(zip(("dataset", "protocol", "k", "model", "stage", "factor"), key)),
                          "relative_l2_mean": float(np.mean(values))} for key, values in replication.items()]
    _csv(root / "replication.csv", replication_rows)
    _csv(root / "cardinality_probes.csv", probe_rows)
    _csv(root / "size_interventions.csv", intervention_rows)
    for dataset, protocol, k in sorted({(r["dataset"], r["protocol"], r["k"]) for r in replication_rows}):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        models = sorted({r["model"] for r in replication_rows if r["dataset"] == dataset and
                         r["protocol"] == protocol and r["k"] == k and r["model"].startswith("allset")})
        for model in models:
            for stage in ("modified", "decoder_input"):
                points = sorted([r for r in replication_rows if r["dataset"] == dataset and
                    r["protocol"] == protocol and r["k"] == k and r["model"] == model and
                    r["stage"] == stage], key=lambda r: r["factor"])
                if points:
                    ax.plot([r["factor"] for r in points], [r["relative_l2_mean"] for r in points],
                            "o-", label=f"{model}: {stage}", markersize=3)
        ax.set(xlabel="Feature-multiset replication factor", ylabel="Relative L2 change from original",
               title=f"{dataset}: {protocol}, k={k}; frozen-node diagnostic")
        if ax.lines:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=.2)
        fig.savefig(plots / f"{dataset}-{protocol}-k{k}-replication.png", dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        stage_order = ["attention", "modified", "post_norm", "representation", "decoder_input"]
        for model in models:
            means, stds = [], []
            for stage in stage_order:
                values = [r["log_size_mae"] for r in probe_rows if r["dataset"] == dataset and
                          r["protocol"] == protocol and r["k"] == k and r["model"] == model and
                          r["method"] == "linear" and r["split"] == "val" and r["stage"] == stage]
                means.append(float(np.mean(values)) if values else np.nan)
                stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.)
            if np.isfinite(means).any():
                ax.errorbar(stage_order, means, yerr=stds, fmt="o-", label=model, capsize=3)
        ax.set(ylabel="Validation MAE of log(1 + size); lower is better",
               title=f"{dataset}: {protocol}, k={k}; train-fitted linear probes")
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(plots / f"{dataset}-{protocol}-k{k}-probes.png", dpi=170)
        plt.close(fig)
    overall = [r for r in summary if r["split"] == "test" and r["size"] == "all"]
    lines = ["# Experiment results", "", "Mean and sample standard deviation across split/training seeds.",
             "One-seed runs are validation runs, not evidence of a stable effect.", "",
             "| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |", "|---|---|---|---|---|---|---|"]
    for r in overall:
        def fmt(metric):
            value, sd = r[f"{metric}_mean"], r[f"{metric}_std"]
            return f"{value:.4f}" + (f" +/- {sd:.4f}" if sd is not None else " (SD unavailable)")
        lines.append(f"| {r['dataset']} | {r['protocol']} | {r['k']} | {r['model']} | {r['seeds']} | {fmt('auprc')} | {fmt('auroc')} |")
    lines += ["", "AP depends on the configured negative prevalence. Compare only matched protocols/candidates.",
              "Small per-size bins are flagged in each run's result.json; seed SD is not a confidence interval.",
              "Interpretability diagnostics and computational interventions are in each model's interpretability.json.",
              "See paired_cpa_deltas.csv for within-seed changes and replication.csv for normalization diagnostics."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "report_manifest.json", {"completed_runs": len(results), "summary_rows": len(summary)})
    return root / "REPORT.md"
