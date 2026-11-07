from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from .train import MIN_BIN_POSITIVES, write_json

CPA_VARIANTS = ("allset_cpa", "allset_cpa_encoder", "allset_cpa_both", "allset_additive", "allset_size")
SUMMARY_KEY = ("dataset", "protocol", "k", "model", "split", "size")


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None)


def _fmt(value, sd=None):
    if value is None:
        return "n/a"
    return f"{value:.4f}" + (f" +/- {sd:.4f}" if sd is not None else "")


def _tag(result: dict) -> str:
    tag = f"{result['protocol']}-k{result['k']}"
    if result["protocol"] == "interpolation":
        tag += "-" + "-".join(map(str, result["interpolation_bounds"]))
    return tag


def load_suite(root: Path) -> list[dict]:
    """Load one suite, refusing results that were not produced by it."""
    if not (root / "config.json").exists():
        nested = sorted(p.parent.name for p in root.rglob("config.json"))
        raise ValueError(f"{root} is not a single suite. Report one of {nested}, or use compare.")
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("result.json"))]
    if not results:
        raise ValueError(f"No completed runs under {root}")
    foreign = {r.get("suite") for r in results} - {root.name}
    if foreign:
        raise ValueError(f"{root} contains results from other suites: {sorted(map(str, foreign))}")
    splits = defaultdict(set)
    for r in results:
        splits[(r["dataset"], r["protocol"], r["k"], r["seed"])].add(r["split_sha256"])
    mismatched = [key for key, hashes in splits.items() if len(hashes) > 1]
    if mismatched:
        raise ValueError(f"Models in {root} did not share candidates for: {mismatched}")
    return results


def _convergence(root: Path, results: list[dict], plots: Path):
    rows, finals = [], defaultdict(list)
    for r in results:
        path = root / r["dataset"] / _tag(r) / f"seed-{r['seed']}" / r["model"] / "training.json"
        training = json.loads(path.read_text(encoding="utf-8"))
        meta = {k: r[k] for k in ("dataset", "protocol", "k", "model", "seed")}
        selected = next(t for t in training["trials"]
                        if t["learning_rate"] == training["selected_learning_rate"])
        for point in selected["history"]:
            rows.append({**meta, "learning_rate": selected["learning_rate"], "epoch": point["epoch"],
                         "loss": point["loss"], "train_bce": point["train"]["bce"],
                         "train_auprc": point["train"]["auprc"], "val_bce": point["val"]["bce"],
                         "val_auprc": point["val"]["auprc"]})
        finals[(r["dataset"], r["protocol"], r["k"], r["model"])].append(
            (selected["history"][0]["val"]["auprc"], training["validation_auprc"],
             training["best_epoch"], training["never_improved"]))
    _csv(root / "convergence.csv", rows)
    summary = [{**dict(zip(("dataset", "protocol", "k", "model"), key)), "seeds": len(values),
                "epoch0_val_auprc_mean": _mean_sd([v[0] for v in values])[0],
                "best_val_auprc_mean": _mean_sd([v[1] for v in values])[0],
                "best_epoch_mean": _mean_sd([v[2] for v in values])[0],
                "seeds_never_improved": sum(v[3] for v in values)} for key, values in finals.items()]
    _csv(root / "convergence_summary.csv", summary)
    for dataset, protocol, k in sorted({(r["dataset"], r["protocol"], r["k"]) for r in results}):
        selected = [r for r in rows if (r["dataset"], r["protocol"], r["k"]) == (dataset, protocol, k)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for model in sorted({r["model"] for r in selected}):
            curves = defaultdict(list)
            for r in selected:
                if r["model"] == model:
                    curves[r["epoch"]].append(r)
            epochs = sorted(curves)
            for ax, metric in zip(axes, ("val_auprc", "train_bce")):
                ax.plot(epochs, [_mean_sd([r[metric] for r in curves[e]])[0] for e in epochs], label=model)
                ax.set(xlabel="Epoch", ylabel=metric)
                ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{dataset}: {protocol}, k={k}; seed mean at the selected learning rate", fontsize=10)
        fig.savefig(plots / f"{dataset}-{protocol}-k{k}-convergence.png", dpi=170)
        plt.close(fig)
    return summary


def _synthesize(summary, probe_rows, intervention_rows, replication_rows, probe_baselines):
    """Join cardinality diagnostics with test performance for each model."""
    rows = []
    for perf in summary:
        if perf["split"] != "test" or perf["size"] != "all":
            continue
        key = tuple(perf[k] for k in ("dataset", "protocol", "k", "model"))
        match = lambda r: tuple(r[k] for k in ("dataset", "protocol", "k", "model")) == key
        probes = [r["log_size_mae"] for r in probe_rows if match(r) and r["stage"] == "representation"
                  and r["method"] == "linear" and r["split"] == "test"]
        interventions = [r for r in intervention_rows if match(r)]
        replication = {r["stage"]: r["relative_l2_mean"] for r in replication_rows
                       if match(r) and r["factor"] == 8}
        rows.append({**dict(zip(("dataset", "protocol", "k", "model"), key)),
            "test_auprc_mean": perf["auprc_mean"], "test_auroc_mean": perf["auroc_mean"],
            "probe_test_mae_mean": _mean_sd(probes)[0],
            "constant_baseline_test_mae_mean": _mean_sd(probe_baselines.get(key, []))[0],
            "replication_modified_factor8": replication.get("modified"),
            "replication_decoder_factor8": replication.get("decoder_input"),
            "logit_change_mean": _mean_sd([r["mean_absolute_logit_change"] for r in interventions])[0],
            "neutralized_auprc_change_mean": _mean_sd([r["neutralized_auprc"] - r["original_auprc"]
                                                       for r in interventions])[0]})
    return rows


def compare(suites: list[Path], output: Path):
    """Set matched results from separate suites side by side, labelled by what differs."""
    configs = [json.loads((root / "config.json").read_text(encoding="utf-8")) for root in suites]
    keys = sorted({k for c in configs for k in c} - {"datasets", "models", "experiments"})
    differing = [k for k in keys if len({json.dumps(c.get(k), sort_keys=True) for c in configs}) > 1]
    if len(suites) < 2 or not differing:
        raise ValueError("Comparison needs at least two suites whose settings differ")
    tables = []
    for root in suites:
        path = root / "summary.csv"
        if not path.exists():
            raise ValueError(f"Run report on {root} first")
        with path.open(encoding="utf-8", newline="") as stream:
            tables.append({tuple(r[k] for k in SUMMARY_KEY): r for r in csv.DictReader(stream)})
    shared = set.intersection(*map(set, tables))
    if not shared:
        raise ValueError("The suites share no matching results")
    rows = []
    for root, config, table in zip(suites, configs, tables):
        label = {k: json.dumps(config.get(k), sort_keys=True) for k in differing}
        rows += [{"suite": root.name, **label, **table[key]} for key in sorted(shared)]
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "comparison.csv", rows)
    header = ["Dataset", "Protocol", "k", "Model", *differing, "Seeds", "Test AP", "Test AUROC"]
    lines = ["# Suite comparison", "", f"Settings that differ: {', '.join(differing)}.",
             "Only results present in every suite are included.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in sorted(rows, key=lambda r: [r[k] for k in ("dataset", "protocol", "k", "model", *differing)]):
        if r["split"] == "test" and r["size"] == "all":
            number = lambda name: _fmt(float(r[f"{name}_mean"]), float(r[f"{name}_std"]) if r[f"{name}_std"] else None)
            values = [r["dataset"], r["protocol"], r["k"], r["model"], *[r[k] for k in differing],
                      r["seeds"], number("auprc"), number("auroc")]
            lines.append("| " + " | ".join(map(str, values)) + " |")
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "COMPARISON.md"


def report(root: Path):
    results = load_suite(root)
    rows, groups, paired = [], defaultdict(list), defaultdict(dict)
    for result in results:
        meta = {k: result[k] for k in ("dataset", "protocol", "k", "model", "seed")}
        for split in ("val", "test"):
            data = result["evaluation"][split]
            for size, metrics in [("all", data["overall"]), *data["by_size"].items()]:
                row = {**meta, "split": split, "size": size, "n": metrics["n"],
                       "positives": metrics["positives"],
                       "small_bin": metrics["positives"] < MIN_BIN_POSITIVES,
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
    deltas, delta_groups = [], defaultdict(list)
    for key, models in paired.items():
        if "allset" not in models:
            continue
        for variant in CPA_VARIANTS:
            if variant not in models or None in (models[variant]["auprc"], models["allset"]["auprc"]):
                continue
            delta = {**dict(zip(("dataset", "protocol", "k", "seed", "size"), key)), "variant": variant,
                     "positives": models["allset"]["positives"],
                     "small_bin": models["allset"]["small_bin"],
                     "delta_auprc": models[variant]["auprc"] - models["allset"]["auprc"],
                     "delta_auroc": models[variant]["auroc"] - models["allset"]["auroc"]}
            deltas.append(delta)
            delta_groups[(key[0], key[1], key[2], key[4], variant)].append(delta)
    delta_summary = []
    for key, records in delta_groups.items():
        row = dict(zip(("dataset", "protocol", "k", "size", "variant"), key))
        row.update(seeds=len(records), small_bin_any_seed=any(r["small_bin"] for r in records))
        for metric in ("delta_auprc", "delta_auroc"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = _mean_sd([r[metric] for r in records])
            row[f"{metric}_seeds_positive"] = sum(r[metric] > 0 for r in records)
        delta_summary.append(row)
    _csv(root / "runs.csv", rows)
    _csv(root / "summary.csv", summary)
    _csv(root / "paired_cpa_deltas.csv", deltas)
    _csv(root / "paired_cpa_summary.csv", delta_summary)
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
    convergence = _convergence(root, results, plots)
    replication, probe_rows, intervention_rows = defaultdict(list), [], []
    probe_baselines, size_records = defaultdict(list), defaultdict(list)
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
        if "constant_train_mean_baseline" in probes:
            probe_baselines[tuple(meta[k] for k in ("dataset", "protocol", "k", "model"))].append(
                probes["constant_train_mean_baseline"]["test"]["log_size_mae"])
        for size, values in intervention["by_size"].items():
            size_records[(*[meta[k] for k in ("dataset", "protocol", "k", "model")], size)].append({
                "logit_change": values["mean_absolute_logit_change"],
                "original_auroc": intervention["original"]["by_size"][size]["auroc"],
                "neutralized_auroc": intervention["neutralized"]["by_size"][size]["auroc"],
                "positives": intervention["original"]["by_size"][size]["positives"]})
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
    synthesis = _synthesize(summary, probe_rows, intervention_rows, replication_rows, probe_baselines)
    _csv(root / "interpretability_summary.csv", synthesis)
    _csv(root / "size_response.csv", [{
        **dict(zip(("dataset", "protocol", "k", "model", "size"), key)),
        "mean_positives": float(np.mean([r["positives"] for r in records])),
        "small_bin_any_seed": any(r["positives"] < MIN_BIN_POSITIVES for r in records),
        **{f"{name}_mean": _mean_sd([r[name] for r in records])[0]
           for name in ("logit_change", "original_auroc", "neutralized_auroc")}}
        for key, records in size_records.items()])
    overall = [r for r in summary if r["split"] == "test" and r["size"] == "all"]
    lines = ["# Experiment results", "", "Mean and sample standard deviation across split/training seeds.",
             "One-seed runs are validation runs, not evidence of a stable effect.", "",
             "| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |", "|---|---|---|---|---|---|---|"]
    for r in overall:
        def fmt(metric):
            value, sd = r[f"{metric}_mean"], r[f"{metric}_std"]
            return _fmt(value, sd) if sd is not None else f"{_fmt(value)} (SD unavailable)"
        lines.append(f"| {r['dataset']} | {r['protocol']} | {r['k']} | {r['model']} | {r['seeds']} | {fmt('auprc')} | {fmt('auroc')} |")
    lines += ["", "## Convergence", "",
              "| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['protocol']} | {r['k']} | {r['model']} | {r['epoch0_val_auprc_mean']:.4f} | "
              f"{r['best_val_auprc_mean']:.4f} | {r['best_epoch_mean']:.1f} | {r['seeds_never_improved']}/{r['seeds']} |"
              for r in convergence]
    lines += ["", "## Paired variant changes against vanilla AllSet (test, all sizes)", "",
              "| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['protocol']} | {r['k']} | {r['variant']} | {r['seeds']} | "
              f"{_fmt(r['delta_auprc_mean'], r['delta_auprc_std'])} | {_fmt(r['delta_auroc_mean'], r['delta_auroc_std'])} | "
              f"{r['delta_auprc_seeds_positive']}/{r['seeds']} |" for r in delta_summary if r["size"] == "all"]
    lines += ["", "## Interpretability against performance (test)", "",
              "Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change "
              "at factor 8. Neutralized: AP change when the decoder size is fixed.", "",
              "| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['protocol']} | {r['k']} | {r['model']} | {_fmt(r['test_auprc_mean'])} | "
              f"{_fmt(r['probe_test_mae_mean'])} ({_fmt(r['constant_baseline_test_mae_mean'])}) | "
              f"{_fmt(r['replication_modified_factor8'])} | {_fmt(r['replication_decoder_factor8'])} | "
              f"{_fmt(r['logit_change_mean'])} | {_fmt(r['neutralized_auprc_change_mean'])} |" for r in synthesis]
    lines += ["", "AP depends on the configured negative prevalence. Compare only matched protocols/candidates.",
              f"Per-size bins with fewer than {MIN_BIN_POSITIVES} positives in any seed are flagged as sparse; "
              "seed SD is not a confidence interval.",
              "Interpretability diagnostics and computational interventions are in each model's interpretability.json.",
              "See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, "
              "and convergence.csv for training curves."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "report_manifest.json", {"completed_runs": len(results), "summary_rows": len(summary)})
    return root / "REPORT.md"
