"""Summarize the verified CPA runs without editing the report or registry."""
import csv
import json
from pathlib import Path

import numpy as np
from current_results import current_campaign, require_current_path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("allset", "Vanilla AST"), ("allset_cpa_encoder", "Encoder CPA"),
          ("allset_cpa", "Readout CPA"), ("allset_cpa_both", "Both")]


def main():
    campaign = current_campaign()
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    suite = require_current_path(ROOT / registry["comparison_suite"])
    verification = json.loads((ROOT / registry["verification"]).read_text())
    config = json.loads((suite / "config.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted(suite.rglob("result.json"))]
    assert len(results) == 24 and all(r["suite"] == suite.name for r in results)
    summaries, rows = [], []
    for dataset in config["datasets"]:
        for metric in ("auprc", "auroc"):
            summary = {"dataset": dataset, "metric": metric, "models": {}, "paired_vs_vanilla": {}}
            baseline = {r["seed"]: r["evaluation"]["test"]["overall"][metric]
                        for r in results if r["dataset"] == dataset and r["model"] == "allset"}
            for model, label in MODELS:
                selected = sorted([r for r in results if r["dataset"] == dataset and r["model"] == model], key=lambda r: r["seed"])
                assert [r["seed"] for r in selected] == [0, 1, 2]
                values = [r["evaluation"]["test"]["overall"][metric] for r in selected]
                delta = [value - baseline[r["seed"]] for r, value in zip(selected, values)]
                summary["models"][model] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)), "per_seed": values}
                summary["paired_vs_vanilla"][model] = {"mean": float(np.mean(delta)), "seed_sd": float(np.std(delta, ddof=1)), "per_seed": delta}
                for r, value, change in zip(selected, values, delta):
                    rows.append({"dataset": dataset, "model": model, "seed": r["seed"], "metric": metric,
                        "test_value": value, "paired_delta_vs_vanilla": change,
                        "selected_lr": r["training"]["selected_learning_rate"], "best_epoch": r["training"]["best_epoch"],
                        "never_improved": r["training"]["never_improved"], "split_sha256": r["split_sha256"]})
            summaries.append(summary)
    artifact = {"status": "VERIFIED_CURRENT", "procedure_commit": registry["procedure_commit"],
                "suite": str(suite.relative_to(ROOT)), "shared_settings": config["dataset_overrides"],
                "test_used_for_tuning": False, "legacy_results_used": False, "records": summaries}
    (campaign / "comparison.json").write_text(json.dumps(artifact, indent=2) + "\n")
    with (campaign / "paired_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# CPA placement results", "",
             "Order shift at k=3, seeds 0–2. Models share context and candidates, with one size-matched negative per positive. Architecture is selected on validation per dataset; each model selects its learning rate (0.001 or 0.01) and checkpoint, including epoch zero.", "",
             "| Dataset | Width | Layers | Dropout |", "|---|---:|---:|---:|"]
    for name, setting in config["dataset_overrides"].items():
        lines.append(f"| {name} | {setting['hidden_dim']} | {setting['layers']} | {setting['dropout']} |")
    lines += ["", "| Dataset | Metric | Vanilla AST | Encoder CPA | Readout CPA | Both |", "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        cells = [f"{r['models'][m]['mean']:.4f} +/- {r['models'][m]['seed_sd']:.4f}" for m, _ in MODELS]
        lines.append(f"| {r['dataset']} | {r['metric'].upper()} | " + " | ".join(cells) + " |")
    lines += ["", "| Dataset | Metric | Encoder - vanilla | Readout - vanilla | Both - vanilla |", "|---|---|---:|---:|---:|"]
    for r in summaries:
        cells = [f"{r['paired_vs_vanilla'][m]['mean']:+.4f} +/- {r['paired_vs_vanilla'][m]['seed_sd']:.4f}" for m, _ in MODELS[1:]]
        lines.append(f"| {r['dataset']} | {r['metric'].upper()} | " + " | ".join(cells) + " |")
    zeros = {label: sum(r["training"]["best_epoch"] == 0 for r in results if r["model"] == model)
             for model, label in MODELS}
    epoch_zero = ", ".join(f"{label}: {n}" for label, n in zeros.items())
    lines += ["", "Values are means and sample SDs. Differences are paired within seeds.", "",
              f"Epoch-zero selections: {epoch_zero}.", "",
              f"[Verification](verification.json) replayed {verification['tuning_selected_checkpoints_replayed']} tuning and "
              f"{verification['comparison_checkpoints_replayed']} comparison checkpoints. Comparison logits were bitwise identical. "
              "The audit also checks the [pinned source and data hashes](run_plan.json), splits, tuning, initialization, and parameter counts. "
              "GCN tuning AP replay allows an absolute error of 1e-6; AST/CPA replay requires an exact match.", ""]
    for label, relative in (("Per-size test metrics", "cardinality-analysis/SUMMARY.md"),
                            ("Frozen encoder diagnostics", "encoder-diagnostics/SUMMARY.md")):
        if (campaign / relative).is_file():
            lines.append(f"- [{label}]({relative})")
    summary_path = campaign / "SUMMARY.md"
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
