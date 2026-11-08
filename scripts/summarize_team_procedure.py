"""Publish only the verified current team-procedure comparison; no legacy inputs."""
import csv
import json
from pathlib import Path

import numpy as np
from current_results import current_campaign

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("allset", "Vanilla AST"), ("allset_cpa_encoder", "Encoder CPA"),
          ("allset_cpa", "Readout CPA"), ("allset_cpa_both", "Both")]


def main():
    campaign = current_campaign()
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    suite = ROOT / registry["comparison_suite"]
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
    lines = ["# CURRENT â€” verified shared-procedure CPA comparison", "",
             f"Procedure: traian `{registry['procedure_commit']}`. Former campaigns are superseded; none of their numerical results are inputs here.", "",
             "Shared architectural settings are selected by the branch's five-model validation tuning. Each final model independently selects learning rate from 0.001/0.01 and a checkpoint including epoch 0. Order k=3; two datasets; seeds 0,1,2. All variants share exact contexts and candidates.", "",
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
    zeros = {label: sum(r["training"]["best_epoch"] == 0 for r in results if r["model"] == model) for model, label in MODELS}
    lines += ["", "Values are means and sample SDs across three split/initialization seeds, not confidence intervals. Differences are paired within seeds. Negatives are 1:1 size-matched; all nodes in scored targets occur in the context. Small per-size bins are flagged by the pinned reporter.", "",
              f"Epoch-0 selections (out of six dataset/seed fits per model): {zeros}.", "",
              "Verification replayed all 80 selected tuning checkpoints and all 24 comparison checkpoints. Saved comparison validation/test logits are bitwise reproducible; source/data hashes, split identities, tuning selection, shared initial weights and parameter counts were checked. GCN tuning validation replay allows absolute AP error up to 1e-6 because CUDA sparse aggregation showed sub-micro variation; exact errors are recorded in verification.json. AST/CPA replay requires exact equality.", "",
              "Former multiplier and downstream signal checks are superseded and were not repeated on these new checkpoints. Their conclusions must not be attributed to this campaign."]
    (campaign / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (campaign / "summary_source.py").write_bytes(Path(__file__).read_bytes())
    registry["summary"] = str((campaign / "SUMMARY.md").relative_to(ROOT))
    (ROOT / "CURRENT_EXPERIMENT.json").write_text(json.dumps(registry, indent=2) + "\n")
    status = ROOT / "EXPERIMENT_STATUS.md"
    s = status.read_text().replace('Status: preparing fresh tuning and fits; no current numerical conclusions yet.',
                                 'Status: VERIFIED CURRENT. Fresh tuning and all 24 comparison fits completed and replayed. See the current campaign SUMMARY.md and verification.json.')
    status.write_text(s, encoding="utf-8", newline="\n")
    report = ROOT / "report/main.tex"
    s = report.read_text()
    s = s.replace('Fresh team-procedure campaign: fits running; no current findings yet.',
                  'Current shared-procedure campaign: completed and verified (2e33dd3).')
    abstract_start, remainder = s.split('\\begin{abstract}', 1)
    _, after_abstract = remainder.split('\\end{abstract}', 1)
    abstract = '''
We investigate cardinality scaling in hyperedge prediction under interaction-order shift using a verified shared team procedure. Vanilla AllSetTransformer-based prediction is compared with CPA in encoder node-to-hyperedge attention, candidate readout, or both. Architectural settings are selected by mean validation AUPRC across five model families; each final model independently selects its learning rate and checkpoint, including epoch 0. Contexts, target candidates and negatives are matched within each seed. The fresh campaign covers Enron and AskUbuntu, threshold three, and three paired seeds. Results are reported by metric and placement below. Earlier local campaigns are superseded and contribute no numerical evidence to this comparison. The former encoder multiplier and downstream interventions have not yet been repeated on these checkpoints.
'''
    s = abstract_start + '\\begin{abstract}' + abstract + '\\end{abstract}' + after_abstract
    selected_lines = []
    for name, setting in config["dataset_overrides"].items():
        selected_lines.append(f"{name}: width {setting['hidden_dim']}, {setting['layers']} layer(s), dropout {setting['dropout']}")
    table = '''\\subsection{Verified current results}
All 24 comparison fits completed after shared validation tuning. Selected architectural settings are ''' + '; '.join(selected_lines) + '''. Learning rate and checkpoint epoch vary by model and seed under the shared selection rule.

\\begin{table}[t]
\\centering
\\small
\\caption{Fresh shared-procedure test results at $k=3$: mean $\\pm$ sample SD across three paired seeds. Former campaign results are excluded.}
\\label{tab:current}
\\begin{tabular}{llcccc}
\\toprule
Dataset & Metric & Vanilla AST & Encoder CPA & Readout CPA & Both \\\\
\\midrule
'''
    for r in summaries:
        label = "Enron" if r["dataset"] == "email-Enron" else "AskUbuntu"
        cells = [f"${r['models'][m]['mean']:.4f}\\pm{r['models'][m]['seed_sd']:.4f}$" for m, _ in MODELS]
        table += label + ' & ' + r['metric'].upper() + ' & ' + ' & '.join(cells) + ' \\\\\n'
    table += '''\\bottomrule
\\end{tabular}
\\end{table}

Differences against vanilla are paired within seeds; SDs are not confidence intervals. Exact per-seed metrics, selected learning rates, epochs, and per-cardinality results accompany the current campaign. Verification replayed 80 selected tuning checkpoints and all 24 comparison checkpoints, reproducing comparison validation/test logits bit-for-bit. GCN tuning replay permits absolute AP error up to $10^{-6}$ owing to observed CUDA sparse-aggregation variation; AST/CPA replay requires exact equality. Source and data identities, six shared splits, tuning selection, initial shared weights, and parameter counts were checked.

'''
    start = s.index('\\subsection{Current results pending}')
    end = s.index('\\section{Discussion}', start)
    s = s[:start] + table + s[end:]
    old = 'Current conclusions are pending. The fresh comparison will assess predictive performance by metric, placement, and test cardinality without selecting settings using test results. Three seeds, one threshold, sparse high-cardinality bins, and shared rather than independently optimized architecture settings delimit the interpretation. Normalization remains an architectural concern, not a mechanism established for the fresh checkpoints.'
    new = 'The table describes a fixed shared-architecture comparison with per-model learning-rate and checkpoint selection, not independently optimized architectures. Three seeds, one threshold, and sparse high-cardinality bins delimit the conclusions; there is no claim of statistical superiority. Metrics must be interpreted separately. Former multiplier and downstream-intervention outcomes are superseded: normalization remains an architectural concern, not a mechanism demonstrated for the fresh checkpoints. Fresh diagnostics and any subsequent architecture selection must not use inspected test scores for tuning.'
    assert old in s
    s = s.replace(old, new)
    report.write_text(s, encoding="utf-8", newline="\n")
    print((campaign / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
