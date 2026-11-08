# Hyperedge cardinality and interpretability

> **Experiment status: ALL former local results are SUPERSEDED.** Read [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) and `CURRENT_EXPERIMENT.json`. Only the verified current campaign supplies current findings. Never quote old campaign scores as current findings.

## Current CPA contribution

The [current comparison](results/team-procedure-2e33dd3-20261107/SUMMARY.md), [cardinality analysis](results/team-procedure-2e33dd3-20261107/cardinality-analysis/SUMMARY.md), and [encoder diagnostics](results/team-procedure-2e33dd3-20261107/encoder-diagnostics/SUMMARY.md) accompany the report draft. The current campaign includes checkpoints, predictions, saved splits, source/data hashes, tuning histories, figures and verification records. Superseded result directories, their archived report draft, and obsolete experiment plans were removed at the user's request on 2026-11-08. Their paths remain denied by the current-results guard.

**The current fits use the exact archived `traian` source at commit `2e33dd3`, exported to an isolated copy. On `traian`, that commit was rewritten as `725bd5e` with an identical source tree.** Restore that isolated source before running current analyses:

```powershell
python scripts/restore_team_source.py
python scripts/current_results.py
python scripts/analyze_current_cardinality.py
python scripts/diagnose_current_encoder.py
python scripts/summarize_current_encoder.py
```

Install the dependencies in `pyproject.toml`; CUDA diagnostics require GPU PyTorch (the recorded environment uses PyTorch 2.11.0+cu128). `source-2e33dd3.zip` and `run_plan.json` make the source export portable to another checkout. Analysis 1 uses saved predictions and does not require CUDA. The frozen encoder diagnostics require CUDA and the restored export, and evaluate validation only. All current result readers reject superseded or unverified campaigns.

`scripts/run_team_procedure.py` is the original full tuning/training controller, not the command for viewing completed results; do not rerun it over the completed campaign. The original verification script records its pre-publication audit and expects the completed-but-unverified state. Check `verification.json` for that audit and the diagnostic outputs for subsequent checks. Downloaded raw dataset files remain excluded; their exact hashes are in `run_plan.json`.

## Project report

Write the team report in [`report/main.tex`](report/main.tex), starting from the supplied Overleaf template. Set `main.tex` as the main document when uploading the `report/` folder to Overleaf. Keep `Project_template.tex` and `Project_template.pdf` as the original references; both `.sty` files are required for compilation.

## Historical scaffold

Earlier runner scripts and their tests remain available as implementation history. Their former result directories and experiment plans have been removed. Use the current campaign and reproduction guidance above for current findings.

## Additional suites from `traian` (not the current campaign)

The root `hnn_project` package now contains the `traian` source from commit `2e33dd3` (merged as `725bd5e`, which has an identical tree), plus the historical `hnn_project/tune.py`. The suites below were run with it before the current campaign, under their own settings. Their outputs are in `results/<suite-id>/` (each suite's `config.json` names its settings). They are separate from `results/team-procedure-2e33dd3-20261107/`; do not mix their numbers with the current campaign. Checkpoints, per-run metrics, training curves, manifests and suite reports are committed; raw interpretability dumps, predictions and splits are not.

### Run

Python 3.10+:

```powershell
python -m pip install -e .
python -m hnn_project download
python -m hnn_project audit
python -m unittest discover -s tests -v
python -m hnn_project run --config configs/core.json
```

Configurations:

- `smoke.json`: synthetic checks for all variants.
- `pilot.json`: short runs on both datasets.
- `tune.json`: validation-only grid for choosing shared settings per dataset.
- `core.json`: five models, three protocols, and three seeds on both datasets.
- `encoder_ablation.json`: CPA placement: vanilla AllSet versus readout-only, encoder-only, and combined.
- `order_k4.json`: order shift at k=4.
- `negatives_easy.json`, `negatives_hard.json`: vanilla AllSet versus CPA with only random or only node-swap negatives. `core.json` provides the 50/50 mix.
- `long_*.json`: AskUbuntu order shift with a 600-epoch limit, to check the effect of the 150-epoch cap. Each model is its own suite so they can run in parallel; `analysis/long_training.py` compares them with the 150-epoch runs.
- `stress_*.json`, `confirm_*.json`: stress test. Arms never see large hyperedges, see them as training targets, or see them in targets and context, scored on the same test set. Design is in `analysis/STRESS_TEST_PLAN.md`; `analysis/stress_test.py` and `analysis/stress_confirm.py` apply its rules. The confirm suites use Enron seeds 3-12 and need overrides for Enron only.

CPA scales candidate attention aggregates before normalization by default. Context and targets are disjoint. Models share size-matched candidates and use validation-only checkpoint selection, including the untrained epoch 0.

### Full workflow

```powershell
python -m hnn_project tune --config configs/tune.json
python -m hnn_project run --config configs/core.json --overrides results/tuning-<id>/best_settings.json
python -m hnn_project run --config configs/encoder_ablation.json --overrides results/tuning-<id>/best_settings.json
python -m hnn_project run --config configs/order_k4.json --overrides results/tuning-<id>/best_settings.json
python -m hnn_project run --config configs/negatives_easy.json --overrides results/tuning-<id>/best_settings.json
python -m hnn_project run --config configs/negatives_hard.json --overrides results/tuning-<id>/best_settings.json
python -m hnn_project compare results/<easy-id> results/<core-id> results/<hard-id> --output results/negative_sensitivity
```

Outputs are saved under `results/<suite-id>/` and completed runs resume automatically.

```powershell
python -m hnn_project report results/<suite-id>
```

`report` accepts exactly one suite and refuses results from another one. Use `compare` for deliberate comparisons across suites: it keeps only results present in every suite and labels them by the settings that differ.

Each suite contains `REPORT.md`, `summary.csv`, `convergence.csv` (epoch 0 onward), `paired_cpa_summary.csv` (variants against vanilla AllSet), `interpretability_summary.csv` (cardinality probes, replication, and size interventions joined with performance), `size_response.csv`, and `figures/`. Per-size bins with fewer than 20 positives are flagged as sparse.
