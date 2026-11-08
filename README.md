# Hyperedge cardinality and interpretability

> **Experiment status: ALL former local results are SUPERSEDED.** Read [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) and `CURRENT_EXPERIMENT.json`. Only the verified current campaign supplies current findings. Never quote old campaign scores as current findings.

## Current CPA contribution

The [current comparison](results/team-procedure-2e33dd3-20261107/SUMMARY.md), [cardinality analysis](results/team-procedure-2e33dd3-20261107/cardinality-analysis/SUMMARY.md), and [encoder diagnostics](results/team-procedure-2e33dd3-20261107/encoder-diagnostics/SUMMARY.md) accompany the report draft. The current campaign includes checkpoints, predictions, saved splits, source/data hashes, tuning histories, figures and verification records. Superseded result directories, their archived report draft, and obsolete experiment plans were removed at the user's request on 2026-11-08. Their paths remain denied by the current-results guard.

**The current fits use the exact archived `traian` source at commit `2e33dd3`, not the historical package at the root of this branch.** Restore that isolated source before running current analyses:

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
