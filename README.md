# Hyperedge cardinality and interpretability

> **Experiment status: ALL former local results are SUPERSEDED.** Read [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) and `CURRENT_EXPERIMENT.json`. Only the verified current campaign supplies current findings. Never quote old campaign scores as current findings.

## Current CPA contribution

The [current comparison](results/team-procedure-2e33dd3-20261107/SUMMARY.md), [cardinality analysis](results/team-procedure-2e33dd3-20261107/cardinality-analysis/SUMMARY.md), and [encoder diagnostics](results/team-procedure-2e33dd3-20261107/encoder-diagnostics/SUMMARY.md) accompany the report draft. The current campaign includes checkpoints, predictions, saved splits, source/data hashes, tuning histories, figures and verification records. Earlier raw runs remain local and historical; their warning markers and archived report are preserved in Git.

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

The commands below describe the retained historical scaffold and earlier procedure; they do not reproduce the current campaign.


Compare DeepSets, GCN, AllSetTransformer, and cardinality-aware variants on Enron and AskUbuntu.

## Project report

Write the team report in [`report/main.tex`](report/main.tex), starting from the supplied Overleaf template. Set `main.tex` as the main document when uploading the `report/` folder to Overleaf. Keep `Project_template.tex` and `Project_template.pdf` as the original references; both `.sty` files are required for compilation.

## Run

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
- `core.json`: five models, three protocols, and three seeds on both datasets.
- `encoder_ablation.json`: isolate encoder and candidate-readout CPA.

Validation-only vanilla AST tuning (CUDA-enabled PyTorch required for `--device cuda`):

```powershell
.\.venv\Scripts\python.exe -m hnn_project.tune --stage audit --device cuda
.\.venv\Scripts\python.exe -m hnn_project.tune --stage pilot --device cuda
.\.venv\Scripts\python.exe -m hnn_project.tune --stage tune --device cuda
```

`configs/ast_tuning.json` fixes the order-shift protocol at `k=3`. The search screens 12 configurations (learning rates 0.0003/0.001/0.003, widths 32/64, layers 1/2) on seed 0, then confirms the top two on seeds 1 and 2. Selection uses mean validation AUPRC across all three seeds, with trial ID breaking ties. Each fit allows 150 epochs with patience 30. Audits and a separate five-epoch pilot precede tuning. The runner saves exact candidates, manifests, code snapshots, checkpoints and validation predictions under `results/ast-tuning/<suite-id>/`; it never scores test candidates. Completed fits resume with identity checks. `selected_config.json` freezes the baseline settings for subsequent matched CPA training; it is not a test result.

After freezing and verifying a tuning suite, `scripts/evaluate_selected_ast.py --suite results/ast-tuning/<suite-id>` evaluates the selected vanilla checkpoints. Then run the matched encoder-only CPA comparison:

```powershell
.\.venv\Scripts\python.exe scripts/train_encoder_cpa.py --suite results/ast-tuning/<suite-id>
```

The CPA runner loads the baseline's saved splits and checkpoint configurations, checks matched initialization and parameter counts, and trains all six dataset/seed runs from initialization on CUDA. Only encoder node-to-hyperedge attention is scaled by cardinality before residual/LayerNorm. Checkpoints are selected by validation AUPRC, then scored on the same test candidates as vanilla. Results, predictions, and paired per-cardinality comparisons are saved under `results/encoder-cpa/<suite-id>/`. No separate CPA tuning or test-based model selection is performed.

Follow-up diagnostics and readout-only preparation:

```powershell
.\.venv\Scripts\python.exe scripts/encoder_signal_check.py --suite results/ast-tuning/<suite-id>
.\.venv\Scripts\python.exe scripts/train_readout_cpa.py --suite results/ast-tuning/<suite-id> --prepare-only
```

The signal check uses validation candidates only and recomputes the frozen CPA encoder with normal scaling, scaling disabled, and constant scaling of 3 on real context edges. Synthetic singleton self-messages retain multiplier 1. See [`report/readout_cpa_plan.md`](report/readout_cpa_plan.md) for the next placement experiment; omit `--prepare-only` only when starting its six training runs.

CPA scales candidate attention aggregates before normalization by default. Context and targets are disjoint. Models share size-matched candidates and use validation-only checkpoint selection.

Outputs are saved under `results/<suite-id>/` and completed runs resume automatically.

```powershell
python -m hnn_project report results/<suite-id>
```

Joint encoder/readout CPA uses the same frozen setup and trains both modifications from initialization:

```powershell
.\.venv\Scripts\python.exe scripts/train_both_cpa.py --suite results/ast-tuning/<suite-id>
.\.venv\Scripts\python.exe scripts/compare_all_cpa_placements.py --suite-id <suite-id>
```

Joint checkpoints are saved separately under `results/both-cpa/<suite-id>/`; the four-way comparison, paired differences, and order-gap plots are under `results/cpa-placement-both/<suite-id>/`.
