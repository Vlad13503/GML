# Hyperedge prediction across interaction orders

We train on small hyperedges and test on larger ones in Enron emails and AskUbuntu tags. DeepSets and a clique-expansion GCN serve as baselines for AllSetTransformer (AST). We compare cardinality-preserving attention (CPA) in the encoder, the readout, or both, and test how the scaling interacts with LayerNorm.

## Repository layout

- `hnn_project/` — data splits, negative sampling, models, training, and metrics.
- `configs/` — settings for the baseline, order-shift, and ablation runs.
- `analysis/` and `scripts/` — stress tests, CPA diagnostics, and result summaries.
- `results/` — saved runs and checkpoints. [CURRENT_EXPERIMENT.json](CURRENT_EXPERIMENT.json) identifies the verified CPA run.
- `report/` and `tests/` — LaTeX source and pipeline tests, respectively.

## Reproduce the results

From the repository root, install the package with Python 3.10+:

```powershell
python -m pip install -e .
```

The [CPA comparison](results/team-procedure-2e33dd3-20261107/SUMMARY.md) is included. Regenerate the per-size analysis from the saved predictions:

```powershell
python scripts/analyze_current_cardinality.py
```

The frozen encoder diagnostics use the archived source and PyTorch `2.11.0+cu128` on CUDA:

```powershell
python scripts/restore_team_source.py
python scripts/diagnose_current_encoder.py
python scripts/summarize_current_encoder.py
```

To retrain, download the data and use the saved run configurations. Use each suite's `environment.json` for the recorded Python and PyTorch versions. For the stress tests and negative-sampling runs, substitute the corresponding suite's `config.json` from `results/` in the first run command.

```powershell
python -m hnn_project download
python -m hnn_project run --config results/5e68c177dfb8/config.json --device cpu --output tmp/reproduction

python scripts/restore_team_source.py
Push-Location tmp/team-procedure-2e33dd3
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -m hnn_project run --config ../../results/team-procedure-2e33dd3-20261107/fits/db95c77a328c/config.json --data-dir ../../data --device cuda --output ../reproduction
Pop-Location
```

The CPA command reuses the selected settings. To repeat tuning, run `python -m hnn_project tune --config configs/tune.json --data-dir ../../data --device cuda --output ../reproduction/tuning` from the archived source directory, then pass the new `best_settings.json` with `--overrides`. The [run plan](results/team-procedure-2e33dd3-20261107/run_plan.json) records the source and data hashes.
