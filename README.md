# Hyperedge cardinality and interpretability

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
- `tune.json`: validation-only grid for choosing shared settings per dataset.
- `core.json`: five models, three protocols, and three seeds on both datasets.
- `encoder_ablation.json`: CPA placement: vanilla AllSet versus readout-only, encoder-only, and combined.
- `order_k4.json`: order shift at k=4.
- `negatives_easy.json`, `negatives_hard.json`: vanilla AllSet versus CPA with only random or only node-swap negatives. `core.json` provides the 50/50 mix.
- `long_*.json`: AskUbuntu order shift with a 600-epoch limit, to check the effect of the 150-epoch cap. Each model is its own suite so they can run in parallel; `analysis/long_training.py` compares them with the 150-epoch runs.
- `stress_*.json`, `confirm_*.json`: stress test. Arms never see large hyperedges, see them as training targets, or see them in targets and context, scored on the same test set. Design is in `analysis/STRESS_TEST_PLAN.md`; `analysis/stress_test.py` and `analysis/stress_confirm.py` apply its rules. The confirm suites use Enron seeds 3-12 and need overrides for Enron only.

CPA scales candidate attention aggregates before normalization by default. Context and targets are disjoint. Models share size-matched candidates and use validation-only checkpoint selection, including the untrained epoch 0.

## Full workflow

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
