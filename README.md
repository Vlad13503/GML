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
- `core.json`: five models, three protocols, and three seeds on both datasets.
- `encoder_ablation.json`: isolate encoder and candidate-readout CPA.

CPA scales candidate attention aggregates before normalization by default. Context and targets are disjoint. Models share size-matched candidates and use validation-only checkpoint selection.

Outputs are saved under `results/<suite-id>/` and completed runs resume automatically.

```powershell
python -m hnn_project report results/<suite-id>
```
