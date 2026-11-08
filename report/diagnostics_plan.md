> **ANTIQUATED / SUPERSEDED.** This document describes the former experiment campaign. Its results, diagnostics, and checkpoints are historical, not current project findings. See `../EXPERIMENT_STATUS.md` and `../CURRENT_EXPERIMENT.json`.

# Follow-up: artificial cardinality multiplier

Status: executed after the initial CPA comparison. All 12 checkpoints completed the sweep, producing 107,520 per-edge/stage measurements. Both invariance controls passed exactly. Results and curves are saved in `results/cardinality-diagnostic/7d0356f3d97c/`. This document records the follow-up design; it is a mechanism diagnostic, not another prediction benchmark.

## Question

For the same attention aggregate and learned query, how much of the change introduced by cardinality scaling survives the pooling block's two LayerNorms? Does the response flatten as the multiplier increases?

## Frozen inputs

- Use both vanilla and encoder-CPA checkpoints for both datasets and seeds 0, 1, 2 from suite `7d0356f3d97c`. No retraining or checkpoint reselection.
- Sample up to 64 original context hyperedges per available cardinality (2 and 3). Use the same context edge IDs for each paired model, with sampling seed `7300 + run_seed`.
- Exclude the artificial singleton hyperedges used for self-messages. Do not use prediction targets as encoder context.
- Capture each layer's attention aggregate once at the checkpoint's ordinary forward pass. Freeze keys, attention weights, values, aggregate, query, and block parameters for the local sweep. This is a conditional layer intervention, not a cascading change across layers.

## Intervention

Replace the multiplier with `m = 1, 2, 3, 4, 8, 16, 32`. Here `m` is an artificial absolute multiplier, not a replication factor or a modified member set. Compare against `m=1` within the same checkpoint, layer, edge, and control condition.

Record:

1. Raw attention aggregate `a` (fixed by construction).
2. Modified aggregate `m*a`.
3. Residual input `q + m*a`.
4. First LayerNorm output.
5. Full pooling output after the feed-forward residual and second LayerNorm.

Measure norm, L2 distance, relative L2 distance, and cosine similarity. Flag zero/near-zero vectors where cosine is undefined. Similar norms do not imply identical representations: direction can carry differences.

## Controls

- **Vanilla actual block:** the artificial multiplier is ignored, so all stages should be invariant. This uses the separately trained vanilla checkpoint.
- **CPA with zero residual query:** remove only the additive query residual. Do not recompute attention with a zero query. This holds the original raw aggregate fixed, isolating the subsequent normalization. LayerNorm should approximately remove positive uniform scaling; finite epsilon and floating-point effects must be distinguished from a meaningful residual response.
- **CPA with scaling disabled:** keep CPA-trained weights but ignore the multiplier. This checks invariance without conflating it with differences between trained vanilla and CPA weights.
- Reconstruct each sampled ordinary forward output from its captured raw aggregate and true cardinality before applying artificial values. Abort if this does not match the original block.

## Analysis to produce

Plot relative L2 and cosine responses against multiplier for pre-normalization, first-normalization, and final stages. Aggregate within each seed first; show the three seed responses rather than treating sampled hyperedges as independent experimental repeats. Keep original context sizes 2 and 3 distinguishable. Examine incremental changes at large multipliers to assess saturation.

Possible interpretations remain conditional: collapse after LayerNorm would support attenuation; retained directional changes would show block sensitivity. Neither finding alone identifies the reason for the observed task-score changes. A downstream prediction intervention would be separate and would require recomputing encoder outputs, rather than varying the decoder size input.

Do not tune architecture or choose an apparently favorable multiplier from these curves and report it on the already inspected test set as a fresh confirmatory result.

## Commands and artifacts

Run from the repository root using its CUDA environment:

```powershell
.\.venv\Scripts\python.exe scripts/cardinality_diagnostics.py --suite results/ast-tuning/7d0356f3d97c --dry-run
.\.venv\Scripts\python.exe scripts/cardinality_diagnostics.py --suite results/ast-tuning/7d0356f3d97c
```

The configuration is `configs/cardinality_diagnostic.json`. Output goes to `results/cardinality-diagnostic/7d0356f3d97c/`: checkpoint hashes, sampled IDs, the exact plan and diagnostic source, and (for the full sweep) `per_edge.csv`. Plots and interpretation should be added only after examining the completed sweep.

## Observed outcome

With the learned residual present, multiplier-dependent directional differences survive both LayerNorms. At m=3 relative to m=1, mean full-block relative L2 is 0.6439 on Enron and 0.9577 on AskUbuntu; with only the query residual removed it is approximately 1.44e-5 and 1.12e-5. The response relative to m=1 flattens at high multipliers. This demonstrates local block sensitivity, not why task performance changed or whether cardinality is globally decodable. See SUMMARY.md, summary.json, per_seed_summary.csv, and normalization_response.png in the results folder.
