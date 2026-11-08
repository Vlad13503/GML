> **ANTIQUATED / SUPERSEDED.** This document describes the former experiment campaign. Its results, diagnostics, and checkpoints are historical, not current project findings. See `../EXPERIMENT_STATUS.md` and `../CURRENT_EXPERIMENT.json`.

# Next experiment: readout-only scaled CPA

Status: all six matched runs have been trained and evaluated on CUDA. Saved checkpoints replay validation/test logits bit-for-bit. Results are in `results/readout-cpa/7d0356f3d97c/`; the three-way comparison is in `results/cpa-placement/7d0356f3d97c/`. The design below records the frozen plan used for this follow-up.

## Modification

Use the existing `allset_cpa` model, which scales candidate-readout attention by the candidate's actual cardinality before its query residual and LayerNorm. Both encoder aggregation directions remain vanilla. This is readout-only CPA, not encoder-plus-readout CPA and not scaling after normalization.

The encoder continues to learn from the same small context hyperedges. Unlike encoder-only CPA, the modified pooling operation directly processes the larger held-out candidate groups.

## Question and hypothesis

Does preserving cardinality at the candidate scoring stage improve larger-group prediction relative to vanilla AST and encoder-only CPA? We hypothesize that this placement can help because it directly receives the candidate size. LayerNorm can still weaken the signal, and the hypothesis does not imply uniformly better performance or improved extrapolation at every order gap.

## Fixed comparison

- Reuse frozen AST suite `7d0356f3d97c`: the exact contexts, train/validation/test candidates and labels, node filtering, negative samples, and seeds 0/1/2.
- Enron: width 32; AskUbuntu: width 64. Both: one encoder layer, four heads, learning rate 0.003, dropout 0.2, AdamW weight decay 0.0001, clipping norm 5, candidate batches of 2,048 with accumulated gradients, maximum 150 epochs, patience 30, validation every five epochs.
- Train end to end from initialization. Shared initial weights, initial-state hashes, and parameter counts match vanilla and encoder CPA. Do not initialize from an encoder-CPA checkpoint.
- Select each run's checkpoint independently by validation AUPRC. No separate readout-CPA tuning, multiplier search, or configuration selection using test scores.
- Score the frozen test candidates after validation selection. Report AUPRC/AUROC, paired differences, seed variation, and per-cardinality results with small-bin flags. Compare all three placements rather than declaring a winner based on one metric.

This is a follow-up placement experiment, not an independently optimized model comparison. The test sets have already been inspected for earlier variants; further architecture search should remain validation-driven and be described as exploratory where appropriate.

## Execution

Prepared identities and plan: `results/readout-cpa/7d0356f3d97c/`. The historical preparation checked all six inputs and matched initial weights before training. `prepared_inputs.json` records that earlier preparation state; `progress.json`, checkpoints, and results record the completed six fits.

```powershell
# Recheck inputs without training
.\.venv\Scripts\python.exe scripts/train_readout_cpa.py --suite results/ast-tuning/7d0356f3d97c --prepare-only

# Start the six matched training runs when ready
.\.venv\Scripts\python.exe scripts/train_readout_cpa.py --suite results/ast-tuning/7d0356f3d97c
```

Outputs will include per-seed checkpoints, training histories, validation/test predictions, and paired comparisons to vanilla. Encoder-only results remain preserved in their separate output directory.

## Observed outcome

Readout-only mean test AUPRC is 0.7250 on Enron and 0.8731 on AskUbuntu, versus 0.7120/0.8744 for vanilla and 0.7322/0.8787 for encoder-only CPA. The readout placement does not provide a consistent improvement over vanilla under these fixed settings. See the paired differences, sample SDs, and per-cardinality curves before drawing broader conclusions. No setting was changed in response to test performance.
