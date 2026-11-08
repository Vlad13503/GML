# Current experiment status

**All results produced before the team-procedure campaign on 2026-11-07 are ANTIQUATED / SUPERSEDED.**

The user requested a fresh comparison using the procedure on `traian`, pinned to commit `2e33dd39c3bd08561d9533d315472af38eb9272d`.

## Current campaign

- Root: `results/team-procedure-2e33dd3-20261107/`
- Status: VERIFIED CURRENT. Fresh tuning and all 24 comparison fits completed and replayed. See the current campaign SUMMARY.md and verification.json.
- Shared tuning: the branch's `configs/tune.json`, averaging validation AP over its five models to select width/layers/dropout per dataset.
- Learning-rate candidates: 0.001 and 0.01, selected independently within each model fit.
- Checkpoint selection: validation AUPRC including epoch 0.
- CPA comparison: vanilla AST, readout-only CPA, encoder-only CPA, and combined CPA; order shift at k=3; both datasets; seeds 0, 1, 2.
- Exact exported source, data hashes, shared overrides, candidates, manifests, and checkpoints must accompany the fresh results.

- Exact-cardinality test analysis: complete and verified; see `results/team-procedure-2e33dd3-20261107/cardinality-analysis/SUMMARY.md`. This is descriptive performance analysis; multiplier/normalization interventions are documented separately below.

- Fresh encoder diagnostics: VERIFIED CURRENT; 12 frozen encoder-only/combined checkpoints, local normalization controls and validation-only downstream interventions. See `results/team-procedure-2e33dd3-20261107/encoder-diagnostics/SUMMARY.md`. No retraining or intervention test scoring.

Machine-readable authority: `CURRENT_EXPERIMENT.json`.

## Superseded material

The former campaign `7d0356f3d97c` and all associated tuning, pilot, test, placement-comparison, multiplier, and downstream diagnostic outputs are historical. Earlier 2026-11-06 scores in chat or reports must not be reused as current evidence.

Legacy roots: `results/ast-tuning`, `results/ast-tuning-smoke`, `results/ast-test`, `results/encoder-cpa`, `results/readout-cpa`, `results/both-cpa`, `results/cpa-placement`, `results/cpa-placement-both`, `results/cardinality-diagnostic`, `results/encoder-signal-check`.

On 2026-11-08, at the user's request, the superseded result directories, archived report draft, and obsolete diagnostic/readout plans were removed. The superseded paths remain in CURRENT_EXPERIMENT.json only as a denylist. Earlier Git history is historical and must not be restored as current evidence. The verified current campaign and its artifacts are unchanged.
