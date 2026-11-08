# Experiment status: mandatory before using numerical results

Read `EXPERIMENT_STATUS.md` and `CURRENT_EXPERIMENT.json` before quoting, comparing, or reporting any experiment result.

On 2026-11-07 the user explicitly superseded ALL former local CPA/AST experiments in favor of the shared procedure pinned to traian commit `2e33dd39c3bd08561d9533d315472af38eb9272d`.

The old campaigns under `results/ast-tuning*`, `results/ast-test`, `results/encoder-cpa`, `results/readout-cpa`, `results/both-cpa`, `results/cpa-placement*`, `results/cardinality-diagnostic`, and `results/encoder-signal-check` are HISTORICAL / SUPERSEDED. Do not use their scores, selected configurations, checkpoint choices, or diagnostic conclusions as current project findings. Preserve them for provenance. Their previous summaries and report drafts are historical even if earlier chat messages described them as final or report-ready.

Only the campaign named in `CURRENT_EXPERIMENT.json` may supply current results. Pending work is not a completed result. Do not merge values across procedures. Fresh diagnostics require the fresh checkpoints.
