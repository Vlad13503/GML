# CURRENT â€” verified shared-procedure CPA comparison

Procedure: traian `2e33dd39c3bd08561d9533d315472af38eb9272d`. Former campaigns are superseded; none of their numerical results are inputs here.

Shared architectural settings are selected by the branch's five-model validation tuning. Each final model independently selects learning rate from 0.001/0.01 and a checkpoint including epoch 0. Order k=3; two datasets; seeds 0,1,2. All variants share exact contexts and candidates.

| Dataset | Width | Layers | Dropout |
|---|---:|---:|---:|
| email-Enron | 32 | 2 | 0.0 |
| tags-ask-ubuntu | 64 | 2 | 0.0 |

| Dataset | Metric | Vanilla AST | Encoder CPA | Readout CPA | Both |
|---|---|---:|---:|---:|---:|
| email-Enron | AUPRC | 0.7598 +/- 0.0530 | 0.7478 +/- 0.0073 | 0.7554 +/- 0.0215 | 0.7569 +/- 0.0154 |
| email-Enron | AUROC | 0.7453 +/- 0.0624 | 0.7561 +/- 0.0188 | 0.7664 +/- 0.0182 | 0.7602 +/- 0.0310 |
| tags-ask-ubuntu | AUPRC | 0.8742 +/- 0.0033 | 0.8838 +/- 0.0009 | 0.8722 +/- 0.0064 | 0.8823 +/- 0.0014 |
| tags-ask-ubuntu | AUROC | 0.8912 +/- 0.0036 | 0.8962 +/- 0.0011 | 0.8911 +/- 0.0028 | 0.8945 +/- 0.0013 |

| Dataset | Metric | Encoder - vanilla | Readout - vanilla | Both - vanilla |
|---|---|---:|---:|---:|
| email-Enron | AUPRC | -0.0121 +/- 0.0461 | -0.0044 +/- 0.0315 | -0.0029 +/- 0.0501 |
| email-Enron | AUROC | +0.0108 +/- 0.0537 | +0.0211 +/- 0.0594 | +0.0149 +/- 0.0392 |
| tags-ask-ubuntu | AUPRC | +0.0096 +/- 0.0036 | -0.0019 +/- 0.0073 | +0.0081 +/- 0.0024 |
| tags-ask-ubuntu | AUROC | +0.0049 +/- 0.0034 | -0.0001 +/- 0.0044 | +0.0033 +/- 0.0022 |

Values are means and sample SDs across three split/initialization seeds, not confidence intervals. Differences are paired within seeds. Negatives are 1:1 size-matched; all nodes in scored targets occur in the context. Small per-size bins are flagged by the pinned reporter.

Epoch-0 selections (out of six dataset/seed fits per model): {'Vanilla AST': 0, 'Encoder CPA': 0, 'Readout CPA': 0, 'Both': 0}.

Verification replayed all 80 selected tuning checkpoints and all 24 comparison checkpoints. Saved comparison validation/test logits are bitwise reproducible; source/data hashes, split identities, tuning selection, shared initial weights and parameter counts were checked. GCN tuning validation replay allows absolute AP error up to 1e-6 because CUDA sparse aggregation showed sub-micro variation; exact errors are recorded in verification.json. AST/CPA replay requires exact equality.

Former multiplier and downstream signal checks remain superseded. Fresh local normalization controls and validation-only downstream interventions have now completed on 12 current encoder-only/combined checkpoints: see [encoder diagnostics](encoder-diagnostics/SUMMARY.md). Exact-cardinality test performance analysis is in [cardinality analysis](cardinality-analysis/SUMMARY.md).
