# CPA placement results

Order shift at k=3, seeds 0–2. Models share context and candidates, with one size-matched negative per positive. Architecture is selected on validation per dataset; each model selects its learning rate (0.001 or 0.01) and checkpoint, including epoch zero.

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

Values are means and sample SDs. Differences are paired within seeds.

Epoch-zero selections: Vanilla AST: 0, Encoder CPA: 0, Readout CPA: 0, Both: 0.

[Verification](verification.json) replayed 80 tuning and 24 comparison checkpoints. Comparison logits were bitwise identical. The audit also checks the [pinned source and data hashes](run_plan.json), splits, tuning, initialization, and parameter counts. GCN tuning AP replay allows an absolute error of 1e-6; AST/CPA replay requires an exact match.

- [Per-size test metrics](cardinality-analysis/SUMMARY.md)
- [Frozen encoder diagnostics](encoder-diagnostics/SUMMARY.md)
