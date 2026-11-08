"""Descriptive, paired cardinality analysis of verified current test predictions.

No training, checkpoint selection, candidate resampling, or legacy inputs.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from current_results import ROOT, current_campaign, require_current_path

MODELS = {"allset": "Vanilla AST", "allset_cpa_encoder": "Encoder CPA",
          "allset_cpa": "Readout CPA", "allset_cpa_both": "Both"}
COLORS = ["#394b59", "#167c80", "#c37627", "#8755a0"]
MIN_POSITIVES = 20


def main():
    campaign = current_campaign()
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    suite = require_current_path(ROOT / registry["comparison_suite"])
    # This report's interpretation is specific to the pinned, inspected campaign.
    # Refuse to carry its prose forward if a future campaign becomes current.
    assert suite.name == "db95c77a328c" and registry["procedure_commit"] == "2e33dd39c3bd08561d9533d315472af38eb9272d"
    config = json.loads((suite / "config.json").read_text())
    out = campaign / "cardinality-analysis"
    out.mkdir(exist_ok=True)
    # Written before metrics: exact sizes, no score-dependent pooling or omission.
    plan = {"suite": str(suite.relative_to(ROOT)), "procedure_commit": registry["procedure_commit"],
            "analysis": "descriptive held-out test analysis; no model selection",
            "grouping": "every exact cardinality; no merged bins",
            "sparse_rule": "flag if any seed has fewer than 20 positives",
            "pairing": "identical ordered candidates within dataset/seed",
            "uncertainty": "sample SD across 3 paired seeds, not confidence intervals",
            "seeds": config["seeds"], "legacy_inputs": False}
    (out / "analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    hashes, rows = {}, []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            folder = suite / dataset / "order-k3" / f"seed-{seed}"
            manifest = json.loads((folder / "manifest.json").read_text())
            with gzip.open(folder / "splits.json.gz", "rt") as stream:
                split = json.load(stream)
            labels = np.asarray(split["test"]["labels"], dtype=np.float32)
            sizes = np.asarray([len(e) for e in split["test"]["edges"]])
            assert np.all(sizes > 3)
            for model in MODELS:
                source = folder / model / "test_predictions.npz"
                result_path = folder / model / "result.json"
                for path in (source, result_path, folder / "splits.json.gz", folder / "manifest.json"):
                    require_current_path(path)
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                result = json.loads(result_path.read_text())
                assert result["suite"] == suite.name and result["split_sha256"] == manifest["split_sha256"]
                with np.load(source) as arrays:
                    np.testing.assert_array_equal(arrays["labels"], labels)
                    np.testing.assert_array_equal(arrays["sizes"], sizes)
                    logits = arrays["logits"]
                assert np.isfinite(logits).all()
                for size in np.unique(sizes):
                    mask = sizes == size
                    y, z = labels[mask], logits[mask]
                    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
                    assert npos == nneg and npos > 0
                    for metric, fn in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
                        value = float(fn(y, z))
                        assert value == result["evaluation"]["test"]["by_size"][str(size)][metric]
                        rows.append(dict(dataset=dataset, seed=seed, model=model, size=int(size),
                                         order_gap=int(size)-3, metric=metric, value=value,
                                         positives=npos, negatives=nneg, split_sha256=result["split_sha256"]))
    summaries = []
    for dataset in config["datasets"]:
        sizes = sorted({r["size"] for r in rows if r["dataset"] == dataset})
        for size in sizes:
            for metric in ("auprc", "auroc"):
                selected = [r for r in rows if (r["dataset"], r["size"], r["metric"]) == (dataset, size, metric)]
                baseline = {r["seed"]: r["value"] for r in selected if r["model"] == "allset"}
                for model in MODELS:
                    runs = sorted((r for r in selected if r["model"] == model), key=lambda r: r["seed"])
                    assert [r["seed"] for r in runs] == config["seeds"]
                    values = [r["value"] for r in runs]
                    deltas = [r["value"] - baseline[r["seed"]] for r in runs]
                    for r, delta in zip(runs, deltas):
                        r["paired_delta"] = delta
                    summaries.append(dict(dataset=dataset, size=size, order_gap=size-3, metric=metric, model=model,
                        mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)),
                        paired_delta_mean=float(np.mean(deltas)), paired_delta_sd=float(np.std(deltas, ddof=1)),
                        paired_deltas=deltas, values=values, positive_counts=[r["positives"] for r in runs],
                        sparse=any(r["positives"] < MIN_POSITIVES for r in runs)))
    for filename, records in (("per_seed.csv", rows), ("summary.csv", summaries)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    artifact = {**plan, "status": "VERIFIED_CURRENT", "source_sha256": hashes,
                "saved_by_size_metrics_recomputed_exactly": True, "records": summaries}
    (out / "analysis.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "figure.dpi": 150})
    for dataset in config["datasets"]:
        sizes = sorted({r["size"] for r in summaries if r["dataset"] == dataset})
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 2, 1]})
        for col, metric in enumerate(("auprc", "auroc")):
            selected = [r for r in summaries if r["dataset"] == dataset and r["metric"] == metric]
            for model, color in zip(MODELS, COLORS):
                runs = sorted((r for r in selected if r["model"] == model), key=lambda r: r["size"])
                x = np.asarray([r["size"] for r in runs])
                axes[0, col].plot(x, [r["mean"] for r in runs], "o-", color=color, label=MODELS[model], ms=4)
                if model == "allset":
                    continue
                shift = {"allset_cpa_encoder": -.13, "allset_cpa": 0, "allset_cpa_both": .13}[model]
                axes[1, col].errorbar(x+shift, [100*r["paired_delta_mean"] for r in runs],
                    yerr=[100*r["paired_delta_sd"] for r in runs], fmt="o", capsize=3,
                    color=color, ms=4, label=MODELS[model])
                for r in runs:
                    axes[1, col].scatter([r["size"]+shift]*3, np.array(r["paired_deltas"])*100,
                                         s=12, color=color, alpha=.35)
            counts = sorted((r for r in selected if r["model"] == "allset"), key=lambda r: r["size"])
            for r in counts:
                lo, hi = min(r["positive_counts"]), max(r["positive_counts"])
                axes[2, col].bar(r["size"], np.mean(r["positive_counts"]), width=.65, color="#a3b8c2")
                axes[2, col].annotate(str(lo) if lo == hi else f"{lo}–{hi}",
                    (r["size"], np.mean(r["positive_counts"])), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8)
                if r["sparse"]:
                    for row in (0, 1, 2):
                        axes[row, col].axvspan(r["size"]-.45, r["size"]+.45, color="#e7c59e", alpha=.3, zorder=-1)
            axes[0, col].set_title(metric.upper())
            axes[0, col].set_ylabel("Mean test score")
            axes[0, col].set_ylim(0, 1.03)
            axes[1, col].axhline(0, color="gray", lw=.8)
            axes[1, col].set_ylabel("CPA − vanilla (percentage points)")
            axes[2, col].set_ylabel("Test positives")
            axes[2, col].set_xlabel("Hyperedge size (training sizes ≤ 3)")
            axes[2, col].set_ylim(0, max(max(r["positive_counts"]) for r in counts)*1.3)
            for row in range(3):
                axes[row, col].set_xticks(sizes)
                axes[row, col].grid(axis="y", alpha=.18)
            axes[0, col].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{dataset}: exact-cardinality test analysis\nPaired points: three seeds; error bars: sample SD. Shading: fewer than 20 positives in a seed.", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{dataset}.{extension}")
        plt.close(fig)
    lines = ["# Current test results by exact cardinality", "", "Descriptive analysis of the verified current campaign. No retraining, resampling, tuning or legacy inputs. Every exact size is retained; sparse means fewer than 20 positives in any seed. AUPRC is average precision, matching the campaign implementation. Counts below are positives per seed; negatives match 1:1.", "", "Means and sample SDs use three paired split/initialization seeds, not independent datasets or confidence intervals. Seed target sets overlap; no significance claim is made. Within-size AP does not average to pooled AP. Scores across sizes use different candidate populations, so their slopes alone do not establish extrapolation behavior.", ""]
    for dataset in config["datasets"]:
        lines += [f"## {dataset}", "", "| Size | Positive count range | Sparse | Encoder Δ AUPRC (pp) | Readout Δ AUPRC (pp) | Both Δ AUPRC (pp) | Encoder Δ AUROC (pp) |", "|---:|---:|:---:|---:|---:|---:|---:|"]
        selected = [r for r in summaries if r["dataset"] == dataset]
        for size in sorted({r["size"] for r in selected}):
            def get(model, metric):
                return next(r for r in selected if (r["model"],r["metric"],r["size"]) == (model,metric,size))
            ref = get("allset", "auprc")
            cells = []
            for model, metric in [(m, "auprc") for m in list(MODELS)[1:]] + [("allset_cpa_encoder", "auroc")]:
                r = get(model, metric)
                cells.append(f"{100*r['paired_delta_mean']:+.2f} ± {100*r['paired_delta_sd']:.2f}")
            lines.append(f"| {size} | {min(ref['positive_counts'])}–{max(ref['positive_counts'])} | {'yes' if ref['sparse'] else 'no'} | " + " | ".join(cells) + " |")
        lines += ["", f"![Cardinality analysis]({dataset}.png)", ""]
    lines += ["## Interpretation", "", "AskUbuntu: encoder CPA improves AUPRC at both test sizes in all three seeds. The mean gain is smaller at size 5 than size 4, so this does not support increasing advantage with order gap. Readout-only scaling has slightly negative mean AP differences at both sizes; combining it with encoder scaling yields smaller gains than encoder-only.", "", "Enron: no monotonic encoder-CPA advantage across sizes 4–8; size 6 has a mean AP loss of 4.57 percentage points. Sizes above 8 contain only 1–9 positives per exact size per seed. Large tail-bin fluctuations are not strong evidence for or against generalization at large orders.", "", "All 24 prediction files were checked against saved candidate labels/sizes and shared split identities. Recomputed exact-size AUPRC/AUROC matched every stored by-size metric exactly. Source file hashes and complete per-seed/model scores are in analysis.json and per_seed.csv."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    registry["cardinality_analysis"] = str(out.relative_to(ROOT))
    (ROOT / "CURRENT_EXPERIMENT.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"Saved verified cardinality analysis to {out}")


if __name__ == "__main__":
    main()
