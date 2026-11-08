"""Train encoder-only scaled CPA against frozen, validation-selected AST runs."""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from hnn_project.data import Candidates, Protocol, make_context
from hnn_project.models import Predictor
from hnn_project.train import evaluate_metrics, fit, predict, seed_everything, write_json
from hnn_project.tune import source_identity


def matched_initialization(n_nodes, config, seed, size_scale):
    models = []
    for name in ("allset", "allset_cpa_encoder"):
        seed_everything(seed)
        models.append(Predictor(n_nodes, name, config["hidden_dim"], config["layers"],
                                config["heads"], config["dropout"], size_scale))
    vanilla, cpa = models
    assert set(vanilla.state_dict()) == set(cpa.state_dict())
    digest = hashlib.sha256()
    for key, value in vanilla.state_dict().items():
        torch.testing.assert_close(value, cpa.state_dict()[key], rtol=0, atol=0)
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    assert cpa.pool.mode == "none"
    assert all(layer.v2e.mode == "scaled" and layer.e2v.mode == "none" for layer in cpa.layers)
    return digest.hexdigest()


def run(suite, baseline_test, output):
    environment = json.loads((suite / "environment.json").read_text())
    if source_identity() != environment["source_sha256"] or torch.__version__ != environment["torch"]:
        raise ValueError("Model source or PyTorch differs from the frozen baseline")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matched experiment")
    config = json.loads((suite / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    output.mkdir(parents=True, exist_ok=True)
    plan = {"model": "allset_cpa_encoder", "modification": "v2e fixed scaling by context hyperedge size before LN",
            "tuning_suite": str(suite.resolve()), "baseline_test": str(baseline_test.resolve()),
            "source_sha256": source_identity(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "datasets": {}, "test_used_for_selection": False}
    for name in config["datasets"]:
        selection_path = suite / name / "selection.json"
        plan["datasets"][name] = {"selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                                  "winner": json.loads(selection_path.read_text())["winner"]}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing CPA output has a different plan")
    write_json(plan_path, plan)
    (output / "training_source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for name in config["datasets"]:
        winner = plan["datasets"][name]["winner"]
        for seed in config["seeds"]:
            seed_dir = suite / name / f"seed-{seed}"
            with gzip.open(seed_dir / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            manifest = json.loads((seed_dir / "manifest.json").read_text())
            parts = {key: Candidates([tuple(e) for e in saved[key]["edges"]],
                                    np.array(saved[key]["labels"], dtype=np.float32))
                     for key in ("train", "val", "test")}
            split_hash = hashlib.sha256(json.dumps({"context": saved["context"],
                "candidates": [parts[key].edges for key in ("train", "val", "test")]},
                separators=(",", ":")).encode()).hexdigest()
            if split_hash != manifest["split_sha256"]:
                raise ValueError("Candidate hash mismatch")
            baseline_dir = suite / name / "trials" / winner["trial_id"] / f"seed-{seed}"
            baseline_path = baseline_dir / "checkpoint.pt"
            checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
            baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            test_dir = baseline_test / name / f"seed-{seed}"
            baseline_result = json.loads((test_dir / "test_result.json").read_text())
            if (checkpoint["model_name"] != "allset" or checkpoint["split_sha256"] != split_hash
                    or baseline_result["split_sha256"] != split_hash
                    or baseline_result["checkpoint_sha256"] != baseline_hash):
                raise ValueError("Baseline provenance mismatch")
            active = {**checkpoint["config"], "models": ["allset_cpa_encoder"]}
            protocol = Protocol(saved["context"], parts["train"], parts["val"], parts["test"],
                                checkpoint["n_nodes"], manifest)
            initial_hash = matched_initialization(protocol.n_nodes, active, seed, checkpoint["size_scale"])
            folder = output / name / f"seed-{seed}"
            identity = {"dataset": name, "seed": seed, "config": active, "split_sha256": split_hash,
                        "baseline_checkpoint_sha256": baseline_hash, "initial_state_sha256": initial_hash,
                        "initial_weights_and_parameter_counts_matched": True, "source_sha256": source_identity()}
            result_path = folder / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["identity"] != identity:
                    raise ValueError("Completed CPA fit has a different identity")
                records.append(result)
                continue
            write_json(folder / "identity.json", identity)
            context = make_context(protocol, "cuda")
            training_protocol = replace(protocol, test=Candidates([], np.empty(0, dtype=np.float32)))
            print(f"Train encoder CPA {name} seed={seed}", flush=True)
            model, training = fit("allset_cpa_encoder", training_protocol, context, active, seed, folder, "cuda")
            if training["parameters"] != json.loads((baseline_dir / "training.json").read_text())["parameters"]:
                raise ValueError("Parameter count differs from baseline")
            evaluation = {}
            for split in ("val", "test"):
                part = parts[split]
                logits = predict(model, context, part, active["batch_size"], "cuda")
                evaluation[split] = evaluate_metrics(part, logits)
                np.savez_compressed(folder / f"{split}_predictions.npz", labels=part.labels,
                                    logits=logits, sizes=part.sizes)
            if abs(evaluation["val"]["overall"]["auprc"] - training["validation_auprc"]) > 1e-12:
                raise ValueError("Selected CPA checkpoint does not replay validation AP")
            with np.load(test_dir / "test_predictions.npz") as baseline_predictions:
                np.testing.assert_array_equal(baseline_predictions["labels"], parts["test"].labels)
                np.testing.assert_array_equal(baseline_predictions["sizes"], parts["test"].sizes)
            result = {"identity": identity, "training": training, "evaluation": evaluation,
                      "baseline_test": baseline_result["evaluation"],
                      "paired_delta": {metric: evaluation["test"]["overall"][metric]
                          - baseline_result["evaluation"]["overall"][metric] for metric in ("auprc", "auroc")}}
            write_json(result_path, result)
            records.append(result)
            write_json(output / "progress.json", {"completed_fits": len(records), "planned_fits": 6})
            print(f"  test AP={evaluation['test']['overall']['auprc']:.4f}; "
                  f"paired delta={result['paired_delta']['auprc']:+.4f}", flush=True)
    summaries = []
    for name in config["datasets"]:
        selected = [r for r in records if r["identity"]["dataset"] == name]
        summary = {"dataset": name, "seeds": config["seeds"], "by_size": {}}
        for metric in ("auprc", "auroc"):
            summary[metric] = {}
            for label, values in (
                    ("vanilla", [r["baseline_test"]["overall"][metric] for r in selected]),
                    ("cpa", [r["evaluation"]["test"]["overall"][metric] for r in selected]),
                    ("paired_delta", [r["paired_delta"][metric] for r in selected])):
                summary[metric][label] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                                         "per_seed": values}
        for size in sorted({s for r in selected for s in r["evaluation"]["test"]["by_size"]}, key=int):
            bins = [(r["baseline_test"]["by_size"][size], r["evaluation"]["test"]["by_size"][size])
                    for r in selected if size in r["evaluation"]["test"]["by_size"]]
            entry = {"order_gap": int(size) - 3, "positive_counts": [b[1]["positives"] for b in bins],
                     "small_bin_any_seed": any(b[1]["small_bin"] for b in bins)}
            for metric in ("auprc", "auroc"):
                entry[metric] = {"vanilla_mean": float(np.mean([b[0][metric] for b in bins])),
                                 "cpa_mean": float(np.mean([b[1][metric] for b in bins])),
                                 "paired_deltas": [b[1][metric] - b[0][metric] for b in bins]}
            summary["by_size"][size] = entry
        summaries.append(summary)
    write_json(output / "comparison.json", {"datasets": summaries, "completed_fits": len(records)})
    lines = ["# Encoder-only scaled CPA versus frozen vanilla AST", "",
             "Same saved contexts, candidates, initialization seeds, parameter counts and validated AST settings. "
             "CPA is trained end to end from initialization. Checkpoints are selected independently by validation AP. "
             "Mean +/- sample SD across three seeds; these are not confidence intervals.", "",
             "| Dataset | Metric | Vanilla | Encoder CPA | Paired delta |", "|---|---|---:|---:|---:|"]
    for summary in summaries:
        for metric in ("auprc", "auroc"):
            m = summary[metric]
            lines.append(f"| {summary['dataset']} | {metric.upper()} | "
                f"{m['vanilla']['mean']:.4f} +/- {m['vanilla']['std']:.4f} | "
                f"{m['cpa']['mean']:.4f} +/- {m['cpa']['std']:.4f} | "
                f"{m['paired_delta']['mean']:+.4f} +/- {m['paired_delta']['std']:.4f} |")
    lines += ["", "Order-shift protocol: k=3, closed context-node universe, 1:1 size-matched negatives. "
              "Only node-to-hyperedge encoder aggregates are scaled before residual/LayerNorm; "
              "hyperedge-to-node and candidate pooling are unchanged. Encoder context contains only small hyperedges.", "",
              "This is a fixed-hyperparameter ablation, not a separately tuned CPA optimum. "
              "Per-cardinality paired results and small-bin flags are in comparison.json. "
              "No normalization mechanism conclusion follows from task scores alone."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run(args.suite, args.baseline_test or Path("results/ast-test") / args.suite.name,
              args.output or Path("results/encoder-cpa") / args.suite.name))
