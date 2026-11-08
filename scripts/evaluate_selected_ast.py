"""Evaluate only the frozen validation-selected AST checkpoints; never train or tune.

Usage: .venv\Scripts\python.exe scripts/evaluate_selected_ast.py --suite results/ast-tuning/SUITE_ID
"""
import argparse
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
from hnn_project.train import evaluate_metrics, predict, seed_everything, write_json
from hnn_project.tune import source_identity


def evaluate(suite, output):
    verification = json.loads((suite / "verification.json").read_text())
    if verification["verified_fits"] != 32:
        raise ValueError("Complete and verify the 32-fit tuning search before testing")
    config = json.loads((suite / "config.json").read_text())
    environment = json.loads((suite / "environment.json").read_text())
    if environment["source_sha256"] != source_identity():
        raise ValueError("Model source has changed since tuning")
    if torch.__version__ != environment["torch"]:
        raise ValueError("PyTorch version differs from the tuning environment")
    torch.set_num_threads(config["threads"])
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"tuning_suite": str(suite.resolve()), "source_sha256": source_identity(),
              "selection": {}, "test_used_for_selection": False}
    for name in config["datasets"]:
        path = suite / name / "selection.json"
        frozen["selection"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "winner": json.loads(path.read_text())["winner"]}
    freeze_path = output / "frozen_selection.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError("Existing test evaluation belongs to a different selection")
    write_json(freeze_path, frozen)
    (output / "evaluation_source.py").write_bytes(Path(__file__).read_bytes())
    summaries = []
    for name in config["datasets"]:
        winner = frozen["selection"][name]["winner"]
        records = []
        for seed in config["seeds"]:
            seed_everything(seed)
            split_dir = suite / name / f"seed-{seed}"
            manifest = json.loads((split_dir / "manifest.json").read_text())
            with gzip.open(split_dir / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            parts = {key: Candidates([tuple(e) for e in saved[key]["edges"]],
                                     np.array(saved[key]["labels"], dtype=np.float32))
                     for key in ("train", "val", "test")}
            split_digest = hashlib.sha256(json.dumps({"context": saved["context"],
                "candidates": [parts[key].edges for key in ("train", "val", "test")]},
                separators=(",", ":")).encode()).hexdigest()
            if split_digest != manifest["split_sha256"]:
                raise ValueError("Saved candidate identity mismatch")
            trial_dir = suite / name / "trials" / winner["trial_id"] / f"seed-{seed}"
            checkpoint_path = trial_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            active = checkpoint["config"]
            if checkpoint["model_name"] != "allset" or checkpoint["split_sha256"] != split_digest:
                raise ValueError("Checkpoint model or split mismatch")
            expected = winner["candidate"]
            if (active["hidden_dim"], active["layers"], active["learning_rates"]) != (
                    expected["hidden_dim"], expected["layers"], [expected["learning_rate"]]):
                raise ValueError("Checkpoint differs from the frozen configuration")
            protocol = Protocol(saved["context"], parts["train"], parts["val"], parts["test"],
                                checkpoint["n_nodes"], manifest)
            model = Predictor(checkpoint["n_nodes"], "allset", active["hidden_dim"], active["layers"],
                              active["heads"], active["dropout"], checkpoint["size_scale"]).to("cuda")
            model.load_state_dict(checkpoint["state_dict"])
            context = make_context(protocol, "cuda")
            logits = predict(model, context, protocol.test, active["batch_size"], "cuda")
            result = {"dataset": name, "seed": seed, "trial_id": winner["trial_id"],
                "split_sha256": split_digest, "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "evaluation": evaluate_metrics(protocol.test, logits),
                "selection": "Frozen using validation only; no training in this evaluation"}
            folder = output / name / f"seed-{seed}"
            write_json(folder / "test_result.json", result)
            np.savez_compressed(folder / "test_predictions.npz", labels=protocol.test.labels,
                                logits=logits, sizes=protocol.test.sizes)
            records.append(result)
            print(name, seed, result["evaluation"]["overall"], flush=True)
        summary = {"dataset": name, "candidate": winner["candidate"], "seeds": config["seeds"],
                   "by_size": {}}
        for metric in ("auprc", "auroc"):
            values = [r["evaluation"]["overall"][metric] for r in records]
            summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                               "per_seed": values}
        for size in sorted({s for r in records for s in r["evaluation"]["by_size"]}, key=int):
            bins = [r["evaluation"]["by_size"][size] for r in records
                    if size in r["evaluation"]["by_size"]]
            entry = {"positive_counts": [b["positives"] for b in bins],
                     "small_bin_any_seed": any(b["small_bin"] for b in bins)}
            for metric in ("auprc", "auroc"):
                values = [b[metric] for b in bins if b[metric] is not None]
                entry[metric] = {"mean": float(np.mean(values)),
                                 "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            summary["by_size"][size] = entry
        summaries.append(summary)
    write_json(output / "summary.json", {"datasets": summaries, "test_used_for_selection": False})
    lines = ["# Frozen vanilla AST test evaluation", "",
             "Selected configurations and checkpoints were frozen using validation before testing. "
             "Mean and sample SD across three paired split/initialization seeds; these are not confidence intervals.", "",
             "| Dataset | Width | Layers | Learning rate | Test AUPRC | Test AUROC |",
             "|---|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        c = summary["candidate"]
        ap, auc = summary["auprc"], summary["auroc"]
        lines.append(f"| {summary['dataset']} | {c['hidden_dim']} | {c['layers']} | {c['learning_rate']} | "
                     f"{ap['mean']:.4f} +/- {ap['std']:.4f} | {auc['mean']:.4f} +/- {auc['std']:.4f} |")
    lines += ["", "Order-shift protocol: k=3; test cardinalities exceed 3; negatives are size-matched at 1:1. "
              "Per-cardinality metrics and small-bin flags are in summary.json and each seed's test_result.json.", "",
              "Test scores must not be used to retune the baseline or select CPA variants. "
              "The same saved candidates and selected training settings are the reference for CPA."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(evaluate(args.suite, args.output or Path("results/ast-test") / args.suite.name))
