"""Validation-only AST search: 12 seed-0 fits, then two finalists on seeds 1/2.

Test candidates are generated and frozen for later matched evaluation, never scored.
Run stages in order: audit, pilot, tune. Completed fits resume by suite identity.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np
import sklearn
import torch

from .cli import DEFAULTS, validate
from .data import Candidates, load_dataset, make_context, make_protocol, synthetic_dataset
from .train import evaluate_metrics, fit, predict, write_json


def grid(config):
    return [{"hidden_dim": dim, "layers": layers, "learning_rate": lr}
            for dim, layers, lr in itertools.product((32, 64), (1, 2), config["learning_rates"])]


def trial_id(candidate):
    return f"d{candidate['hidden_dim']}-l{candidate['layers']}-lr{candidate['learning_rate']:g}"


def rank_candidates(records):
    # Explicit tie-breaker makes selection reproducible without consulting test data.
    return sorted(records, key=lambda r: (-r["validation_auprc"], r["trial_id"]))


def source_identity():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_protocol(root, protocol):
    folder = root / f"seed-{protocol.manifest['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    expected_manifest = json.loads(json.dumps(protocol.manifest))
    if manifest.exists() and json.loads(manifest.read_text()) != expected_manifest:
        raise ValueError("Protocol differs from the saved manifest")
    write_json(manifest, protocol.manifest)
    path = folder / "splits.json.gz"
    if not path.exists():
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"context": protocol.context, **{
                name: {"edges": part.edges, "labels": part.labels.tolist()}
                for name, part in (("train", protocol.train), ("val", protocol.val), ("test", protocol.test))
            }}, stream, separators=(",", ":"))
    return folder


def validation_fit(protocol, context, config, seed, output, device, candidate):
    output.mkdir(parents=True, exist_ok=True)
    active = {**config, **{k: candidate[k] for k in ("hidden_dim", "layers")},
              "learning_rates": [candidate["learning_rate"]]}
    identity = {"config": active, "seed": seed, "split_sha256": protocol.manifest["split_sha256"],
                "candidate": candidate, "device": device, "source_sha256": source_identity()}
    result_path = output / "validation_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["identity"] != identity:
            raise ValueError("Saved trial has a different identity")
        return result
    # fit needs only train/validation plus metadata. Remove test candidates defensively.
    training_protocol = replace(protocol, test=Candidates([], np.empty(0, dtype=np.float32)))
    write_json(output / "identity.json", identity)
    print(f"Fit {protocol.manifest['dataset']} {trial_id(candidate)} seed={seed}", flush=True)
    model, training = fit("allset", training_protocol, context, active, seed, output, device)
    logits = predict(model, context, protocol.val, config["batch_size"], device)
    evaluation = evaluate_metrics(protocol.val, logits)
    if not np.isclose(evaluation["overall"]["auprc"], training["validation_auprc"], atol=1e-12):
        raise ValueError("Validation checkpoint replay differs from selection score")
    np.savez_compressed(output / "val_predictions.npz", labels=protocol.val.labels,
                        logits=logits, sizes=protocol.val.sizes)
    result = {"identity": identity, "trial_id": trial_id(candidate), "candidate": candidate,
              "seed": seed, "validation_auprc": training["validation_auprc"],
              "evaluation": evaluation, "training": training, "test_evaluated": False}
    write_json(result_path, result)
    print(f"  validation AP={result['validation_auprc']:.4f}; "
          f"{training['training_seconds_all_trials']:.1f}s", flush=True)
    return result


def run(config, data_root, output, device, stage):
    validate(config)
    if config["models"] != ["allset"] or len(config["experiments"]) != 1:
        raise ValueError("This search supports vanilla AST and one fixed protocol")
    if config["seeds"] != [0, 1, 2] or len(grid(config)) != 12:
        raise ValueError("Expected 12 configurations and seeds [0, 1, 2]")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["threads"])
    datasets = {name: synthetic_dataset() if name == "synthetic" else load_dataset(name, data_root)
                for name in config["datasets"]}
    provenance = {"python": platform.python_version(), "torch": torch.__version__,
                  "numpy": np.__version__, "sklearn": sklearn.__version__, "device": device,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                  "source_sha256": source_identity(), "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG")}
    signature = hashlib.sha256(json.dumps({"config": config, "provenance": provenance,
        "data": {n: d.audit for n, d in datasets.items()}}, sort_keys=True).encode()).hexdigest()[:12]
    root = output / signature
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    write_json(root / "environment.json", provenance)
    # Preserve the executed Python sources even before changes have been committed.
    for source in Path(__file__).parent.glob("*.py"):
        snapshot = root / "source" / source.name
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    write_json(root / "selection_plan.json", {"grid": grid(config), "screen_seed": 0,
        "confirmation_seeds": [1, 2], "finalists": 2, "selection": "mean validation AUPRC",
        "tie_break": "trial_id ascending", "max_fits_per_dataset": 16, "test_evaluated": False})
    print(f"Tuning suite: {root.resolve()} | stage={stage} | device={device}", flush=True)
    for name, dataset in datasets.items():
        folder = root / name
        write_json(folder / "audit.json", dataset.audit)
        protocols = {}
        for seed in config["seeds"]:
            protocol = make_protocol(dataset, {**config, **config["experiments"][0]}, seed)
            save_protocol(folder, protocol)
            protocols[seed] = protocol
            print(f"Audit {name} seed={seed}: {protocol.manifest['partitions']}; "
                  f"dropped={protocol.manifest['dropped_cold_node_positives']}", flush=True)
        if stage == "audit":
            continue
        if stage == "pilot":
            context = make_context(protocols[0], device)
            # Benchmark the largest planned architecture; these pilot fits cannot enter selection.
            active = {**config, "epochs": 5, "patience": 5, "eval_every": 1}
            validation_fit(protocols[0], context, active, 0, folder / "pilot", device,
                           {"hidden_dim": 64, "layers": 2, "learning_rate": .001})
            continue
        records = []
        context = make_context(protocols[0], device)
        for candidate in grid(config):
            records.append(validation_fit(protocols[0], context, config, 0,
                folder / "trials" / trial_id(candidate) / "seed-0", device, candidate))
            write_json(folder / "progress.json", {"completed_fits": len(records), "phase": "screen",
                       "test_evaluated": False})
        finalists = rank_candidates(records)[:2]
        write_json(folder / "finalists.json", [{"trial_id": r["trial_id"], "candidate": r["candidate"],
                   "seed0_validation_auprc": r["validation_auprc"]} for r in finalists])
        for seed in (1, 2):
            context = make_context(protocols[seed], device)
            for finalist in finalists:
                candidate = finalist["candidate"]
                records.append(validation_fit(protocols[seed], context, config, seed,
                    folder / "trials" / trial_id(candidate) / f"seed-{seed}", device, candidate))
                write_json(folder / "progress.json", {"completed_fits": len(records), "phase": "confirm",
                           "test_evaluated": False})
        comparison = []
        for finalist in finalists:
            scores = [r["validation_auprc"] for r in records if r["trial_id"] == finalist["trial_id"]]
            comparison.append({"trial_id": finalist["trial_id"], "candidate": finalist["candidate"],
                "validation_auprc": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
                "per_seed": scores})
        ranked = rank_candidates(comparison)
        winner = ranked[0]
        selected = {**config, **{k: winner["candidate"][k] for k in ("hidden_dim", "layers")},
                    "datasets": [name], "learning_rates": [winner["candidate"]["learning_rate"]]}
        write_json(folder / "selected_config.json", selected)
        write_json(folder / "selection.json", {"winner": winner, "finalists": ranked,
            "completed_fits": len(records), "test_evaluated": False,
            "scope": "Two-stage validation-selected configuration; not a test result"})
        print(f"Selected {name}: {winner}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ast_tuning.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/ast-tuning"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("audit", "pilot", "tune"), default="audit")
    args = parser.parse_args()
    # Must precede the first CUDA context for deterministic cuBLAS matrix products.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    supplied = json.loads(args.config.read_text(encoding="utf-8"))
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    print(run({**DEFAULTS, **supplied}, args.data_dir, args.output, args.device, args.stage))


if __name__ == "__main__":
    main()
