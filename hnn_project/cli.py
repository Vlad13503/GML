from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
import torch

from .data import (DATASETS, download, load_dataset, make_context, make_protocol,
                   synthetic_dataset)
from .interpret import analyze
from .models import CORE_MODELS, MODELS
from .report import report
from .train import evaluate_metrics, fit, predict, write_json

DEFAULTS = {
    "datasets": list(DATASETS), "seeds": [0, 1, 2], "models": CORE_MODELS,
    "experiments": [{"protocol": "random", "k": 3}, {"protocol": "order", "k": 3},
                    {"protocol": "interpolation", "k": 3, "interpolation_bounds": [2, 5]}],
    "hidden_dim": 64, "layers": 2, "heads": 4, "dropout": .2,
    "learning_rates": [.001], "weight_decay": .0001, "epochs": 150,
    "patience": 30, "eval_every": 5, "batch_size": 2048, "threads": 4,
    "negative_ratio": 1, "hard_negative_fraction": .5, "context_fraction": .6,
    "min_partition_positives": 10, "probe_limit": 2000,
}


def validate(config):
    for key in ("epochs", "patience", "eval_every", "batch_size", "threads", "layers",
                "hidden_dim", "heads", "negative_ratio", "min_partition_positives", "probe_limit"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["hidden_dim"] % config["heads"]:
        raise ValueError("hidden_dim must be divisible by heads")
    if not 0 < config["context_fraction"] < 1:
        raise ValueError("context_fraction must be in (0, 1)")
    if not 0 <= config["hard_negative_fraction"] <= 1 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid probability")
    for key in ("datasets", "seeds", "models", "experiments", "learning_rates"):
        if not config[key]:
            raise ValueError(f"{key} cannot be empty")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Repeated seeds are not independent runs")
    if any(not isinstance(s, int) or s < 0 for s in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if any(n not in (*DATASETS, "synthetic") for n in config["datasets"]):
        raise ValueError("Unknown dataset")
    if any(n not in MODELS for n in config["models"]):
        raise ValueError("Unknown model")
    if len(set(config["models"])) != len(config["models"]) or len(set(config["datasets"])) != len(config["datasets"]):
        raise ValueError("Datasets and models must be unique")
    if any(lr <= 0 for lr in config["learning_rates"]):
        raise ValueError("Learning rates must be positive")
    seen_experiments = set()
    for experiment in config["experiments"]:
        if set(experiment) - {"protocol", "k", "interpolation_bounds"}:
            raise ValueError("Experiments may set only protocol, k, and interpolation_bounds")
        identity = (experiment["protocol"], experiment.get("k", 3))
        if identity in seen_experiments:
            raise ValueError("Each (protocol, k) must be unique; run alternative interpolation bounds in a separate suite")
        seen_experiments.add(identity)
        if experiment["protocol"] not in ("random", "order", "interpolation"):
            raise ValueError("Unknown experiment protocol")
        if experiment.get("k", 3) < 2:
            raise ValueError("k must be at least 2")
        if experiment["protocol"] == "interpolation":
            low, high = experiment["interpolation_bounds"]
            if low < 2 or high <= low + 1:
                raise ValueError("Interpolation requires 2 <= low < high - 1")
    return config


def run_suite(config, data_root: Path, output: Path, device: str):
    validate(config)
    torch.set_num_threads(config["threads"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    datasets = {name: synthetic_dataset() if name == "synthetic" else load_dataset(name, data_root)
                for name in config["datasets"]}
    provenance = {"device": device, "torch": torch.__version__, "numpy": np.__version__,
                  "sklearn": sklearn.__version__, "data": {n: d.audit for n, d in datasets.items()}}
    signature = hashlib.sha256(json.dumps({"config": config, "provenance": provenance}, sort_keys=True).encode()
                               + code_hash.digest()).hexdigest()[:12]
    root = output / signature
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    write_json(root / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
        "device": device, "threads": config["threads"], "code_sha256": code_hash.hexdigest(),
        "deterministic_algorithms": True})
    print(f"Suite: {root.resolve()} | device={device}", flush=True)
    for name in config["datasets"]:
        dataset = datasets[name]
        write_json(root / name / "audit.json", dataset.audit)
        for experiment in config["experiments"]:
            active = {**config, **experiment}
            tag = f"{active['protocol']}-k{active.get('k', 3)}"
            if active["protocol"] == "interpolation":
                tag += "-" + "-".join(map(str, active["interpolation_bounds"]))
            for seed in config["seeds"]:
                protocol = make_protocol(dataset, active, seed)
                split_dir = root / name / tag / f"seed-{seed}"
                split_dir.mkdir(parents=True, exist_ok=True)
                previous = split_dir / "manifest.json"
                if previous.exists() and json.loads(previous.read_text())["split_sha256"] != protocol.manifest["split_sha256"]:
                    raise ValueError("Existing split differs; use a fresh output directory")
                write_json(previous, protocol.manifest)
                split_file = split_dir / "splits.json.gz"
                if not split_file.exists():
                    with gzip.open(split_file, "wt", encoding="utf-8") as stream:
                        json.dump({"context": protocol.context,
                            **{n: {"edges": p.edges, "labels": p.labels.tolist()} for n, p in
                               (("train", protocol.train), ("val", protocol.val), ("test", protocol.test))}},
                                  stream, separators=(",", ":"))
                context = make_context(protocol, device, clique="gcn" in config["models"])
                for model_name in config["models"]:
                    model_dir = split_dir / model_name
                    if (model_dir / "result.json").exists():
                        print(f"Skip completed: {name}/{tag}/seed-{seed}/{model_name}", flush=True)
                        continue
                    print(f"Run {name}/{tag}/seed-{seed}/{model_name}", flush=True)
                    model, training = fit(model_name, protocol, context, active, seed, model_dir, device)
                    started, evaluation = time.perf_counter(), {}
                    for split_name in ("val", "test"):
                        part = getattr(protocol, split_name)
                        logits = predict(model, context, part, active["batch_size"], device)
                        evaluation[split_name] = evaluate_metrics(part, logits)
                        np.savez_compressed(model_dir / f"{split_name}_predictions.npz",
                                            labels=part.labels, logits=logits, sizes=part.sizes)
                    elapsed = time.perf_counter() - started
                    analyze(model, context, protocol, active, seed, model_dir, device)
                    training = {k: v for k, v in training.items() if k != "trials"}
                    write_json(model_dir / "result.json", {
                        "dataset": name, "protocol": active["protocol"], "k": active.get("k", 3),
                        "interpolation_bounds": active.get("interpolation_bounds"),
                        "model": model_name, "seed": seed, "split_sha256": protocol.manifest["split_sha256"],
                        "training": training, "evaluation_seconds": elapsed, "evaluation": evaluation})
                    print(f"  complete: test_AP={evaluation['test']['overall']['auprc']:.4f}", flush=True)
    print(f"Report: {report(root).resolve()}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        command.add_argument("--data-dir", type=Path, default=Path("data"))
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/core.json"))
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    command = sub.add_parser("report")
    command.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        for name in args.datasets:
            print(download(name, args.data_dir).resolve(), flush=True)
    elif args.command == "audit":
        for name in args.datasets:
            dataset = load_dataset(name, args.data_dir)
            write_json(args.data_dir / name / "audit.json", dataset.audit)
            print(json.dumps(dataset.audit, indent=2))
    elif args.command == "run":
        supplied = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        run_suite({**DEFAULTS, **supplied}, args.data_dir, args.output, args.device)
    else:
        print(report(args.path).resolve())
