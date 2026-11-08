"""Frozen encoder aggregation intervention. --dry-run checks inputs without running the sweep."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from hnn_project.data import Candidates, Protocol, make_context
from hnn_project.models import Predictor
from hnn_project.train import seed_everything, write_json
from hnn_project.tune import source_identity


def stages_from_raw(pool, raw, multiplier, zero_residual=False, disable_scaling=False):
    """Hold raw attention fixed: zero_residual never changes keys, scores or weights."""
    if pool.mode not in ("none", "scaled"):
        raise ValueError("This diagnostic supports vanilla and fixed scaled CPA only")
    scale = torch.as_tensor(multiplier, device=raw.device, dtype=raw.dtype)
    if scale.ndim == 1:
        scale = scale[:, None]
    modified = raw * scale if pool.mode == "scaled" and not disable_scaling else raw
    query = torch.zeros_like(pool.query) if zero_residual else pool.query
    residual = modified + query.reshape(1, -1)
    post_norm = pool.norm1(residual)
    representation = pool.norm2(post_norm + F.relu(pool.ff(post_norm)))
    return {"attention": raw, "modified": modified, "pre_norm": residual,
            "post_norm": post_norm, "representation": representation}


def distances(value, reference):
    norm = value.norm(dim=1)
    ref_norm = reference.norm(dim=1)
    l2 = (value - reference).norm(dim=1)
    valid = (norm > 1e-8) & (ref_norm > 1e-8)
    cosine = F.cosine_similarity(value, reference, dim=1, eps=1e-8)
    cosine = torch.where(valid, cosine, torch.full_like(cosine, float("nan")))
    return {"norm": norm, "l2": l2, "relative_l2": l2 / ref_norm.clamp_min(1e-8),
            "cosine": cosine, "cosine_valid": valid}


@torch.no_grad()
def run(suite, cpa_root, output, config, dry_run):
    environment = json.loads((suite / "environment.json").read_text())
    if source_identity() != environment["source_sha256"] or torch.__version__ != environment["torch"]:
        raise ValueError("Model code or PyTorch differs from the frozen experiment")
    if not config["exclude_synthetic_singletons"] or config["reference_multiplier"] not in config["multipliers"]:
        raise ValueError("Require real context edges and a reference in the multiplier grid")
    if any(m <= 0 for m in config["multipliers"]):
        raise ValueError("Only positive cardinality multipliers are supported")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    plan = {"config": config, "suite": str(suite.resolve()), "cpa_root": str(cpa_root.resolve()),
            "source_sha256": source_identity(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference": "m=1 within the same checkpoint, edge, layer and control condition",
            "controls": ["vanilla actual (multiplier ignored)", "CPA residual zero (attention fixed)",
                         "CPA scaling disabled (same trained weights)"],
            "scope": "Encoder block sensitivity only; no task scoring or causal real-interaction interpretation",
            "aggregation": "Summarize sampled edges within each seed before comparing the three seeds",
            "checkpoint_inputs": []}
    rows = []
    prepared = []
    for name in json.loads((suite / "config.json").read_text())["datasets"]:
        selection = json.loads((suite / name / "selection.json").read_text())
        for seed in (0, 1, 2):
            seed_everything(seed)
            with gzip.open(suite / name / f"seed-{seed}" / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            sizes = np.array(list(map(len, saved["context"])))
            rng = np.random.default_rng(seed + config["sampling_seed_offset"])
            selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(sizes == size),
                min(config["sample_per_cardinality"], int((sizes == size).sum())), replace=False)
                for size in np.unique(sizes) if size >= 2]))
            ids = torch.as_tensor(selected, device="cuda")
            for model_name, checkpoint_path in (
                    ("allset", suite / name / "trials" / selection["winner"]["trial_id"] / f"seed-{seed}" / "checkpoint.pt"),
                    ("allset_cpa_encoder", cpa_root / name / f"seed-{seed}" / "checkpoint.pt")):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                active = checkpoint["config"]
                if checkpoint["model_name"] != model_name:
                    raise ValueError("Unexpected checkpoint model")
                manifest = json.loads((suite / name / f"seed-{seed}" / "manifest.json").read_text())
                if checkpoint["split_sha256"] != manifest["split_sha256"]:
                    raise ValueError("Checkpoint and context split differ")
                empty = Candidates([], np.empty(0, dtype=np.float32))
                protocol = Protocol(saved["context"], empty, empty, empty, checkpoint["n_nodes"], manifest)
                model = Predictor(protocol.n_nodes, model_name, active["hidden_dim"], active["layers"],
                                  active["heads"], active["dropout"], checkpoint["size_scale"]).to("cuda").eval()
                model.load_state_dict(checkpoint["state_dict"])
                captured = []
                hooks = []
                for layer_id, layer in enumerate(model.layers):
                    def capture(pool, args, actual, layer_id=layer_id):
                        _, trace = pool.forward(*args, trace=True)
                        raw = trace["attention"][ids].clone()
                        reconstructed = stages_from_raw(pool, raw, args[3][ids])
                        torch.testing.assert_close(reconstructed["representation"], actual[ids], rtol=1e-5, atol=1e-6)
                        captured.append((layer_id, pool, raw))
                    hooks.append(layer.v2e.register_forward_hook(capture))
                try:
                    model.encode(make_context(protocol, "cuda"))
                finally:
                    for hook in hooks:
                        hook.remove()
                plan["checkpoint_inputs"].append({"dataset": name, "model": model_name, "seed": seed,
                    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "split_sha256": manifest["split_sha256"], "context_edge_ids": selected.tolist()})
                prepared.append({"dataset": name, "model": model_name, "seed": seed,
                                 "layers": len(captured), "sampled_context_edges": len(selected)})
                if dry_run:
                    continue
                for layer_id, pool, raw in captured:
                    conditions = [("actual", False, False)]
                    if model_name == "allset_cpa_encoder":
                        conditions += [("residual_zero", True, False), ("scaling_disabled", False, True)]
                    for condition, zero, disabled in conditions:
                        reference = stages_from_raw(pool, raw, config["reference_multiplier"], zero, disabled)
                        for multiplier in config["multipliers"]:
                            stages = stages_from_raw(pool, raw, multiplier, zero, disabled)
                            for stage, values in stages.items():
                                metrics = {k: v.cpu().numpy() for k, v in distances(values, reference[stage]).items()}
                                for index, edge_id in enumerate(selected):
                                    rows.append({"dataset": name, "model": model_name, "seed": seed,
                                        "layer": layer_id, "context_edge_id": int(edge_id), "actual_size": int(sizes[edge_id]),
                                        "condition": condition, "multiplier": multiplier, "stage": stage,
                                        **{k: v[index].item() for k, v in metrics.items()}})
    write_json(output / ("dry_run_plan.json" if dry_run else "plan.json"), plan)
    write_json(output / ("dry_run_inputs.json" if dry_run else "inputs.json"), prepared)
    if not dry_run:
        with (output / "per_edge.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "diagnostic_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"dry_run": dry_run, "checkpoint_inputs": len(prepared),
                      "metric_rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cpa-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/cardinality_diagnostic.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.suite, args.cpa_root or Path("results/encoder-cpa") / args.suite.name,
        args.output or Path("results/cardinality-diagnostic") / args.suite.name,
        json.loads(args.config.read_text()), args.dry_run)
