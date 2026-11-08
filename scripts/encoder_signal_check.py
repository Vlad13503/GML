"""Validation-only downstream intervention on frozen encoder-CPA checkpoints."""
import argparse
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
from hnn_project.train import evaluate_metrics, predict, seed_everything, write_json
from hnn_project.tune import source_identity


@torch.no_grad()
def encode_with_condition(model, context, condition, constant=3):
    """Override only v2e scaling, recomputing every encoder layer; preserve self-messages."""
    if condition not in ("normal", "disabled", "constant"):
        raise ValueError("Unknown intervention")
    if model.name != "allset_cpa_encoder" or model.pool.mode != "none":
        raise ValueError("Expected encoder-only CPA with unchanged readout")
    model.eval()
    hooks = []

    def override(pool, args, kwargs):
        sizes = args[3]
        # All singleton incidences here are synthetic self-messages, not target interactions.
        multiplier = 1 if condition == "disabled" else constant
        effective = torch.where(sizes == 1, torch.ones_like(sizes), torch.full_like(sizes, multiplier))
        return args, {**kwargs, "size_override": effective}

    try:
        if condition != "normal":
            for layer in model.layers:
                if layer.v2e.mode != "scaled" or layer.e2v.mode != "none":
                    raise ValueError("Unexpected encoder CPA placement")
                hooks.append(layer.v2e.register_forward_pre_hook(override, with_kwargs=True))
        return model.encode(context)
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def run(suite, cpa_root, output):
    environment = json.loads((suite / "environment.json").read_text())
    if source_identity() != environment["source_sha256"] or torch.__version__ != environment["torch"]:
        raise ValueError("Model code or PyTorch changed since the frozen experiment")
    torch.set_num_threads(4)
    conditions = ["normal", "disabled", "constant"]
    plan = {"suite": str(suite.resolve()), "cpa_root": str(cpa_root.resolve()),
        "conditions": {"normal": "actual context cardinality", "disabled": "m=1 for real context edges",
                       "constant": "m=3 for real context edges"},
        "synthetic_singletons": "m=1 in every condition", "evaluation_split": "validation only",
        "scope": "Frozen CPA-trained weights, no retraining or condition selection",
        "source_sha256": source_identity(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    (output / "intervention_source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for dataset in json.loads((suite / "config.json").read_text())["datasets"]:
        for seed in (0, 1, 2):
            seed_everything(seed)
            folder = cpa_root / dataset / f"seed-{seed}"
            checkpoint_path = folder / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["model_name"] != "allset_cpa_encoder":
                raise ValueError("Incorrect checkpoint variant")
            config = checkpoint["config"]
            split_dir = suite / dataset / f"seed-{seed}"
            manifest = json.loads((split_dir / "manifest.json").read_text())
            with gzip.open(split_dir / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            digest = hashlib.sha256(json.dumps({"context": saved["context"],
                "candidates": [saved[key]["edges"] for key in ("train", "val", "test")]},
                separators=(",", ":")).encode()).hexdigest()
            if digest != checkpoint["split_sha256"] or digest != manifest["split_sha256"]:
                raise ValueError("Frozen split identity mismatch")
            val = Candidates([tuple(e) for e in saved["val"]["edges"]],
                             np.array(saved["val"]["labels"], dtype=np.float32))
            empty = Candidates([], np.empty(0, dtype=np.float32))
            protocol = Protocol(saved["context"], empty, val, empty, checkpoint["n_nodes"], manifest)
            model = Predictor(protocol.n_nodes, checkpoint["model_name"], config["hidden_dim"],
                config["layers"], config["heads"], config["dropout"], checkpoint["size_scale"]).to("cuda")
            model.load_state_dict(checkpoint["state_dict"])
            context = make_context(protocol, "cuda")
            covered = torch.tensor(sorted({v for edge in saved["context"] for v in edge}), device="cuda")
            reference_embeddings = None
            reference_logits = None
            reference_metrics = None
            results = {}
            run_folder = output / dataset / f"seed-{seed}"
            for condition in conditions:
                embeddings = encode_with_condition(model, context, condition)
                logits = predict(model, None, val, config["batch_size"], "cuda", embeddings=embeddings)
                metrics = evaluate_metrics(val, logits)
                if condition == "normal":
                    reference_embeddings = embeddings.clone()
                    reference_logits = logits.copy()
                    reference_metrics = metrics
                    with np.load(folder / "val_predictions.npz") as expected:
                        np.testing.assert_array_equal(logits, expected["logits"])
                        np.testing.assert_array_equal(val.labels, expected["labels"])
                        np.testing.assert_array_equal(val.sizes, expected["sizes"])
                before, after = reference_embeddings[covered], embeddings[covered]
                node_relative_l2 = (after - before).norm(dim=1) / before.norm(dim=1).clamp_min(1e-8)
                results[condition] = {"evaluation": metrics,
                    "covered_node_embedding_mean_relative_l2": float(node_relative_l2.mean()),
                    "covered_node_embedding_mean_cosine": float(F.cosine_similarity(after, before, dim=1).mean()),
                    "mean_absolute_logit_change": float(np.mean(np.abs(logits - reference_logits))),
                    "max_absolute_logit_change": float(np.max(np.abs(logits - reference_logits))),
                    "zero_logit_threshold_flip_fraction": float(np.mean((logits >= 0) != (reference_logits >= 0))),
                    "delta_auprc": metrics["overall"]["auprc"] - reference_metrics["overall"]["auprc"],
                    "delta_auroc": metrics["overall"]["auroc"] - reference_metrics["overall"]["auroc"]}
                run_folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(run_folder / f"{condition}_validation.npz", labels=val.labels,
                                    sizes=val.sizes, logits=logits,
                                    covered_node_ids=covered.cpu().numpy(), covered_node_embeddings=after.cpu().numpy())
            # Check intervention hooks cannot affect subsequent ordinary inference.
            restored = predict(model, context, val, config["batch_size"], "cuda")
            np.testing.assert_array_equal(restored, reference_logits)
            result = {"dataset": dataset, "seed": seed, "split_sha256": digest,
                      "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      "conditions": results, "normal_replay_bitwise_exact": True, "test_evaluated": False}
            write_json(run_folder / "result.json", result)
            records.append(result)
            print(dataset, seed, {c: results[c]["evaluation"]["overall"]["auprc"] for c in conditions}, flush=True)
    summaries = []
    for dataset in sorted({r["dataset"] for r in records}):
        selected = [r for r in records if r["dataset"] == dataset]
        for condition in conditions:
            summary = {"dataset": dataset, "condition": condition}
            for metric in ("auprc", "auroc"):
                values = [r["conditions"][condition]["evaluation"]["overall"][metric] for r in selected]
                summary[metric] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1))}
            for metric in ("covered_node_embedding_mean_relative_l2", "mean_absolute_logit_change", "delta_auprc", "delta_auroc"):
                values = [r["conditions"][condition][metric] for r in selected]
                summary[metric] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)),
                                   "per_seed": values}
            summaries.append(summary)
    write_json(output / "summary.json", {"records": summaries, "test_evaluated": False})
    lines = ["# Downstream encoder scaling check (validation only)", "",
        "Same CPA-trained checkpoints and validation candidates; no retraining. Real context edges use actual size, "
        "m=1, or constant m=3. Synthetic singleton self-messages retain m=1 throughout. "
        "The whole encoder is recomputed; candidate readout and classifier weights are unchanged.", "",
        "| Dataset | Condition | Validation AP | Validation AUROC | AP change vs normal | Mean absolute logit change |",
        "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['condition']} | {row['auprc']['mean']:.4f} +/- {row['auprc']['seed_sd']:.4f} | "
            f"{row['auroc']['mean']:.4f} +/- {row['auroc']['seed_sd']:.4f} | "
            f"{row['delta_auprc']['mean']:+.4f} | {row['mean_absolute_logit_change']['mean']:.4f} |")
    lines += ["", "Normal CPA reproduces saved validation logits bit-for-bit in every seed. "
        "Hooks are removed after each intervention and ordinary inference is checked again afterward.", "",
        "Disabled scaling on CPA-trained weights is not the separately trained vanilla AST baseline. "
        "Prediction changes demonstrate computational dependence, not that the intervention is a better trained architecture. "
        "The normal checkpoints were selected on validation, so these diagnostics are not an independent generalization estimate. "
        "No condition is selected for further test evaluation."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cpa-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run(args.suite, args.cpa_root or Path("results/encoder-cpa") / args.suite.name,
              args.output or Path("results/encoder-signal-check") / args.suite.name))
