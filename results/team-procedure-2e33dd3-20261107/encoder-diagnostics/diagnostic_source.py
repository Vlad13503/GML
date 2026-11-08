"""Frozen validation-only encoder interventions on the verified current campaign."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

from current_results import ROOT, current_campaign, require_current_path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
EXPORT = ROOT / "tmp/team-procedure-2e33dd3"
sys.path.insert(0, str(EXPORT))
import numpy as np
import torch
from torch.nn import functional as F
import hnn_project
from hnn_project.data import Candidates, Protocol, make_context
from hnn_project.models import Predictor
from hnn_project.train import predict, evaluate_metrics, seed_everything, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(value, reference):
    return {"mean_norm": float(value.norm(dim=1).mean()),
            "mean_l2": float((value-reference).norm(dim=1).mean()),
            "mean_relative_l2": float(((value-reference).norm(dim=1) / reference.norm(dim=1).clamp_min(1e-8)).mean()),
            "mean_cosine": float(F.cosine_similarity(value, reference, dim=1, eps=1e-8).mean())}


def stages(pool, raw, sizes, multiplier, zero_query=False):
    scale = sizes if multiplier == "actual" else torch.full_like(sizes, multiplier)
    modified = raw * scale[:, None]
    pre_norm = modified + (0 if zero_query else pool.query.reshape(1, -1))
    post_norm = pool.norm1(pre_norm)
    representation = pool.norm2(post_norm + F.relu(pool.ff(post_norm)))
    return {"attention": raw, "modified": modified, "pre_norm": pre_norm,
            "post_norm": post_norm, "representation": representation}


@torch.no_grad()
def local_traces(model, context, folder):
    """Normal upstream inputs are fixed within each layer's local sweep."""
    rows = []
    x = model.phi(model.embedding.weight)
    inc = context.incidence
    real = inc.sizes > 1
    sizes = inc.sizes[real]
    for index, layer in enumerate(model.layers):
        pool = layer.v2e
        ordinary, traced = pool(x[inc.nodes], inc.groups, len(inc.sizes), inc.sizes, trace=True)
        raw = traced["attention"][real]
        actual = stages(pool, raw, sizes, "actual")
        # Verify reconstructed operation against the unmodified source forward.
        for key in ("modified", "post_norm", "representation"):
            torch.testing.assert_close(actual[key], traced[key][real], rtol=0, atol=0)
        np.savez_compressed(folder / f"layer-{index+1}-fixed-attention.npz",
            raw=raw.cpu().numpy(), query=pool.query.cpu().numpy(), sizes=sizes.cpu().numpy())
        for control in ("query_residual", "zero_query_residual"):
            reference = stages(pool, raw, sizes, 1, zero_query=control.startswith("zero"))
            for multiplier in (1, 2, 3, 4, 8, "actual"):
                changed = stages(pool, raw, sizes, multiplier, zero_query=control.startswith("zero"))
                for size in (2, 3):
                    mask = sizes == size
                    if not mask.any():
                        continue
                    for stage in changed:
                        rows.append({"layer": index+1, "control": control, "multiplier": str(multiplier),
                            "context_size": size, "count": int(mask.sum()), "stage": stage,
                            **distance(changed[stage][mask], reference[stage][mask])})
        # Next layer receives normal upstream computation, not an intervention.
        edge = F.relu(ordinary)
        degrees = torch.bincount(inc.nodes, minlength=context.n_nodes)
        x = F.relu(layer.e2v(edge[inc.groups], inc.nodes, context.n_nodes, degrees))
    torch.testing.assert_close(x, model.encode(context), rtol=0, atol=0)
    return rows


@torch.no_grad()
def encode_override(model, context, effective):
    handles = []
    def hook(module, args, kwargs):
        return args, {**kwargs, "size_override": effective}
    try:
        if effective is not None:
            for layer in model.layers:
                handles.append(layer.v2e.register_forward_pre_hook(hook, with_kwargs=True))
        return model.encode(context)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def main():
    campaign = current_campaign()
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    suite = require_current_path(ROOT / registry["comparison_suite"])
    assert suite.name == "db95c77a328c"  # Campaign-specific source and interpretation.
    assert Path(hnn_project.__file__).resolve().is_relative_to(EXPORT)
    provenance = json.loads((campaign / "run_plan.json").read_text())
    for relative, expected in provenance["source_files"].items():
        assert digest(EXPORT / relative) == expected
    environment = json.loads((suite / "environment.json").read_text())
    assert torch.__version__ == environment["torch"] and torch.cuda.is_available()
    torch.set_num_threads(4)
    out = campaign / "encoder-diagnostics"
    out.mkdir(exist_ok=True)
    config = json.loads((suite / "config.json").read_text())
    plan = {"status": "RUNNING", "suite": str(suite.relative_to(ROOT)),
        "procedure_commit": registry["procedure_commit"], "legacy_inputs": False,
        "models": ["allset_cpa_encoder", "allset_cpa_both"], "seeds": [0, 1, 2],
        "local_sweep": {"multipliers": [1, 2, 3, 4, 8, "actual"], "reference": "m=1 in same layer/edge/control",
                        "controls": ["query residual retained", "query residual removed; attention unchanged"],
                        "inputs": "normal upstream layer inputs held fixed within local sweep"},
        "downstream": {"conditions": ["normal", "disabled", "constant3", "shuffled0", "shuffled1", "shuffled2"],
            "evaluation": "validation only; no retraining or condition selection",
            "shuffle_seed": "10000 + 100*split_seed + replicate",
            "shuffle_scope": "real context edge sizes, same permutation in all layers",
            "synthetic_singletons": "m=1 throughout", "readout": "unchanged in each model"},
        "limitations": "interventions on frozen CPA-trained weights are not retrained architectures; validation was used for checkpoint selection",
        "runner_sha256": digest(Path(__file__))}
    write_json(out / "plan.json", plan)
    (out / "diagnostic_source.py").write_bytes(Path(__file__).read_bytes())
    traces, records, hashes = [], [], {}
    for dataset in config["datasets"]:
        for seed in plan["seeds"]:
            seed_everything(seed)
            split_dir = suite / dataset / "order-k3" / f"seed-{seed}"
            with gzip.open(split_dir / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            manifest = json.loads((split_dir / "manifest.json").read_text())
            split_hash = hashlib.sha256(json.dumps({"context": saved["context"],
                "candidates": [saved[k]["edges"] for k in ("train", "val", "test")]},
                separators=(",", ":")).encode()).hexdigest()
            assert split_hash == manifest["split_sha256"]
            val = Candidates([tuple(e) for e in saved["val"]["edges"]], np.array(saved["val"]["labels"], dtype=np.float32))
            empty = Candidates([], np.empty(0, dtype=np.float32))
            for model_name in plan["models"]:
                source = split_dir / model_name
                checkpoint_path = require_current_path(source / "checkpoint.pt")
                checkpoint_hash = digest(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                assert checkpoint["split_sha256"] == split_hash and checkpoint["model_name"] == model_name
                active = checkpoint["config"]
                model = Predictor(checkpoint["n_nodes"], model_name, active["hidden_dim"], active["layers"],
                                  active["heads"], active["dropout"], checkpoint["size_scale"]).to("cuda").eval()
                model.load_state_dict(checkpoint["state_dict"])
                assert active["dropout"] == 0  # Explicit local reconstruction uses eval ReLU path.
                protocol = Protocol(saved["context"], empty, val, empty, checkpoint["n_nodes"], manifest)
                context = make_context(protocol, "cuda")
                sizes = context.incidence.sizes
                real = sizes > 1
                assert set(sizes[real].cpu().tolist()) <= {2, 3}
                covered = torch.tensor(sorted({v for e in saved["context"] for v in e}), device="cuda")
                folder = out / dataset / f"seed-{seed}" / model_name
                folder.mkdir(parents=True, exist_ok=True)
                for row in local_traces(model, context, folder):
                    traces.append({"dataset": dataset, "seed": seed, "model": model_name, **row})
                baseline_z = baseline_logits = baseline_metrics = None
                results = {}
                effective_sizes = {"normal": None,
                    "disabled": torch.ones_like(sizes),
                    "constant3": torch.where(real, torch.full_like(sizes, 3), torch.ones_like(sizes))}
                for rep in range(3):
                    effective = sizes.clone()
                    permutation = np.random.default_rng(10000+100*seed+rep).permutation(int(real.sum()))
                    effective[real] = sizes[real][torch.tensor(permutation, device="cuda")]
                    np.testing.assert_array_equal(np.sort(effective[real].cpu()), np.sort(sizes[real].cpu()))
                    effective_sizes[f"shuffled{rep}"] = effective
                for condition, effective in effective_sizes.items():
                    z = encode_override(model, context, effective)
                    logits = predict(model, None, val, active["batch_size"], "cuda", embeddings=z)
                    metrics = evaluate_metrics(val, logits)
                    if condition == "normal":
                        baseline_z, baseline_logits, baseline_metrics = z.clone(), logits.copy(), metrics
                        with np.load(source / "val_predictions.npz") as expected:
                            for key, value in (("logits", logits), ("labels", val.labels), ("sizes", val.sizes)):
                                np.testing.assert_array_equal(value, expected[key])
                    result = {"evaluation": metrics, "embedding": distance(z[covered], baseline_z[covered]),
                        "mean_absolute_logit_change": float(np.abs(logits-baseline_logits).mean()),
                        "max_absolute_logit_change": float(np.abs(logits-baseline_logits).max()),
                        "delta_auprc": metrics["overall"]["auprc"]-baseline_metrics["overall"]["auprc"],
                        "delta_auroc": metrics["overall"]["auroc"]-baseline_metrics["overall"]["auroc"]}
                    results[condition] = result
                    np.savez_compressed(folder / f"{condition}-validation.npz", logits=logits, labels=val.labels,
                        sizes=val.sizes, covered_node_ids=covered.cpu().numpy(), covered_node_embeddings=z[covered].cpu().numpy(),
                        encoder_multipliers=(sizes if effective is None else effective).cpu().numpy())
                np.testing.assert_array_equal(predict(model, context, val, active["batch_size"], "cuda"), baseline_logits)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), checkpoint["state_dict"][key], rtol=0, atol=0)
                assert digest(checkpoint_path) == checkpoint_hash
                for path in (checkpoint_path, source / "val_predictions.npz", split_dir / "splits.json.gz", split_dir / "manifest.json"):
                    hashes[str(path.relative_to(ROOT))] = digest(path)
                record = {"dataset": dataset, "seed": seed, "model": model_name, "split_sha256": split_hash,
                    "checkpoint_sha256": checkpoint_hash, "normal_replay_exact": True, "restored_replay_exact": True,
                    "weights_unchanged": True, "test_evaluated": False, "conditions": results}
                records.append(record)
                write_json(folder / "result.json", record)
                print(dataset, seed, model_name, {k: round(v["delta_auprc"], 6) for k, v in results.items()}, flush=True)
    assert len(records) == 12
    with (out / "local_traces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0])); writer.writeheader(); writer.writerows(traces)
    write_json(out / "results.json", {"status": "VERIFIED_CURRENT", "test_evaluated": False,
        "records": records, "source_sha256": hashes, "local_trace_rows": len(traces)})
    plan["status"] = "VERIFIED_CURRENT"
    write_json(out / "plan.json", plan)
    registry["encoder_diagnostics"] = str(out.relative_to(ROOT))
    write_json(ROOT / "CURRENT_EXPERIMENT.json", registry)
    print("VERIFIED: 12 fresh checkpoints; exact normal/restored validation replay; frozen weights; no test scoring.")


if __name__ == "__main__":
    main()
