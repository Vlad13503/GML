"""Verify only the fresh pinned campaign; superseded outputs are never inputs."""
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tmp/team-procedure-2e33dd3"
CAMPAIGN = ROOT / "results/team-procedure-2e33dd3-20261107"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(EXPORT))

import numpy as np
import sklearn
import torch
import hnn_project
from hnn_project.data import Candidates, Protocol, load_dataset, make_context
from hnn_project.models import Predictor
from hnn_project.train import evaluate_metrics, metrics, predict, seed_everything, write_json


def main():
    assert Path(hnn_project.__file__).resolve().is_relative_to(EXPORT)
    plan = json.loads((CAMPAIGN / "run_plan.json").read_text())
    for relative, expected in plan["source_files"].items():
        assert hashlib.sha256((EXPORT / relative).read_bytes()).hexdigest() == expected
    for relative, expected in plan["data_files"].items():
        assert hashlib.sha256((ROOT / "data" / relative).read_bytes()).hexdigest() == expected
    current = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    assert current["status"] == "fits_complete_verification_pending"
    suite = ROOT / current["comparison_suite"]
    assert suite.resolve().is_relative_to(CAMPAIGN)
    tuning = ROOT / current["tuning_root"]
    environment = json.loads((suite / "environment.json").read_text())
    assert environment["torch"] == torch.__version__
    assert environment["numpy"] == np.__version__ and environment["sklearn"] == sklearn.__version__
    assert environment["device"] == "cuda"
    code_digest = hashlib.sha256()
    for path in sorted((EXPORT / "hnn_project").glob("*.py")):
        code_digest.update(path.name.encode())
        code_digest.update(path.read_bytes())
    assert code_digest.hexdigest() == environment["code_sha256"]
    torch.set_num_threads(4)
    config = json.loads((suite / "config.json").read_text())
    assert config["models"] == ["allset", "allset_cpa", "allset_cpa_encoder", "allset_cpa_both"]
    assert config["seeds"] == [0, 1, 2] and config["learning_rates"] == [.001, .01]
    assert config["experiments"] == [{"protocol": "order", "k": 3}]
    assert config["dataset_overrides"] == json.loads((tuning / "best_settings.json").read_text())["dataset_overrides"]
    protocols, contexts = {}, {}
    split_checks = []
    for name in config["datasets"]:
        dataset = load_dataset(name, ROOT / "data")
        observed = set(dataset.edges)
        for seed in config["seeds"]:
            folder = suite / name / "order-k3" / f"seed-{seed}"
            manifest = json.loads((folder / "manifest.json").read_text())
            with gzip.open(folder / "splits.json.gz", "rt") as stream:
                saved = json.load(stream)
            parts = {key: Candidates([tuple(e) for e in saved[key]["edges"]],
                                    np.array(saved[key]["labels"], dtype=np.float32))
                     for key in ("train", "val", "test")}
            digest = hashlib.sha256(json.dumps({"context": saved["context"],
                "candidates": [parts[k].edges for k in ("train", "val", "test")]},
                separators=(",", ":")).encode()).hexdigest()
            assert digest == manifest["split_sha256"]
            context_nodes = {v for e in saved["context"] for v in e}
            collections = [set(map(tuple, saved["context"]))]
            assert collections[0] <= observed and all(len(e) <= 3 for e in collections[0])
            for key, part in parts.items():
                positives = {e for e, y in zip(part.edges, part.labels) if y == 1}
                negatives = {e for e, y in zip(part.edges, part.labels) if y == 0}
                assert positives <= observed and not negatives & observed
                assert len(negatives) == len(part.edges) - len(positives)
                assert all(set(e) <= context_nodes for e in part.edges)
                assert all((len(e) > 3) == (key == "test") for e in positives)
                for size in np.unique(part.sizes):
                    labels = part.labels[part.sizes == size]
                    assert (labels == 1).sum() == (labels == 0).sum()
                collections.append(set(part.edges))
            assert all(not collections[i] & collections[j] for i in range(4) for j in range(i))
            protocols[(name, seed)] = Protocol(saved["context"], parts["train"], parts["val"],
                                             parts["test"], dataset.n_nodes, manifest)
            split_checks.append({"dataset": name, "seed": seed, "split_sha256": digest})
    fit_paths = sorted(tuning.glob("*/order-k3/setting-*/seed-0/*/checkpoint.pt"))
    test_paths = sorted(suite.glob("*/order-k3/seed-*/*/checkpoint.pt"))
    assert len(fit_paths) == 80 and len(test_paths) == 24
    replay = []
    for index, checkpoint_path in enumerate(fit_paths + test_paths):
        is_comparison = index >= len(fit_paths)
        folder = checkpoint_path.parent
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        active = checkpoint["config"]
        model_name = checkpoint["model_name"]
        name = folder.relative_to(suite if is_comparison else tuning).parts[0]
        seed = int(folder.parent.name.removeprefix("seed-"))
        protocol = protocols[(name, seed)]
        assert checkpoint["split_sha256"] == protocol.manifest["split_sha256"]
        training = json.loads((folder / "training.json").read_text())
        assert training["never_improved"] == (training["best_epoch"] == 0)
        assert len(training["trials"]) == 2
        assert all(t["history"][0]["epoch"] == 0 and t["history"][0]["loss"] is None for t in training["trials"])
        initial_scores = [t["history"][0]["val"]["auprc"] for t in training["trials"]]
        # CUDA sparse GCN aggregation exhibits sub-micro AP variation even with
        # deterministic algorithms enabled. Preserve the observed discrepancy;
        # AST/CPA and all saved comparison predictions still require exact replay.
        initial_tolerance = 1e-6 if model_name == "gcn" else 0
        assert abs(initial_scores[0] - initial_scores[1]) <= initial_tolerance
        assert training["validation_auprc"] == max(t["validation_auprc"] for t in training["trials"])
        context_key = (name, seed, model_name == "gcn")
        if context_key not in contexts:
            contexts[context_key] = make_context(protocol, "cuda", clique=model_name == "gcn")
        context = contexts[context_key]
        seed_everything(seed)
        model = Predictor(protocol.n_nodes, model_name, active["hidden_dim"], active["layers"],
                          active["heads"], active["dropout"], checkpoint["size_scale"]).to("cuda")
        model.load_state_dict(checkpoint["state_dict"])
        if training["best_epoch"] == 0:
            seed_everything(seed)
            initial = Predictor(protocol.n_nodes, model_name, active["hidden_dim"], active["layers"],
                                active["heads"], active["dropout"], checkpoint["size_scale"])
            for key, value in checkpoint["state_dict"].items():
                torch.testing.assert_close(value.cpu(), initial.state_dict()[key], rtol=0, atol=0)
        val_logits = predict(model, context, protocol.val, active["batch_size"], "cuda")
        val_score = metrics(protocol.val.labels, val_logits)["auprc"]
        replay_error = abs(val_score - training["validation_auprc"])
        assert replay_error <= initial_tolerance, (checkpoint_path, val_score, training["validation_auprc"])
        if is_comparison:
            result = json.loads((folder / "result.json").read_text())
            assert result["suite"] == suite.name
            assert result["split_sha256"] == protocol.manifest["split_sha256"]
            expected = config["dataset_overrides"][name]
            assert all(active[k] == v for k, v in expected.items())
            for split, logits in (("val", val_logits), ("test", None)):
                part = getattr(protocol, split)
                if logits is None:
                    logits = predict(model, context, part, active["batch_size"], "cuda")
                with np.load(folder / f"{split}_predictions.npz") as arrays:
                    np.testing.assert_array_equal(arrays["labels"], part.labels)
                    np.testing.assert_array_equal(arrays["sizes"], part.sizes)
                    np.testing.assert_array_equal(arrays["logits"], logits)
                assert evaluate_metrics(part, logits) == result["evaluation"][split]
        replay.append({"checkpoint": str(checkpoint_path.relative_to(CAMPAIGN)),
                       "model": model_name, "dataset": name, "seed": seed,
                       "selected_learning_rate": training["selected_learning_rate"],
                       "best_epoch": training["best_epoch"], "validation_replay_exact": replay_error == 0,
                       "validation_replay_absolute_error": replay_error,
                       "validation_replay_tolerance": initial_tolerance,
                       "initial_lr_trial_ap_difference": abs(initial_scores[0] - initial_scores[1]),
                       "test_logits_replayed": is_comparison})
        if (index + 1) % 10 == 0:
            print(f"Replayed {index+1}/104 selected checkpoints", flush=True)
    tuning_config = json.loads((tuning / "tuning_config.json").read_text())
    grid = tuning_config["grid"]
    settings = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    with (tuning / "tuning.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 80
    selected = {}
    for name in config["datasets"]:
        scores = {i: float(np.mean([float(r["validation_auprc"]) for r in rows
                   if r["dataset"] == name and int(r["setting"]) == i])) for i in range(len(settings))}
        chosen = max(scores, key=lambda i: (scores[i], -i))
        assert settings[chosen] == config["dataset_overrides"][name]
        selected[name] = {"grid_index": chosen, "settings": settings[chosen], "mean_validation_ap": scores[chosen]}
    initialization_checks = []
    for name in config["datasets"]:
        for seed in config["seeds"]:
            active = {**config, **config["dataset_overrides"][name]}
            reference = None
            counts = []
            for model_name in config["models"]:
                seed_everything(seed)
                model = Predictor(protocols[(name, seed)].n_nodes, model_name, active["hidden_dim"],
                                  active["layers"], active["heads"], active["dropout"])
                if reference is None:
                    reference = model.state_dict()
                else:
                    assert set(reference) == set(model.state_dict())
                    for key, value in reference.items():
                        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                counts.append(sum(p.numel() for p in model.parameters()))
            assert len(set(counts)) == 1
            initialization_checks.append({"dataset": name, "seed": seed, "parameters": counts[0], "exactly_matched": True})
    verification = {"status": "VERIFIED_CURRENT", "procedure_commit": plan["commit"],
        "source_and_data_hashes_verified": True, "tuning_selected_checkpoints_replayed": 80,
        "comparison_checkpoints_replayed": 24, "comparison_val_test_logits": "bitwise identical",
        "splits": split_checks, "shared_tuning_selection_verified": selected,
        "initialization_checks": initialization_checks, "replay": replay,
        "legacy_results_used": False, "fresh_encoder_diagnostics_run": False, "automatic_decoder_diagnostics_replayed": False}
    write_json(CAMPAIGN / "verification.json", verification)
    (CAMPAIGN / "verification_source.py").write_bytes(Path(__file__).read_bytes())
    current.update(status="verified_current", verification=str((CAMPAIGN / "verification.json").relative_to(ROOT)))
    write_json(ROOT / "CURRENT_EXPERIMENT.json", current)
    write_json(CAMPAIGN / "status.json", current)
    print("VERIFIED: fresh tuning selection, 104 checkpoints, 6 splits, and all 24 validation/test result files.")


if __name__ == "__main__":
    main()
