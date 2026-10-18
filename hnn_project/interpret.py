"""Cardinality probes and computational interventions on frozen representations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .data import Candidates, pack
from .train import batches, evaluate_metrics, predict, write_json


def _probe_metrics(y, predicted):
    return {"log_size_mae": float(mean_absolute_error(y, predicted)),
            "log_size_r2": float(r2_score(y, predicted)) if len(y) > 1 and np.ptp(y) > 0 else None}


def _sample_positives(part, limit, rng):
    positives = [e for e, y in zip(part.edges, part.labels) if y == 1]
    sizes = np.array(list(map(len, positives)))
    groups = np.unique(sizes)
    cap = max(1, limit // len(groups))
    selected = [i for s in groups for i in rng.choice(np.flatnonzero(sizes == s),
                min(cap, int((sizes == s).sum())), replace=False)]
    edges = [positives[i] for i in selected]
    return Candidates(edges, np.ones(len(edges), dtype=np.float32))


@torch.no_grad()
def collect_stages(model, z, part, batch_size, device):
    stages = {}
    for _, packed, _ in batches(part, batch_size, device):
        _, trace = model.decode(z, packed, trace=True)
        for key, values in trace.items():
            stages.setdefault(key, []).append(values.cpu().numpy())
    return {key: np.concatenate(values) for key, values in stages.items()}


@torch.no_grad()
def cardinality_probes(model, z, protocol, config, seed, device):
    rng = np.random.default_rng(seed + 1729)
    samples = [_sample_positives(p, config.get("probe_limit", 2000), rng)
               for p in (protocol.train, protocol.val, protocol.test)]
    targets = [np.log1p(p.sizes) for p in samples]
    if np.ptp(targets[0]) == 0:
        return {"skipped": "Training positives have only one cardinality; size is not identifiable from variation."}
    features = [collect_stages(model, z, p, config["batch_size"], device) for p in samples]
    results = {}
    for stage in features[0]:
        stage_results = {}
        for kind in ("linear", "norm_only", "permuted_target"):
            xs = [f[stage] for f in features]
            if kind == "norm_only":
                xs = [np.linalg.norm(x, axis=1, keepdims=True) for x in xs]
            y = rng.permutation(targets[0]) if kind == "permuted_target" else targets[0]
            # Fit only on training representations.
            probe = make_pipeline(StandardScaler(), Ridge(alpha=1.))
            probe.fit(xs[0], y)
            stage_results[kind] = {name: _probe_metrics(t, probe.predict(x))
                                  for name, x, t in zip(("train", "val", "test"), xs, targets)}
        results[stage] = stage_results
    return {"target": "log(1 + cardinality)", "sampling": "capped equally per available size; positive edges only",
            "n": {n: len(s.edges) for n, s in zip(("train", "val", "test"), samples)},
            "constant_train_mean_baseline": {n: _probe_metrics(t, np.full_like(t, targets[0].mean()))
                for n, t in zip(("val", "test"), targets[1:])},
            "stages": results,
            "caveat": "Real-data probes can exploit composition correlated with size; replication is the controlled check."}


@torch.no_grad()
def replication_intervention(model, z, protocol, seed, device, limit=32):
    rng = np.random.default_rng(seed + 2718)
    known = np.array(sorted({v for e in protocol.context for v in e}))
    records = []
    for base_id in range(limit):
        base = tuple(rng.choice(known, size=min(3, len(known)), replace=False).tolist())
        original, baseline_logit = None, None
        for factor in (1, 2, 4, 8):
            candidate = pack([base * factor], device)
            logits, stages = model.decode(z, candidate, trace=True)
            if original is None:
                original = {k: v.clone() for k, v in stages.items()}
                baseline_logit = logits.item()
            for stage, value in stages.items():
                reference = original[stage]
                records.append({"base_id": base_id, "factor": factor, "stage": stage,
                    "size": len(base) * factor, "norm": value.norm().item(),
                    "l2_from_base": (value - reference).norm().item(),
                    "relative_l2_from_base": ((value - reference).norm() / reference.norm().clamp_min(1e-8)).item(),
                    "logit_change": logits.item() - baseline_logit})
    return {"scope": "Synthetic multiset intervention on frozen node representations; not task accuracy or causal social effects.",
            "records": records}


@torch.no_grad()
def size_intervention(model, z, protocol, config, device):
    reference = float(np.median(protocol.train.sizes))
    original = predict(model, None, protocol.test, config["batch_size"], device, embeddings=z)
    neutralized = predict(model, None, protocol.test, config["batch_size"], device,
                          size_override=reference, embeddings=z)
    by_size = {str(int(s)): {"n": int((protocol.test.sizes == s).sum()),
        "mean_absolute_logit_change": float(np.mean(np.abs(original[protocol.test.sizes == s]
                                                              - neutralized[protocol.test.sizes == s])))}
               for s in np.unique(protocol.test.sizes)}
    return {"reference_size_train_median": reference,
            "mean_absolute_logit_change": float(np.mean(np.abs(original - neutralized))),
            "original": evaluate_metrics(protocol.test, original),
            "neutralized": evaluate_metrics(protocol.test, neutralized), "by_size": by_size,
            "scope": "Only decoder size input/scaling is clamped; members, attention, and encoder are frozen. "
                     "This diagnoses computational dependence, not an intervention on real interaction size."}


def analyze(model, context, protocol, config, seed, output: Path, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(context)
        results = {"probes": cardinality_probes(model, z, protocol, config, seed, device),
                   "replication": replication_intervention(model, z, protocol, seed, device),
                   "size_intervention": size_intervention(model, z, protocol, config, device)}
    write_json(output / "interpretability.json", results)
    return results
