from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from .data import Candidates, Context, Protocol, pack
from .models import Predictor


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if math.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def metrics(labels, logits) -> dict:
    labels, logits = np.asarray(labels), np.asarray(logits)
    if not len(labels):
        return {"n": 0, "positives": 0, "auroc": None, "auprc": None}
    both = len(np.unique(labels)) == 2
    return {"n": len(labels), "positives": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, logits)) if both else None,
            "auprc": float(average_precision_score(labels, logits)) if both else None,
            "logit_std": float(logits.std()), "unique_logits": int(len(np.unique(logits))),
            "bce": float(np.mean(np.logaddexp(0, logits) - labels * logits))}


def evaluate_metrics(candidates: Candidates, logits: np.ndarray) -> dict:
    sizes = candidates.sizes
    by_size = {str(int(s)): {**metrics(candidates.labels[sizes == s], logits[sizes == s]),
                            "small_bin": int((sizes == s).sum()) < 40}
               for s in np.unique(sizes)}
    return {"overall": metrics(candidates.labels, logits), "by_size": by_size,
            "macro_size_auprc": float(np.mean([r["auprc"] for r in by_size.values()
                                                if r["auprc"] is not None]))}


def batches(candidates: Candidates, batch_size: int, device, order=None):
    if order is None:
        order = np.arange(len(candidates.edges))
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        yield ids, pack([candidates.edges[i] for i in ids], device), torch.as_tensor(
            candidates.labels[ids], device=device)


@torch.no_grad()
def predict(model: Predictor, context: Context, candidates: Candidates, batch_size: int,
            device, size_override=None, embeddings=None):
    model.eval()
    z = model.encode(context) if embeddings is None else embeddings
    outputs = []
    for _, packed, _ in batches(candidates, batch_size, device):
        outputs.append(model.decode(z, packed, size_override=size_override).cpu().numpy())
    return np.concatenate(outputs)


def train_epoch(model, context, candidates, optimizer, batch_size, device, rng):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z = model.encode(context)
    # Backpropagate accumulated decoder gradients through the encoder once.
    latent = z.detach().requires_grad_(True)
    total_loss = 0.
    for _, packed, labels in batches(candidates, batch_size, device,
                                     rng.permutation(len(candidates.edges))):
        logits = model.decode(latent, packed)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / len(candidates.edges)
        loss.backward()
        total_loss += loss.item()
    z.backward(latent.grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
    return total_loss


def fit(model_name: str, protocol: Protocol, context: Context, config: dict, seed: int,
        output: Path, device="cpu") -> tuple[Predictor, dict]:
    """Select checkpoints and learning rates using validation AP."""
    output.mkdir(parents=True, exist_ok=True)
    trials, winner, winner_info, winner_score = [], None, None, -float("inf")
    total_start = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for lr in config.get("learning_rates", [.001]):
        seed_everything(seed)
        model = Predictor(protocol.n_nodes, model_name, hidden_dim=config["hidden_dim"],
                          layers=config["layers"], heads=config["heads"], dropout=config["dropout"],
                          size_scale=float(np.log1p(protocol.train.sizes.max()))).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
        rng = np.random.default_rng(seed)
        history, best_state, best_score, best_epoch, stale = [], None, -float("inf"), 0, 0
        started = time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            loss = train_epoch(model, context, protocol.train, optimizer,
                               config["batch_size"], device, rng)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Non-finite training objective for {model_name}, lr={lr}")
            if epoch % config["eval_every"] and epoch != config["epochs"]:
                continue
            logits = predict(model, context, protocol.val, config["batch_size"], device)
            val = metrics(protocol.val.labels, logits)
            history.append({"epoch": epoch, "loss": loss, "val": val})
            score = val["auprc"]
            if score is not None and score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += config["eval_every"]
            if epoch % config.get("log_every", 25) == 0:
                print(f"  {model_name} lr={lr} epoch={epoch} loss={loss:.4f} val_AP={score:.4f}", flush=True)
            if stale >= config["patience"]:
                break
        if best_state is None:
            raise ValueError("Validation partition must contain both labels")
        trial = {"learning_rate": lr, "best_epoch": best_epoch, "validation_auprc": best_score,
                 "seconds": time.perf_counter() - started, "history": history}
        trials.append(trial)
        if best_score > winner_score:
            winner, winner_info, winner_score = best_state, copy.deepcopy(trial), best_score
    model.load_state_dict(winner)
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    info = {"selected_learning_rate": winner_info["learning_rate"],
            "best_epoch": winner_info["best_epoch"], "validation_auprc": winner_score,
            "training_seconds_all_trials": time.perf_counter() - total_start,
            "parameters": sum(p.numel() for p in model.parameters()), "trials": trials,
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                                           if str(device).startswith("cuda") else None)}
    torch.save({"state_dict": winner, "model_name": model_name, "n_nodes": protocol.n_nodes,
                "config": config, "size_scale": model.size_scale,
                "split_sha256": protocol.manifest["split_sha256"]}, output / "checkpoint.pt")
    write_json(output / "training.json", info)
    return model, info
