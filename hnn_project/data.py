from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import ssl
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import torch

DATASETS = ("email-Enron", "tags-ask-ubuntu")
DOWNLOAD_IDS = {"email-Enron": "1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77",
                "tags-ask-ubuntu": "1tb1ZJlXEJnlRkXpTuBZlOqqsFknWCkUV"}
Edge = tuple[int, ...]


def download(name: str, root: Path) -> Path:
    if name not in DATASETS:
        raise ValueError(f"Choose a dataset from {DATASETS}")
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{name}.tar.gz"
    if not archive.exists():
        url = f"https://drive.google.com/uc?export=download&id={DOWNLOAD_IDS[name]}"
        temporary = archive.with_suffix(".partial")
        try:
            context = ssl.create_default_context()
            try:
                import truststore
                context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                pass
            with urllib.request.urlopen(url, timeout=120, context=context) as source, temporary.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if not tarfile.is_tarfile(archive):
        raise ValueError(f"{archive} is not a tar archive. Download the public dataset archive "
                         "from Benson's dataset page and replace this file.")
    with tarfile.open(archive, "r:gz") as tar:
        for suffix in ("nverts", "simplices"):
            dest = folder / f"{name}-{suffix}.txt"
            if dest.exists():
                continue
            members = [m for m in tar.getmembers() if m.isfile()
                       and Path(m.name).name == dest.name]
            if len(members) != 1:
                raise ValueError(f"Archive must contain exactly one {dest.name}")
            with tar.extractfile(members[0]) as source, dest.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
    return folder


@dataclass
class Dataset:
    name: str
    edges: list[Edge]
    n_nodes: int
    audit: dict


def load_dataset(name: str, root: Path) -> Dataset:
    folder = root / name
    paths = [folder / f"{name}-{suffix}.txt" for suffix in ("nverts", "simplices")]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"Run: python -m hnn_project download --datasets {name}")
    sizes, vertices = [np.loadtxt(p, dtype=np.int64, ndmin=1) for p in paths]
    if (sizes < 1).any() or int(sizes.sum()) != len(vertices):
        raise ValueError("Malformed dataset: simplex lengths do not match vertices")
    node_ids, vertices = np.unique(vertices, return_inverse=True)
    offsets = np.r_[0, sizes.cumsum()]
    unique = set()
    for i in range(len(sizes)):
        edge = tuple(sorted(set(vertices[offsets[i]:offsets[i + 1]].tolist())))
        if len(edge) >= 2:
            unique.add(edge)
    edges = sorted(unique)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    audit = {
        "name": name, "raw_interactions": len(sizes), "n_nodes": len(node_ids),
        "unique_non_singleton_interactions": len(edges),
        "raw_cardinalities": dict(sorted(Counter(map(int, sizes)).items())),
        "unique_cardinalities": dict(sorted(Counter(map(len, edges)).items())),
        "source_sha256": digest.hexdigest(),
        "note": "Exact sets deduplicated; singleton interactions discarded; timestamps not used.",
    }
    return Dataset(name, edges, len(node_ids), audit)


def synthetic_dataset(seed: int = 0) -> Dataset:
    """Synthetic fixture for software validation."""
    rng = np.random.default_rng(seed)
    edges = set()
    for size in range(2, 9):
        for _ in range(100):
            group = int(rng.integers(3))
            pool = np.arange(group * 12, (group + 1) * 12)
            edges.add(tuple(sorted(rng.choice(pool, size, replace=False).tolist())))
    return Dataset("synthetic", sorted(edges), 36, {"purpose": "software validation only"})


@dataclass
class Candidates:
    edges: list[Edge]
    labels: np.ndarray

    @property
    def sizes(self) -> np.ndarray:
        return np.array(list(map(len, self.edges)), dtype=np.int64)


@dataclass
class Protocol:
    context: list[Edge]
    train: Candidates
    val: Candidates
    test: Candidates
    n_nodes: int
    manifest: dict


def _partition(edges: list[Edge], rng, fraction: float) -> tuple[list[Edge], list[Edge]]:
    if len(edges) < 2:
        raise ValueError("Insufficient distinct edges to split")
    ids = rng.permutation(len(edges))
    cut = min(len(edges) - 1, max(1, round(len(edges) * fraction)))
    return [edges[i] for i in ids[:cut]], [edges[i] for i in ids[cut:]]


def sample_negatives(positives: list[Edge], universe: np.ndarray, forbidden: set[Edge],
                     rng, ratio: int, hard_fraction: float) -> Candidates:
    """Unobserved sets are assumed negative; all observed positives are excluded."""
    negatives = []
    used: set[Edge] = set()
    for edge in positives:
        if len(edge) > len(universe):
            raise ValueError("Candidate exceeds node universe")
        for _ in range(ratio):
            for attempt in range(5000):
                if attempt < 100 and rng.random() < hard_fraction:
                    candidate = list(edge)
                    candidate[int(rng.integers(len(edge)))] = int(rng.choice(universe))
                    candidate = tuple(sorted(set(candidate)))
                else:
                    candidate = tuple(sorted(rng.choice(universe, len(edge), replace=False).tolist()))
                if len(candidate) == len(edge) and candidate not in forbidden and candidate not in used:
                    used.add(candidate)
                    negatives.append(candidate)
                    break
            else:
                raise ValueError(f"Negative space exhausted at size {len(edge)}; lower negative_ratio")
    return Candidates(positives + negatives,
                      np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(np.float32))


def make_protocol(dataset: Dataset, config: dict, seed: int) -> Protocol:
    rng = np.random.default_rng(seed)
    mode, k = config["protocol"], config.get("k", 3)
    edges = dataset.edges
    if mode == "random":
        train_pool, heldout = _partition(edges, rng, .70)
        val, test = _partition(heldout, rng, .50)
    elif mode == "order":
        train_pool, val = _partition([e for e in edges if len(e) <= k], rng, .80)
        test = [e for e in edges if len(e) > k]
    elif mode == "interpolation":
        low, high = config["interpolation_bounds"]
        if high <= low + 1:
            raise ValueError("Interpolation needs at least one strictly interior cardinality")
        train_pool, val = _partition([e for e in edges if len(e) <= low or len(e) >= high], rng, .80)
        test = [e for e in edges if low < len(e) < high]
    else:
        raise ValueError(f"Unknown protocol {mode}")
    context, train = _partition(train_pool, rng, config.get("context_fraction", .60))
    known = set(v for e in context for v in e)
    dropped = {}
    # Restrict node novelty to isolate interaction-order shift.
    filtered = []
    for name, targets in (("train", train), ("val", val), ("test", test)):
        kept = [e for e in targets if set(e) <= known]
        dropped[name] = len(targets) - len(kept)
        if len(kept) < config.get("min_partition_positives", 10):
            raise ValueError(f"Only {len(kept)} {name} positives after node-coverage filtering. "
                             "Inspect audit, adjust k/context_fraction, or use another dataset.")
        filtered.append(kept)
    forbidden = set(edges)  # Exclusion oracle only; no held-out structural features are constructed.
    candidates = []
    for positives in filtered:
        part = sample_negatives(positives, np.array(sorted(known)), forbidden, rng,
                                config.get("negative_ratio", 1), config.get("hard_negative_fraction", .5))
        forbidden.update(e for e, y in zip(part.edges, part.labels) if y == 0)
        candidates.append(part)
    split_hash = hashlib.sha256(json.dumps({
        "context": context,
        "candidates": [c.edges for c in candidates],
    }, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "dataset": dataset.name, "protocol": mode, "k": k, "seed": seed,
        "context_fraction": config.get("context_fraction", .6),
        "context_edges": len(context), "context_nodes": len(known),
        "dropped_cold_node_positives": dropped,
        "split_sha256": split_hash, "source": dataset.audit,
        "partitions": {n: {"positives": int(c.labels.sum()), "negatives": int((c.labels == 0).sum()),
                            "positive_cardinalities": dict(sorted(Counter(len(e) for e, y in
                                zip(c.edges, c.labels) if y == 1).items()))}
                       for n, c in zip(("train", "val", "test"), candidates)},
        "scope": "Static, closed-node hyperedge completion; not temporal forecasting or unseen-node induction.",
    }
    return Protocol(context, *candidates, dataset.n_nodes, manifest)


@dataclass
class PackedSets:
    nodes: torch.Tensor
    groups: torch.Tensor
    sizes: torch.Tensor


def pack(edges: list[Edge], device="cpu") -> PackedSets:
    sizes = torch.tensor(list(map(len, edges)), dtype=torch.long, device=device)
    nodes = torch.tensor([v for e in edges for v in e], dtype=torch.long, device=device)
    groups = torch.repeat_interleave(torch.arange(len(edges), device=device), sizes)
    return PackedSets(nodes, groups, sizes)


@dataclass
class Context:
    incidence: PackedSets
    adjacency: torch.Tensor | None
    n_nodes: int


def make_context(protocol: Protocol, device="cpu", clique: bool = False) -> Context:
    # Singleton hyperedges provide self-messages, not prediction targets.
    incidence = pack(protocol.context + [(i,) for i in range(protocol.n_nodes)], device)
    adjacency = None
    if clique:
        pairs = {(v, v) for v in range(protocol.n_nodes)}
        for edge in protocol.context:
            pairs.update((u, v) for u in edge for v in edge if u != v)
        indices = torch.tensor(sorted(pairs), dtype=torch.long, device=device).T
        values = torch.ones(indices.shape[1], device=device)
        degree = torch.zeros(protocol.n_nodes, device=device).index_add_(0, indices[0], values)
        values = (degree[indices[0]] * degree[indices[1]]).rsqrt()
        adjacency = torch.sparse_coo_tensor(indices, values,
                                            (protocol.n_nodes, protocol.n_nodes)).coalesce()
    return Context(incidence, adjacency, protocol.n_nodes)
