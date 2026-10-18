"""AllSet-style PMA (Chien et al., 2022), adapted for hyperedge prediction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .data import Context, PackedSets

CORE_MODELS = ["deepsets", "gcn", "allset", "allset_cpa", "allset_size"]
MODELS = CORE_MODELS + ["allset_cpa_encoder", "allset_cpa_both", "allset_additive"]


def segment_sum(values: torch.Tensor, groups: torch.Tensor, count: int) -> torch.Tensor:
    return values.new_zeros((count, *values.shape[1:])).index_add_(0, groups, values)


def segment_softmax(scores: torch.Tensor, groups: torch.Tensor, count: int) -> torch.Tensor:
    indices = groups[:, None].expand_as(scores)
    maxima = scores.new_full((count, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
    exponent = (scores - maxima[groups]).exp()
    return exponent / segment_sum(exponent, groups, count)[groups].clamp_min(1e-12)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int, mode="none"):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.dim, self.heads, self.mode = dim, heads, mode
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.empty(heads, dim // heads))
        nn.init.xavier_uniform_(self.query)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        if mode == "additive":
            self.add_weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, groups: torch.Tensor, count: int,
                sizes: torch.Tensor, size_override=None, trace=False):
        keys = self.key(x).view(-1, self.heads, self.dim // self.heads)
        values = self.value(x).view_as(keys)
        scores = F.leaky_relu((keys * self.query).sum(-1), .2)
        alpha = segment_softmax(scores, groups, count)
        raw = segment_sum(values * alpha[..., None], groups, count).reshape(count, self.dim)
        effective = sizes.to(raw.dtype) if size_override is None else torch.as_tensor(
            size_override, dtype=raw.dtype, device=raw.device).expand(count)
        modified = raw
        if self.mode == "scaled":
            modified = raw * effective[:, None]
        elif self.mode == "additive":
            summed = segment_sum(values.reshape(-1, self.dim), groups, count)
            modified = raw + self.add_weight * summed * (effective / sizes.clamp_min(1))[:, None]
        post_norm = self.norm1(modified + self.query.reshape(1, -1))
        representation = self.norm2(post_norm + F.relu(self.ff(post_norm)))
        if trace:
            return representation, {"attention": raw, "modified": modified,
                                    "post_norm": post_norm, "representation": representation}
        return representation


class HypergraphLayer(nn.Module):
    def __init__(self, dim, heads, dropout, cpa=False):
        super().__init__()
        self.v2e = AttentionPool(dim, heads, "scaled" if cpa else "none")
        self.e2v = AttentionPool(dim, heads)
        self.dropout = dropout

    def forward(self, x, context: Context):
        incidence = context.incidence
        edge = self.v2e(x[incidence.nodes], incidence.groups, len(incidence.sizes), incidence.sizes)
        edge = F.dropout(F.relu(edge), self.dropout, self.training)
        degrees = torch.bincount(incidence.nodes, minlength=context.n_nodes)
        node = self.e2v(edge[incidence.groups], incidence.nodes, context.n_nodes, degrees)
        return F.dropout(F.relu(node), self.dropout, self.training)


class Predictor(nn.Module):
    def __init__(self, n_nodes: int, name: str, hidden_dim=64, layers=2, heads=4,
                 dropout=.2, size_scale=1.):
        super().__init__()
        if name not in MODELS:
            raise ValueError(f"Unknown model {name}")
        self.name, self.dropout, self.size_scale = name, dropout, size_scale
        self.embedding = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=.1)
        self.phi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        if name == "gcn":
            self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        elif name.startswith("allset"):
            self.layers = nn.ModuleList(HypergraphLayer(hidden_dim, heads, dropout,
                cpa=name in ("allset_cpa_encoder", "allset_cpa_both")) for _ in range(layers))
        else:
            self.layers = nn.ModuleList()
        mode = ("scaled" if name in ("allset_cpa", "allset_cpa_both") else
                "additive" if name == "allset_additive" else "none")
        self.pool = None if name == "deepsets" else AttentionPool(hidden_dim, heads, mode)
        # Controls receive zero in the size channel to match classifier parameters.
        self.scorer = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def encode(self, context: Context):
        x = self.phi(self.embedding.weight)
        for layer in self.layers:
            if self.name == "gcn":
                if context.adjacency is None:
                    raise ValueError("GCN requires a clique-expanded context")
                x = F.dropout(F.relu(layer(torch.sparse.mm(context.adjacency, x))),
                              self.dropout, self.training)
            else:
                x = layer(x, context)
        return x

    def decode(self, z: torch.Tensor, candidates: PackedSets, size_override=None, trace=False):
        if self.pool is None:
            representation = segment_sum(z[candidates.nodes], candidates.groups, len(candidates.sizes))
            stages = {"representation": representation}
        else:
            result = self.pool(z[candidates.nodes], candidates.groups, len(candidates.sizes),
                               candidates.sizes, size_override=size_override, trace=trace)
            representation, stages = result if trace else (result, None)
        size = candidates.sizes.to(z.dtype) if size_override is None else torch.as_tensor(
            size_override, dtype=z.dtype, device=z.device).expand(len(candidates.sizes))
        feature = (size.log1p() / self.size_scale if self.name == "allset_size"
                   else torch.zeros_like(size))
        logits = self.scorer(torch.cat((representation, feature[:, None]), -1)).squeeze(-1)
        if trace:
            stages["decoder_input"] = torch.cat((representation, feature[:, None]), -1)
            return logits, stages
        return logits
