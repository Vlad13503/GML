import copy
from collections import Counter
from pathlib import Path
import unittest
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from hnn_project.cli import DEFAULTS, validate
from hnn_project.data import (Candidates, load_dataset, make_context, make_protocol, pack,
                              sample_negatives, synthetic_dataset)
from hnn_project.models import AttentionPool, Predictor, segment_softmax
from hnn_project.train import metrics, seed_everything, train_epoch


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset = synthetic_dataset()
        cls.config = {**DEFAULTS, "protocol": "order", "k": 3}

    def setUp(self):
        seed_everything(11)

    def test_softmax_matches_dense_reference_and_gradients(self):
        scores = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1, 1, 3, 3, 3])
        output = segment_softmax(scores, groups, 5)
        expected = torch.cat([scores[groups == i].softmax(0) for i in (0, 1, 3)])
        torch.testing.assert_close(output, expected)
        weights = torch.randn_like(output)
        grad1 = torch.autograd.grad((output * weights).sum(), scores, retain_graph=True)[0]
        grad2 = torch.autograd.grad((expected * weights).sum(), scores)[0]
        torch.testing.assert_close(grad1, grad2)

    def test_attention_pool_matches_dense_formula(self):
        pool = AttentionPool(8, 2).double()
        x = torch.randn(5, 8, dtype=torch.float64)
        groups = torch.tensor([0, 0, 1, 1, 1])
        actual = pool(x, groups, 2, torch.tensor([2, 3]))
        expected = []
        for members in (x[:2], x[2:]):
            keys = pool.key(members).view(-1, 2, 4)
            values = pool.value(members).view(-1, 2, 4)
            weights = F.leaky_relu((keys * pool.query).sum(-1), .2).softmax(0)
            raw = (weights[..., None] * values).sum(0).reshape(1, -1)
            hidden = pool.norm1(raw + pool.query.reshape(1, -1))
            expected.append(pool.norm2(hidden + F.relu(pool.ff(hidden))))
        torch.testing.assert_close(actual, torch.cat(expected))

    def test_replication_invariance_and_cpa_scaling(self):
        pool = AttentionPool(8, 2)
        x = torch.randn(3, 8)
        _, original = pool(x, torch.zeros(3, dtype=torch.long), 1, torch.tensor([3]), trace=True)
        _, repeated = pool(x.repeat(4, 1), torch.zeros(12, dtype=torch.long), 1,
                           torch.tensor([12]), trace=True)
        for stage in original:
            torch.testing.assert_close(original[stage], repeated[stage], atol=1e-6, rtol=1e-5)
        pool.mode = "scaled"
        _, original = pool(x, torch.zeros(3, dtype=torch.long), 1, torch.tensor([3]), trace=True)
        _, repeated = pool(x.repeat(4, 1), torch.zeros(12, dtype=torch.long), 1,
                           torch.tensor([12]), trace=True)
        torch.testing.assert_close(repeated["attention"], original["attention"])
        torch.testing.assert_close(repeated["modified"], original["modified"] * 4)

    def test_candidates_permutation_invariant(self):
        p = make_protocol(self.dataset, self.config, 0)
        context = make_context(p, clique=True)
        edges = [(0, 2, 3), (4, 8), (9, 10, 11, 12)]
        for name in ("deepsets", "gcn", "allset", "allset_cpa", "allset_size", "allset_additive"):
            model = Predictor(p.n_nodes, name, hidden_dim=8, heads=2, layers=1).eval()
            with torch.no_grad():
                z = model.encode(context)
                a = model.decode(z, pack(edges))
                b = model.decode(z, pack([tuple(reversed(e)) for e in reversed(edges)]))
            torch.testing.assert_close(a, b.flip(0), atol=1e-6, rtol=1e-5)

    def test_splits_and_negative_exclusion(self):
        p = make_protocol(self.dataset, self.config, 2)
        positives = [{e for e, y in zip(c.edges, c.labels) if y == 1} for c in (p.train, p.val, p.test)]
        all_parts = [set(p.context), *positives]
        for i, first in enumerate(all_parts):
            for second in all_parts[i + 1:]:
                self.assertFalse(first & second)
        seen_negatives = set()
        for part in (p.train, p.val, p.test):
            negatives = {e for e, y in zip(part.edges, part.labels) if y == 0}
            self.assertFalse(negatives & set(self.dataset.edges))
            self.assertFalse(negatives & seen_negatives)
            seen_negatives.update(negatives)
            positive_sizes = Counter(len(e) for e, y in zip(part.edges, part.labels) if y == 1)
            negative_sizes = Counter(map(len, negatives))
            self.assertEqual(positive_sizes, negative_sizes)
        self.assertTrue(all(len(e) <= 3 for e in p.context))
        self.assertTrue(all(len(e) > 3 for e in positives[-1]))
        again = make_protocol(self.dataset, self.config, 2)
        self.assertEqual(p.manifest["split_sha256"], again.manifest["split_sha256"])

    def test_interpolation_has_no_boundary_overlap(self):
        p = make_protocol(self.dataset, {**self.config, "protocol": "interpolation",
                                         "interpolation_bounds": [2, 5]}, 0)
        self.assertTrue(all(len(e) <= 2 or len(e) >= 5 for e in p.context))
        self.assertTrue(all(2 < len(e) < 5 for e, y in zip(p.test.edges, p.test.labels) if y))

    def test_clique_contains_only_context_pairs(self):
        p = make_protocol(self.dataset, self.config, 0)
        adjacency = make_context(p, clique=True).adjacency
        expected = {(i, i) for i in range(p.n_nodes)}
        expected.update((u, v) for e in p.context for u in e for v in e if u != v)
        self.assertEqual(set(map(tuple, adjacency.indices().T.tolist())), expected)
        dense = adjacency.to_dense()
        torch.testing.assert_close(dense, dense.T)

    def test_batched_encoder_gradient_matches_full_objective(self):
        p = make_protocol(self.dataset, self.config, 0)
        context = make_context(p, clique=True)
        part = Candidates(p.train.edges[:7] + p.train.edges[-7:],
                          np.r_[np.ones(7), np.zeros(7)].astype(np.float32))
        for name in ("deepsets", "gcn", "allset_cpa"):
            full = Predictor(p.n_nodes, name, hidden_dim=8, heads=2, layers=1, dropout=0)
            batched = copy.deepcopy(full)
            logits = full.decode(full.encode(context), pack(part.edges))
            F.binary_cross_entropy_with_logits(logits, torch.tensor(part.labels)).backward()
            torch.nn.utils.clip_grad_norm_(full.parameters(), 5.)
            optimizer = torch.optim.SGD(batched.parameters(), lr=0)
            train_epoch(batched, context, part, optimizer, 3, "cpu", np.random.default_rng(0))
            for (key, a), (_, b) in zip(full.named_parameters(), batched.named_parameters()):
                self.assertIsNotNone(a.grad, key)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=key)

    def test_readout_cpa_is_parameter_matched_and_initially_identical(self):
        seed_everything(7)
        original = Predictor(10, "allset", hidden_dim=8, heads=2, layers=1)
        seed_everything(7)
        modified = Predictor(10, "allset_cpa", hidden_dim=8, heads=2, layers=1)
        self.assertEqual(set(original.state_dict()), set(modified.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, modified.state_dict()[key])

    def test_explicit_size_intervention_has_local_scope(self):
        edges, z = pack([(0, 1), (0, 1, 2)]), torch.randn(3, 8)
        plain = Predictor(3, "allset", hidden_dim=8, heads=2, layers=1).eval()
        torch.testing.assert_close(plain.decode(z, edges), plain.decode(z, edges, size_override=2))
        modified = Predictor(3, "allset_cpa", hidden_dim=8, heads=2, layers=1).eval()
        a, b = modified.decode(z, edges), modified.decode(z, edges, size_override=2)
        torch.testing.assert_close(a[:1], b[:1])
        self.assertGreater(abs((a - b)[1].item()), 1e-7)

    def test_saturated_negative_space_fails_instead_of_hanging(self):
        with self.assertRaisesRegex(ValueError, "exhausted"):
            sample_negatives([(0, 1)], np.array([0, 1]), {(0, 1)}, np.random.default_rng(0), 1, .5)

    def test_config_rejects_invalid_interpolation(self):
        with self.assertRaises(ValueError):
            validate({**DEFAULTS, "experiments": [{"protocol": "interpolation", "interpolation_bounds": [3, 4]}]})

    def test_loader_canonicalizes_before_deduplication(self):
        temporary = Path(__file__).resolve().parent / f".fixture-{uuid.uuid4().hex}"
        folder = temporary / "fixture"
        folder.mkdir(parents=True)
        paths = [folder / "fixture-nverts.txt", folder / "fixture-simplices.txt"]
        try:
            (folder / "fixture-nverts.txt").write_text("2\n2\n1\n3\n")
            (folder / "fixture-simplices.txt").write_text("8\n4\n4\n8\n6\n8\n8\n4\n")
            dataset = load_dataset("fixture", temporary)
            self.assertEqual(dataset.edges, [(0, 2)])
            self.assertEqual(dataset.n_nodes, 3)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
            folder.rmdir()
            temporary.rmdir()

    def test_constant_predictions_have_chance_performance(self):
        result = metrics(np.array([0, 1, 0, 1]), np.zeros(4))
        self.assertEqual(result["auroc"], .5)
        self.assertEqual(result["auprc"], .5)
        self.assertEqual(result["unique_logits"], 1)


if __name__ == "__main__":
    unittest.main()
