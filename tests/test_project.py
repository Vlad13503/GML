import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from hnn_project.cli import DEFAULTS, run_suite, tune_suite, validate
from hnn_project.data import (Candidates, load_dataset, make_context, make_protocol, pack,
                              sample_negatives, synthetic_dataset)
from hnn_project.models import AttentionPool, Predictor, segment_softmax
from hnn_project.report import compare, report
from hnn_project.train import evaluate_metrics, metrics, seed_everything, train_epoch


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

    def test_matched_arms_share_evaluation_and_differ_only_in_training(self):
        arms = {arm: make_protocol(self.dataset, {**self.config, "protocol": f"matched_{arm}"}, 0)
                for arm in ("unseen", "targets", "all")}
        unseen, targets, full = arms["unseen"], arms["targets"], arms["all"]
        for p in arms.values():
            self.assertEqual(p.val.edges, unseen.val.edges)
            self.assertEqual(p.test.edges, unseen.test.edges)
            self.assertEqual(p.manifest["shared_eval_sha256"], unseen.manifest["shared_eval_sha256"])
            evaluation = set(p.val.edges) | set(p.test.edges)
            self.assertFalse(evaluation & set(p.train.edges))
            self.assertFalse(evaluation & set(p.context))
            self.assertTrue(all(len(e) > 3 for e, y in zip(p.test.edges, p.test.labels) if y))
            self.assertTrue(all(len(e) <= 3 for e, y in zip(p.val.edges, p.val.labels) if y))
        positives = lambda part: [e for e, y in zip(part.edges, part.labels) if y]
        self.assertTrue(all(len(e) <= 3 for e in unseen.context + positives(unseen.train)))
        self.assertEqual(targets.context, unseen.context)
        self.assertEqual(len(positives(targets.train)), len(positives(unseen.train)))
        self.assertTrue(any(len(e) > 3 for e in positives(targets.train)))
        self.assertTrue(any(len(e) > 3 for e in full.context))
        self.assertTrue(set(unseen.context) < set(full.context))
        self.assertTrue(set(positives(unseen.train)) < set(positives(full.train)))

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


    def test_sparse_bins_follow_positive_counts(self):
        edges = [(0, 1, 2)] * 19 + [(0, 1, 2)] * 41 + [(0, 1)] * 40
        labels = np.r_[np.ones(19), np.zeros(41), np.ones(20), np.zeros(20)].astype(np.float32)
        by_size = evaluate_metrics(Candidates(edges, labels), np.arange(100.))["by_size"]
        self.assertTrue(by_size["3"]["small_bin"])
        self.assertFalse(by_size["2"]["small_bin"])


class SuiteTests(unittest.TestCase):
    small = {"datasets": ["synthetic"], "seeds": [0], "experiments": [{"protocol": "order", "k": 3}],
             "models": ["allset", "allset_cpa"], "hidden_dim": 8, "heads": 2, "layers": 1, "epochs": 2,
             "eval_every": 1, "patience": 2, "batch_size": 64, "probe_limit": 50}

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.output = Path(folder.name)

    def run_small(self, **changes):
        return run_suite({**DEFAULTS, **self.small, **changes}, Path("data"), self.output, "cpu")

    def test_training_records_epoch_zero_and_convergence(self):
        root = self.run_small()
        training = json.loads((root / "synthetic/order-k3/seed-0/allset/training.json").read_text())
        history = training["trials"][0]["history"]
        self.assertEqual([h["epoch"] for h in history], [0, 1, 2])
        self.assertIsNone(history[0]["loss"])
        self.assertIsNotNone(history[1]["loss"])
        self.assertIn("auprc", history[0]["train"])
        self.assertEqual(training["never_improved"], training["best_epoch"] == 0)
        report(root)
        for name in ("convergence.csv", "paired_cpa_summary.csv", "interpretability_summary.csv",
                     "size_response.csv", "figures/synthetic-order-k3-convergence.png"):
            self.assertTrue((root / name).exists(), name)

    def test_report_refuses_mixed_suites(self):
        root = self.run_small()
        with self.assertRaisesRegex(ValueError, "not a single suite"):
            report(self.output)
        path = root / "synthetic/order-k3/seed-0/allset/result.json"
        result = json.loads(path.read_text())
        path.write_text(json.dumps({**result, "suite": "another"}))
        with self.assertRaisesRegex(ValueError, "other suites"):
            report(root)
        path.write_text(json.dumps({**result, "split_sha256": "different"}))
        with self.assertRaisesRegex(ValueError, "did not share candidates"):
            report(root)

    def test_compare_labels_only_what_differs(self):
        easy = self.run_small(hard_negative_fraction=0.)
        hard = self.run_small(hard_negative_fraction=1.)
        output = compare([easy, hard], self.output / "comparison")
        self.assertIn("hard_negative_fraction", output.read_text())
        with self.assertRaisesRegex(ValueError, "differ"):
            compare([easy, easy], self.output / "same")

    def test_tuning_never_scores_the_test_partition(self):
        real = make_protocol

        def without_test_labels(*args):
            protocol = real(*args)
            protocol.test.labels = np.full_like(protocol.test.labels, np.nan)
            return protocol

        with mock.patch("hnn_project.cli.make_protocol", without_test_labels):
            root = tune_suite({**DEFAULTS, **self.small}, {"dropout": [0., .2]}, Path("data"), self.output, "cpu")
        best = json.loads((root / "best_settings.json").read_text())["dataset_overrides"]["synthetic"]
        self.assertIn(best["dropout"], (0., .2))
        validate({**DEFAULTS, **self.small, "dataset_overrides": {"synthetic": best}})

    def test_overrides_only_accept_listed_datasets_and_tunable_keys(self):
        for overrides in ({"other": {"dropout": .1}}, {"synthetic": {"epochs": 1}}):
            with self.assertRaisesRegex(ValueError, "Overrides"):
                validate({**DEFAULTS, **self.small, "dataset_overrides": overrides})


if __name__ == "__main__":
    unittest.main()
