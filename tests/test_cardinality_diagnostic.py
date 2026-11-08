import unittest

import torch

from hnn_project.data import pack
from hnn_project.models import AttentionPool
from scripts.cardinality_diagnostics import distances, stages_from_raw


class CardinalityDiagnosticTests(unittest.TestCase):
    def test_fixed_raw_stages_reproduce_pool_forward(self):
        torch.manual_seed(1)
        x = torch.randn(5, 8, dtype=torch.float64)
        packed = pack([(0, 1), (2, 3, 4)])
        for mode in ("none", "scaled"):
            pool = AttentionPool(8, 2, mode).double()
            _, trace = pool(x, packed.groups, 2, packed.sizes, trace=True)
            reconstructed = stages_from_raw(pool, trace["attention"], packed.sizes)
            for stage, expected in trace.items():
                torch.testing.assert_close(reconstructed[stage], expected, rtol=0, atol=0)

    def test_zero_residual_normalization_and_disabled_scaling_controls(self):
        torch.manual_seed(2)
        raw = torch.randn(3, 8, dtype=torch.float64)
        pool = AttentionPool(8, 2, "scaled").double()
        # Test mathematical scale invariance separately from finite-epsilon effects.
        pool.norm1.eps = 0
        a = stages_from_raw(pool, raw, 1, zero_residual=True)
        b = stages_from_raw(pool, raw, 8, zero_residual=True)
        torch.testing.assert_close(b["modified"], a["modified"] * 8)
        torch.testing.assert_close(a["post_norm"], b["post_norm"], rtol=1e-12, atol=1e-12)
        for mode in ("none", "scaled"):
            pool.mode = mode
            a = stages_from_raw(pool, raw, 1, disable_scaling=True)
            b = stages_from_raw(pool, raw, 8, disable_scaling=True)
            for stage in a:
                torch.testing.assert_close(a[stage], b[stage], rtol=0, atol=0)

    def test_distances_distinguish_magnitude_from_direction(self):
        reference = torch.tensor([[1., 2.], [0., 0.]])
        result = distances(reference * 3, reference)
        self.assertAlmostEqual(result["relative_l2"][0].item(), 2)
        self.assertAlmostEqual(result["cosine"][0].item(), 1, places=6)
        self.assertFalse(result["cosine_valid"][1])
        self.assertTrue(torch.isnan(result["cosine"][1]))
