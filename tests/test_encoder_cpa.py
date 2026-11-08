import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from hnn_project.cli import DEFAULTS
from hnn_project.data import Candidates, make_context, make_protocol, pack, synthetic_dataset
from hnn_project.models import Predictor
from hnn_project.train import seed_everything, train_epoch


class EncoderCPATests(unittest.TestCase):
    def test_encoder_only_placement_and_exact_shared_initialization(self):
        models = []
        for name in ("allset", "allset_cpa_encoder"):
            seed_everything(0)
            models.append(Predictor(12, name, hidden_dim=8, heads=2, layers=2))
        vanilla, cpa = models
        self.assertEqual(set(vanilla.state_dict()), set(cpa.state_dict()))
        for key, value in vanilla.state_dict().items():
            torch.testing.assert_close(value, cpa.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(cpa.pool.mode, "none")
        for layer in cpa.layers:
            self.assertEqual(layer.v2e.mode, "scaled")
            self.assertEqual(layer.e2v.mode, "none")

    def test_encoder_cpa_batched_gradient_matches_full_objective(self):
        torch.set_num_threads(2)
        seed_everything(3)
        protocol = make_protocol(synthetic_dataset(), {**DEFAULTS, "protocol": "order", "k": 3}, 0)
        context = make_context(protocol)
        part = Candidates(protocol.train.edges[:5] + protocol.train.edges[-5:],
                          np.array([1] * 5 + [0] * 5, dtype=np.float32))
        full = Predictor(protocol.n_nodes, "allset_cpa_encoder", hidden_dim=8, heads=2, layers=1, dropout=0)
        batched = copy.deepcopy(full)
        F.binary_cross_entropy_with_logits(full.decode(full.encode(context), pack(part.edges)),
                                          torch.tensor(part.labels)).backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 5.)
        train_epoch(batched, context, part, torch.optim.SGD(batched.parameters(), lr=0),
                    3, "cpu", np.random.default_rng(0))
        for (key, a), (_, b) in zip(full.named_parameters(), batched.named_parameters()):
            self.assertIsNotNone(a.grad, key)
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=key)


if __name__ == "__main__":
    unittest.main()
