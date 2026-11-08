import copy
import unittest
from unittest.mock import patch

import torch

from hnn_project.cli import DEFAULTS
from hnn_project.data import make_context, make_protocol, synthetic_dataset
from hnn_project.models import Predictor
from scripts.encoder_signal_check import encode_with_condition


class EncoderSignalCheckTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        self.protocol = make_protocol(synthetic_dataset(), {**DEFAULTS, "protocol": "order", "k": 3}, 0)
        self.context = make_context(self.protocol)
        self.model = Predictor(self.protocol.n_nodes, "allset_cpa_encoder", hidden_dim=8, heads=2, layers=2).eval()

    def test_disabled_matches_unscaled_operation_and_restores_normal(self):
        with torch.no_grad():
            original = self.model.encode(self.context)
            disabled = encode_with_condition(self.model, self.context, "disabled")
            unscaled = copy.deepcopy(self.model)
            for layer in unscaled.layers:
                layer.v2e.mode = "none"
            torch.testing.assert_close(disabled, unscaled.encode(self.context), rtol=0, atol=0)
            torch.testing.assert_close(self.model.encode(self.context), original, rtol=0, atol=0)

    def test_constant_preserves_synthetic_singletons_and_hooks_clean_up_on_error(self):
        seen = []
        def observe(pool, args, kwargs, output):
            seen.append((args[3].clone(), kwargs["size_override"].clone()))
        handles = [layer.v2e.register_forward_hook(observe, with_kwargs=True) for layer in self.model.layers]
        try:
            encode_with_condition(self.model, self.context, "constant", 3)
        finally:
            for handle in handles:
                handle.remove()
        for sizes, effective in seen:
            self.assertTrue((effective[sizes == 1] == 1).all())
            self.assertTrue((effective[sizes > 1] == 3).all())
        with patch.object(self.model, "encode", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                encode_with_condition(self.model, self.context, "disabled")
        self.assertTrue(all(not layer.v2e._forward_pre_hooks for layer in self.model.layers))
