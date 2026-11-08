import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hnn_project.cli import DEFAULTS
from hnn_project.data import make_protocol, synthetic_dataset
from hnn_project.tune import grid, rank_candidates, save_protocol, validation_fit


class TuningTests(unittest.TestCase):
    def test_saved_protocol_roundtrip_and_changed_manifest_rejection(self):
        protocol = make_protocol(synthetic_dataset(), {**DEFAULTS, "protocol": "order", "k": 3}, 0)
        with tempfile.TemporaryDirectory() as temporary:
            folder = save_protocol(Path(temporary), protocol)
            self.assertEqual(save_protocol(Path(temporary), protocol), folder)
            protocol.manifest["split_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "Protocol differs"):
                save_protocol(Path(temporary), protocol)

    def test_grid_and_selection_are_validation_only_with_stable_ties(self):
        config = {**DEFAULTS, "learning_rates": [.0003, .001, .003]}
        candidates = grid(config)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({tuple(c.items()) for c in candidates}), 12)
        records = [{"trial_id": "b", "validation_auprc": .7, "test_auprc": 1},
                   {"trial_id": "c", "validation_auprc": .8, "test_auprc": 0},
                   {"trial_id": "a", "validation_auprc": .7, "test_auprc": 0}]
        self.assertEqual([r["trial_id"] for r in rank_candidates(records)], ["c", "a", "b"])

    def test_fit_cannot_access_test_candidates_and_replays_validation(self):
        config = {**DEFAULTS, "protocol": "order", "k": 3}
        protocol = make_protocol(synthetic_dataset(), config, 0)
        training = {"validation_auprc": .5, "training_seconds_all_trials": 0}
        with tempfile.TemporaryDirectory() as temporary:
            with patch("hnn_project.tune.fit", return_value=(object(), training)) as fit_mock:
                with patch("hnn_project.tune.predict", return_value=np.zeros(len(protocol.val.edges))) as predict_mock:
                    result = validation_fit(protocol, None, config, 0, Path(temporary), "cpu",
                        {"hidden_dim": 32, "layers": 1, "learning_rate": .001})
            self.assertEqual(len(fit_mock.call_args.args[1].test.edges), 0)
            self.assertIs(predict_mock.call_args.args[2], protocol.val)
            self.assertFalse(result["test_evaluated"])
            self.assertTrue((Path(temporary) / "val_predictions.npz").exists())
            self.assertFalse((Path(temporary) / "test_predictions.npz").exists())
            with patch("hnn_project.tune.fit", side_effect=AssertionError("Should resume")):
                again = validation_fit(protocol, None, config, 0, Path(temporary), "cpu",
                    {"hidden_dim": 32, "layers": 1, "learning_rate": .001})
            self.assertEqual(again["identity"], result["identity"])


if __name__ == "__main__":
    unittest.main()
