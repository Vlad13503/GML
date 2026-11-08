import json
from pathlib import Path
import tempfile
import unittest

from scripts.current_results import current_campaign, require_current_path


class CurrentResultsTests(unittest.TestCase):
    def test_superseded_and_pending_results_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh, old = root / "results/fresh", root / "results/old"
            fresh.mkdir(parents=True)
            old.mkdir()
            record = {"campaign": "results/fresh", "status": "tuning_running", "superseded_roots": ["results/old"],
                      "verification": "results/fresh/verification.json"}
            path = root / "CURRENT_EXPERIMENT.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "not verified"):
                current_campaign(root)
            self.assertEqual(current_campaign(root, False), fresh.resolve())
            record["status"] = "verified_current"
            path.write_text(json.dumps(record))
            (fresh / "verification.json").write_text(json.dumps({"status": "VERIFIED_CURRENT"}))
            self.assertEqual(require_current_path(fresh / "SUMMARY.md", root), (fresh / "SUMMARY.md").resolve())
            with self.assertRaisesRegex(ValueError, "historical"):
                require_current_path(old / "SUMMARY.md", root)
            record["campaign"] = "results/old"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "superseded"):
                current_campaign(root)
