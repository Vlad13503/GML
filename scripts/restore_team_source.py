"""Restore the exact current campaign source without switching or merging branches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / "CURRENT_EXPERIMENT.json").read_text())
    campaign = ROOT / registry["campaign"]
    plan = json.loads((campaign / "run_plan.json").read_text())
    assert plan["commit"] == "2e33dd39c3bd08561d9533d315472af38eb9272d"
    export = ROOT / "tmp/team-procedure-2e33dd3"
    with zipfile.ZipFile(campaign / "source-2e33dd3.zip") as archive:
        files = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
        assert set(files) == set(plan["source_files"])
        # Validate every member and any existing file before writing anything.
        for relative, entry in files.items():
            target = (export / relative).resolve()
            assert target.is_relative_to(export.resolve())
            expected = plan["source_files"][relative]
            assert hashlib.sha256(archive.read(entry)).hexdigest() == expected
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing source differs; preserving it: {target}")
        for relative, entry in files.items():
            target = export / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    print(f"Restored and hash-verified {len(files)} files to {export}")


if __name__ == "__main__":
    main()
