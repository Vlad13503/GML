"""Resolve current result artifacts; refuse superseded or unverified campaigns."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def current_campaign(root=ROOT, require_verified=True):
    root = Path(root).resolve()
    registry = json.loads((root / "CURRENT_EXPERIMENT.json").read_text())
    campaign = (root / registry["campaign"]).resolve()
    if not campaign.is_relative_to(root / "results"):
        raise ValueError("Current campaign must be inside this repository's results directory")
    for relative in registry["superseded_roots"]:
        if campaign.is_relative_to((root / relative).resolve()):
            raise ValueError("Registry points into a superseded campaign")
    if (campaign / "SUPERSEDED.json").exists():
        raise ValueError("Current campaign has a superseded marker")
    if require_verified:
        if registry["status"] != "verified_current":
            raise ValueError(f"Current results are not verified: {registry['status']}")
        verification = root / registry["verification"]
        if not verification.resolve().is_relative_to(campaign):
            raise ValueError("Verification must belong to the current campaign")
        if json.loads(verification.read_text())["status"] != "VERIFIED_CURRENT":
            raise ValueError("Verification did not pass")
    return campaign


def require_current_path(path, root=ROOT, require_verified=True):
    campaign = current_campaign(root, require_verified)
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(campaign):
        raise ValueError("Artifact is historical, superseded, or outside the current campaign")
    return resolved


if __name__ == "__main__":
    print(current_campaign())
