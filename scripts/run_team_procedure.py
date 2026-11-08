"""Run the pinned team procedure in an isolated Git export, never the legacy package."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tmp/team-procedure-2e33dd3"
CAMPAIGN = ROOT / "results/team-procedure-2e33dd3-20261107"
COMMIT = "2e33dd39c3bd08561d9533d315472af38eb9272d"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(phase, **details):
    path = ROOT / "CURRENT_EXPERIMENT.json"
    current = json.loads(path.read_text())
    current.update(status=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    write(path, current)
    write(CAMPAIGN / "status.json", current)
    print(phase, details, flush=True)


def command(arguments, log):
    environment = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, "-u", "-m", "hnn_project", *arguments],
            cwd=EXPORT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")


def main():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    # Byte hashes include the exact normalized sources exported from the Git commit.
    sources = {str(p.relative_to(EXPORT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(EXPORT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    data = {str(p.relative_to(ROOT / "data")).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "data").glob("*/*.txt"))}
    plan = {"commit": COMMIT, "source_export": str(EXPORT), "source_files": sources,
        "data_files": data, "device": "cuda", "cublas_workspace_config": ":4096:8",
        "tuning": "Exact configs/tune.json from pinned traian commit: five models, eight settings, two LR trials, seed 0",
        "comparison": "Exact configs/encoder_ablation.json: four models, two datasets, three seeds, two LR trials",
        "test_used_for_tuning": False, "legacy_results_eligible": False}
    frozen = CAMPAIGN / "run_plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("Frozen source/data campaign identity changed")
    write(frozen, plan)
    (CAMPAIGN / "controller_source.py").write_bytes(Path(__file__).read_bytes())
    (CAMPAIGN / "tune.json").write_bytes((EXPORT / "configs/tune.json").read_bytes())
    (CAMPAIGN / "comparison_config.json").write_bytes((EXPORT / "configs/encoder_ablation.json").read_bytes())
    with (CAMPAIGN / "environment-freeze.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    status("tuning_running", planned_tuning_fit_calls=80, planned_tuning_lr_trials=160,
           planned_comparison_fit_calls=24, planned_comparison_lr_trials=48)
    command(["tune", "--config", str(CAMPAIGN / "tune.json"), "--data-dir", str(ROOT / "data"),
             "--output", str(CAMPAIGN / "tuning"), "--device", "cuda"], CAMPAIGN / "tuning.log")
    choices = list((CAMPAIGN / "tuning").glob("tuning-*/best_settings.json"))
    if len(choices) != 1:
        raise ValueError("Expected one completed tuning output")
    overrides = choices[0]
    status("comparison_running", tuning_root=str(overrides.parent.relative_to(ROOT)),
           shared_settings=json.loads(overrides.read_text()))
    command(["run", "--config", str(CAMPAIGN / "comparison_config.json"), "--overrides", str(overrides),
             "--data-dir", str(ROOT / "data"), "--output", str(CAMPAIGN / "fits"), "--device", "cuda"],
            CAMPAIGN / "comparison.log")
    suites = [p.parent for p in (CAMPAIGN / "fits").glob("*/config.json")]
    if len(suites) != 1:
        raise ValueError("Expected one completed comparison suite")
    results = list(suites[0].rglob("result.json"))
    if len(results) != 24:
        raise ValueError(f"Expected 24 comparison results, got {len(results)}")
    status("fits_complete_verification_pending", comparison_suite=str(suites[0].relative_to(ROOT)),
           completed_comparison_fit_calls=len(results))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise
