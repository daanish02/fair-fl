"""Add finished runs copied from Colab/Kaggle (folders with config.json + summary.json) to results/runs.jsonl.

Usage (from code/): uv run python -m fairfl.experiments.import_runs [runs/colab]
Runs already in the registry with the same summary are skipped; a changed summary is appended (the latest wins).
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from fairfl.experiments.runner import RESULTS_DIR


def import_runs(root: Path) -> list[str]:
    reg = RESULTS_DIR / "runs.jsonl"
    latest: dict[str, dict] = {}
    if reg.exists():
        for line in reg.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["name"]] = row["summary"]
    added = []
    for summary_path in sorted(root.glob("**/summary.json")):
        d = summary_path.parent
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if latest.get(cfg["name"]) == summary:
            continue
        row = {"time": datetime.datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds"),
               "name": cfg["name"], "seed": cfg["seed"], "strategy": cfg["strategy"]["name"],
               "dataset": cfg["data"]["name"], "run_dir": f"colab:{d.as_posix()}", "summary": summary, "config": cfg}
        with open(reg, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        added.append(cfg["name"])
    return added


if __name__ == "__main__":
    for name in import_runs(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/colab")):
        print("added", name)
