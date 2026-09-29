"""Published vs ours: joins results/reproduction/published.csv with the latest matching run in results/runs.jsonl."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fairfl.experiments.runner import RESULTS_DIR


def latest_runs(path: Path = RESULTS_DIR / "runs.jsonl") -> dict[str, dict]:
    runs: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                runs[row["name"]] = row  # later lines win
    return runs


def compare(rel_tol: float = 0.02, abs_tol: float = 0.01) -> list[dict]:
    runs = latest_runs()
    out = []
    with open(RESULTS_DIR / "reproduction" / "published.csv", encoding="utf-8") as f:
        for p in csv.DictReader(f):
            run = runs.get(p["run_name"])
            ours = None
            if run is not None and run["summary"].get(p["our_field"]) is not None:
                ours = float(run["summary"][p["our_field"]]) * float(p["scale"])
            pub = float(p["published"])
            if ours is None:
                status, diff = "NOT RUN", ""
            else:
                diff = ours - pub
                scale = float(p["scale"])
                # absolute slack for rates (0.01 on [0,1], 1 point on %); variances (%^2) use the relative slack only
                slack = abs_tol * scale if scale <= 100 else 0.0
                ok = abs(diff) <= max(slack, rel_tol * abs(pub))
                status = "OK" if ok else "GAP"
                diff = f"{diff:+.4f}"
            out.append({**{k: p[k] for k in ("paper", "setting", "run_name", "metric", "published", "source")},
                        "ours": "" if ours is None else f"{ours:.4f}", "diff": diff, "status": status})
    with open(RESULTS_DIR / "reproduction" / "published_vs_ours.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out
