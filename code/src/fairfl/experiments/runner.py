from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fairfl.core.config import ExperimentConfig
from fairfl.core.engine import Simulator
from fairfl.core.types import RoundLog
from fairfl.fairness.scheme import FairnessScheme, StatusSpec
from fairfl.fairness.scopes import evaluate_all_scopes
from fairfl.metrics.group import dp_gap, eo_gap, total

REPORT_STATUSES: dict[str, StatusSpec] = {
    "global_dp": StatusSpec(benefit="dp", level="global"),
    "global_eo": StatusSpec(benefit="eo", level="global"),
    "local_dp_max": StatusSpec(benefit="dp", level="local_max"),
    "client_acc": StatusSpec(benefit="accuracy", level="clients"),
}


def set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    import yaml

    raw = cfg.model_dump()
    for item in overrides:
        k, v = item.split("=", 1)
        set_dotted(raw, k, yaml.safe_load(v))
    return ExperimentConfig.model_validate(raw)


def psi(evals) -> float | None:
    """FedFDP Eq. 2: sum_i p_i (F_i - F)^2 with p_i proportional to data size, F = sum_i p_i F_i."""
    evals = [e for e in (evals or []) if e.num_samples]
    if not evals:
        return None
    n = np.array([e.num_samples for e in evals], float)
    f = np.array([e.loss for e in evals])
    p = n / n.sum()
    return float((p * (f - (p * f).sum()) ** 2).sum())


def balanced_accuracy(c) -> float | None:
    """Binary tasks: (TPR + TNR) / 2 pooled over groups (FairWeight's metric)."""
    n = np.array(c.n, float).sum(0)
    cor = np.array(c.correct, float).sum(0)
    if len(n) != 2 or n.min() == 0:
        return None
    return float((cor[1] / n[1] + cor[0] / n[0]) / 2)


def _nan(fn, values) -> float | None:
    """nan-aware reduction that returns None instead of warning when nothing is defined."""
    v = np.array(values, float)
    return float(fn(v)) if np.isfinite(v).any() else None


def _last10(logs: list[RoundLog], fn) -> float | None:
    accs = [log.global_test.accuracy for log in logs if log.global_test is not None][-10:]
    return float(fn(accs)) if accs else None


def summarise_run(logs: list[RoundLog], scheme: FairnessScheme) -> dict[str, Any]:
    last = logs[-1]
    train_evals = next((log.train_evals for log in reversed(logs) if log.train_evals), None)
    accs = np.array([c.accuracy for c in last.clients if c.num_samples])
    g = total([c.groups for c in last.clients])
    out: dict[str, Any] = {
        "final_accuracy": last.global_accuracy,
        "final_global_test_accuracy": last.global_test.accuracy if last.global_test else None,
        "final_global_test_ba": balanced_accuracy(last.global_test.groups) if last.global_test else None,
        "final_global_test_dp": dp_gap(last.global_test.groups) if last.global_test else None,
        # FedMut / FedCDA report mean +- std over the last 10 evaluations of the global model.
        "global_test_acc_last10_mean": _last10(logs, np.mean),
        "global_test_acc_last10_std": _last10(logs, np.std),
        "final_client_acc_std": float(accs.std()) if accs.size else None,
        "final_worst_client_acc": float(accs.min()) if accs.size else None,
        "final_client_acc_var": float(accs.var()) if accs.size else None,
        "final_worst10_acc": float(np.sort(accs)[: max(1, len(accs) // 10)].mean()) if accs.size else None,
        "final_best10_acc": float(np.sort(accs)[-max(1, len(accs) // 10):].mean()) if accs.size else None,
        "final_psi_train": psi(train_evals),
        "final_psi_test": psi(last.clients),
        "final_dp_gap": dp_gap(g),
        "final_local_dp_max": _nan(np.nanmax, [dp_gap(c.groups) for c in last.clients]),
        "final_local_dp_mean": _nan(np.nanmean, [dp_gap(c.groups) for c in last.clients]),
        "final_local_eo_max": _nan(np.nanmax, [eo_gap(c.groups) for c in last.clients]),
        "final_eo_gap": eo_gap(g),
        "seconds": float(sum(log.seconds for log in logs)),
        "scopes": {},
    }
    for name, status in REPORT_STATUSES.items():
        s = scheme.model_copy(update={"status": status})
        out["scopes"][name] = {k: v.model_dump() for k, v in evaluate_all_scopes(logs, s).items()}
    return out


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path | None = None, verbose: bool = True) -> Path:
    sim = Simulator(cfg)

    def show(log: RoundLog) -> None:
        if verbose:
            print(f"[{cfg.name} s{cfg.seed}] round {log.round:3d} acc={log.global_accuracy:.4f} "
                  f"loss={log.global_loss:.4f} ({log.seconds:.1f}s)", flush=True)

    out = Path(out_dir or Path(cfg.output_dir) / cfg.name / f"seed{cfg.seed}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    logs = sim.run(on_round=show, checkpoint=ckpt)
    path = sim.save(out)
    summary = summarise_run(logs, cfg.fairness)
    (path / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    record_result(cfg, summary, path)
    ckpt.unlink(missing_ok=True)
    return path


# Persistent results live in code/results unless FAIRFL_RESULTS_DIR points elsewhere (e.g. Google Drive on Colab).
RESULTS_DIR = Path(os.environ.get("FAIRFL_RESULTS_DIR") or Path(__file__).resolve().parents[3] / "results")


def record_result(cfg: ExperimentConfig, summary: dict[str, Any], run_dir: Path) -> None:
    """Append the finished run to results/runs.jsonl: the persistent record of every result (runs/ is scratch)."""
    import datetime

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "name": cfg.name, "seed": cfg.seed,
           "strategy": cfg.strategy.name, "dataset": cfg.data.name, "run_dir": str(run_dir),
           "summary": summary, "config": cfg.model_dump()}
    line = json.dumps(row, default=float) + "\n"
    with open(RESULTS_DIR / "runs.jsonl", "a", encoding="utf-8") as f:  # one short append per run
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
