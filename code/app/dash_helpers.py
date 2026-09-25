"""Non-UI helpers for the dashboard: config building, run discovery/loading, trajectory and scope tables."""

from __future__ import annotations

import json
import typing
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from fairfl.core.config import ExperimentConfig
from fairfl.core.engine import Simulator, load_logs
from fairfl.core.types import RoundLog
from fairfl.experiments.runner import summarise_run
from fairfl.fairness.scheme import FairnessScheme, ScopeSpec, StatusSpec
from fairfl.fairness.scopes import evaluate_all_scopes, evaluation_points
from fairfl.fairness.status import unfairness_trajectory

CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_DIR / "configs"
RUNS_DIR = CODE_DIR / "runs"
DASH_RUNS_DIR = RUNS_DIR / "dashboard"


def base_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def load_base(path: str | Path) -> dict[str, Any]:
    return ExperimentConfig.from_yaml(path).model_dump()


def field_specs(params_cls: type[BaseModel]) -> list[dict[str, Any]]:
    """Describe each pydantic field as a widget spec: kind in {bool, int, float, choice, text}."""
    out = []
    for name, f in params_cls.model_fields.items():
        ann = f.annotation
        spec = {"name": name, "default": f.get_default(call_default_factory=True), "help": f.description}
        if typing.get_origin(ann) is typing.Literal:
            spec.update(kind="choice", choices=list(typing.get_args(ann)))
        elif ann is bool:
            spec["kind"] = "bool"
        elif ann is int:
            spec["kind"] = "int"
        elif ann is float:
            spec["kind"] = "float"
        else:
            spec["kind"] = "text"
        out.append(spec)
    return out


def parse_text_value(s: str) -> Any:
    return yaml.safe_load(s) if s.strip() else None


def build_scheme(benefit: str = "dp", level: str = "global", accumulation: str = "cumulative",
                 window: int = 10, discount: float = 0.9, scope: str = "anytime", period: int = 5,
                 bounded_rounds: list[int] | None = None, deploy_every: int = 1,
                 epsilon: float = 0.05) -> FairnessScheme:
    return FairnessScheme(
        status=StatusSpec(benefit=benefit, level=level, accumulation=accumulation, window=window, discount=discount),
        scope=ScopeSpec(kind=scope, period=period, rounds=bounded_rounds or []),
        deploy_every=deploy_every, epsilon=epsilon,
    )


def build_config(base: dict[str, Any], *, data: dict[str, Any], train: dict[str, Any], seed: int,
                 strategy: str, params: dict[str, Any], scheme: FairnessScheme,
                 name: str | None = None) -> ExperimentConfig:
    raw = json.loads(json.dumps(base))
    raw["data"].update(data)
    raw["train"].update(train)
    raw["seed"] = seed
    raw["strategy"] = {"name": strategy, "params": params}
    raw["fairness"] = scheme.model_dump()
    raw["name"] = name or f"{raw['data']['name']}_{strategy}"
    if not Path(raw["data"]["root"]).is_absolute():
        raw["data"]["root"] = str(CODE_DIR / raw["data"]["root"])
    return ExperimentConfig.model_validate(raw)


def new_run_dir(cfg: ExperimentConfig) -> Path:
    return DASH_RUNS_DIR / f"{cfg.name}_{datetime.now():%Y%m%d-%H%M%S}"


def save_run(sim: Simulator, logs: list[RoundLog], out: Path) -> Path:
    path = sim.save(out)
    summary = summarise_run(logs, sim.cfg.fairness)
    (path / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return path


def find_runs(root: Path = RUNS_DIR) -> list[Path]:
    return sorted((p.parent for p in root.rglob("rounds.jsonl")), key=lambda p: str(p).lower())


def run_label(run_dir: Path, root: Path = RUNS_DIR) -> str:
    try:
        return run_dir.relative_to(root).as_posix()
    except ValueError:
        return str(run_dir)


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], list[RoundLog]]:
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return cfg, load_logs(run_dir)


def accuracy_frame(logs: list[RoundLog], label: str) -> pd.DataFrame:
    return pd.DataFrame({"round": [log.round for log in logs],
                         "accuracy": [log.global_accuracy for log in logs], "run": label})


def trajectory_frame(logs: list[RoundLog], scheme: FairnessScheme, label: str) -> pd.DataFrame:
    """Unfairness at each deployed round, flagging scope evaluation points and violations (u > epsilon)."""
    if not logs:
        return pd.DataFrame(columns=["round", "u", "evaluated", "violation", "run"])
    dep, u = unfairness_trajectory(logs, scheme)
    rounds = [logs[i].round for i in dep]
    pts = set(evaluation_points(rounds, scheme.scope))
    evaluated = [i in pts for i in range(len(dep))]
    u = np.asarray(u, float)
    violation = [bool(e and not np.isnan(x) and x > scheme.epsilon) for e, x in zip(evaluated, u)]
    return pd.DataFrame({"round": rounds, "u": u, "evaluated": evaluated, "violation": violation, "run": label})


def scope_table(runs: dict[str, list[RoundLog]], scheme: FairnessScheme) -> pd.DataFrame:
    """One row per (run, scope): max/mean/final unfairness and violation rate, plus the run's final accuracy."""
    rows = []
    for label, logs in runs.items():
        res = evaluate_all_scopes(logs, scheme, period=scheme.scope.period)
        for kind, s in res.items():
            rows.append({"run": label, "scope": kind, "points": s.points, "max": s.max, "mean": s.mean,
                         "final": s.final, "violation_rate": s.violation_rate,
                         "final_accuracy": logs[-1].global_accuracy})
    return pd.DataFrame(rows)


def rank_table(table: pd.DataFrame, metric: str = "max") -> pd.DataFrame:
    """Rank of each run (1 = fairest) per scope under `metric`; shows how the ordering shifts across scopes."""
    if table.empty:
        return table
    piv = table.pivot(index="run", columns="scope", values=metric)
    return piv.rank(method="min", na_option="bottom").astype("Int64")
