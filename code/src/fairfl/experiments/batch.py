"""Run a suite of experiments in parallel CPU processes, skipping runs that already have a summary.json.

Suite YAML:
    name: fcfl_mnist
    runs:
      - config: configs/repro/fcfl/mnist_shards.yaml
        seeds: [0, 1, 2, 3, 4]
        set: {strategy.name: fcfl}
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SuiteRun(BaseModel):
    config: str
    seeds: list[int] = Field(default_factory=lambda: [0])
    set: dict = Field(default_factory=dict)


class Suite(BaseModel):
    name: str
    runs: list[SuiteRun]


def _job(args: tuple[str, int, dict, str, int]) -> tuple[str, str, float]:
    config, seed, overrides, out, threads = args
    import torch

    torch.set_num_threads(threads)
    from fairfl.core.config import ExperimentConfig
    from fairfl.experiments.runner import apply_overrides, run_experiment

    t0 = time.time()
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "log.txt", "a", encoding="utf-8") as log:  # append: a resumed run keeps its history
        sys.stdout = sys.stderr = log
        try:
            cfg = ExperimentConfig.from_yaml(config)
            cfg = apply_overrides(cfg, [f"{k}={json.dumps(v)}" for k, v in overrides.items()])
            cfg = cfg.model_copy(update={"seed": seed})
            run_experiment(cfg, out_p, verbose=True)
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "error"
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    return out, status, time.time() - t0


def plan(suite: Suite, root: Path) -> list[tuple[str, int, dict, str]]:
    from fairfl.core.config import ExperimentConfig

    jobs = []
    for r in suite.runs:
        name = r.set.get("name") or ExperimentConfig.from_yaml(r.config).name
        for seed in r.seeds:
            out = root / suite.name / name / f"seed{seed}"
            if not (out / "summary.json").exists():
                jobs.append((r.config, seed, r.set, str(out)))
    return jobs


def run_suite(path: str, workers: int = 3, threads: int = 4, root: str = "runs") -> None:
    with open(path, encoding="utf-8") as f:
        suite = Suite.model_validate(yaml.safe_load(f))
    jobs = plan(suite, Path(root))
    total = sum(len(r.seeds) for r in suite.runs)
    print(f"[{suite.name}] {len(jobs)} of {total} runs to do; {workers} workers x {threads} threads", flush=True)
    if not jobs:
        return
    keep_awake(True)
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for out, status, secs in pool.imap_unordered(_job, [(*j, threads) for j in jobs]):
                print(f"[{suite.name}] {status:5s} {secs / 60:6.1f} min  {out}", flush=True)
    finally:
        keep_awake(False)


def keep_awake(on: bool) -> None:
    """Windows: stop the machine from idle-sleeping while a suite runs (it can still be put to sleep manually)."""
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
