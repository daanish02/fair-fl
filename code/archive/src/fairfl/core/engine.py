from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from fairfl.clients.base import argmax_rule, evaluate
from fairfl.core.config import ExperimentConfig
from fairfl.core.params import get_params, set_params
from fairfl.core.registry import get_strategy
from fairfl.core.types import EvalRes, RoundLog
from fairfl.data.federated import FederatedData, build_federated
from fairfl.models import build_model
from fairfl.scenarios.base import build_scenario


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def place_data(data: FederatedData, device: torch.device, budget: float = 0.5) -> bool:
    """Move every client's tensors to `device` once if they fit in `budget` of free device memory.
    Otherwise they stay on the CPU and each batch is moved when used. Returns True if moved."""
    if device.type == "cpu":
        return True
    free, _ = torch.cuda.mem_get_info(device)
    if sum(c.nbytes() for c in data.clients) > budget * free:
        return False
    data.clients = [c.to(device) for c in data.clients]
    if data.global_test is not None:
        data.global_test = data.global_test.to(device)
    return True


class Simulator:
    """Sequential FL simulation: every client is evaluated on its local test split after every round,
    so fairness can be scored for any deployed model and any temporal scope afterwards."""

    def __init__(self, cfg: ExperimentConfig, data: FederatedData | None = None):
        self.cfg = cfg
        self.diverged_at: int | None = None
        seed_everything(cfg.seed)
        self.data = data or build_federated(cfg.data, cfg.seed)
        self.device = resolve_device(cfg.device)
        self.model = build_model(cfg.model, self.data.num_features, self.data.num_classes).to(self.device)
        if cfg.model.init_from:
            state = torch.load(cfg.model.init_from, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                self.model.load_state_dict(state)
            else:
                set_params(self.model, state.float().to(self.device))
        self.data_on_device = place_data(self.data, self.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.strategy = get_strategy(cfg.strategy.name).build(cfg.strategy.params, cfg.train, self.rng)
        self.client_algo = self.strategy.make_client()
        self.scenarios = [build_scenario(s.name, s.params, cfg.seed) for s in cfg.scenarios]
        self.states: dict[int, dict] = {c.client_id: {} for c in self.data.clients}
        self.logs: list[RoundLog] = []

    def evaluate_all(self, split: str = "test", rule=None) -> list[EvalRes]:
        rule = rule or self.strategy.decision_rule()
        return [evaluate(self.model, getattr(c, split), c.client_id, rule) for c in self.data.clients]

    def _eval_round(self, rnd: int) -> bool:
        k = self.cfg.train.eval_every
        return (rnd + 1) % k == 0 or rnd == self.cfg.train.rounds - 1

    def _log(self, rnd: int, selected: list[int], train_loss: float | None, t0: float) -> RoundLog:
        evals = self.evaluate_all()
        n = sum(e.num_samples for e in evals)
        gt = (evaluate(self.model, self.data.global_test, -1, argmax_rule)
              if self.data.global_test is not None else None)
        if n:
            acc = sum(e.accuracy * e.num_samples for e in evals if e.num_samples) / n
            loss = sum(e.loss * e.num_samples for e in evals if e.num_samples) / n
        else:  # no local test data: report the global test set (or NaN if there is none)
            acc, loss = (gt.accuracy, gt.loss) if gt is not None else (float("nan"), float("nan"))
        log = RoundLog(
            round=rnd, selected=selected, train_loss=train_loss,
            global_accuracy=acc,
            global_loss=loss,
            clients=evals, strategy_info=dict(self.strategy.info), seconds=time.perf_counter() - t0,
            # The official test set has no client, so client-specific decision rules (LoGoFair) do not apply.
            global_test=gt,
        )
        self.logs.append(log)
        return log

    def _save_checkpoint(self, path: Path, rnd: int, gparams, velocity) -> None:
        state = {"round": rnd, "gparams": gparams, "velocity": velocity, "strategy": self.strategy,
                 "states": self.states, "logs": self.logs, "rng": self.rng, "scenarios": self.scenarios,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)  # atomic: a crash mid-write never leaves a broken checkpoint

    def _load_checkpoint(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.strategy, self.states, self.logs = state["strategy"], state["states"], state["logs"]
        self.rng, self.scenarios = state["rng"], state["scenarios"]
        self.client_algo = self.strategy.make_client()
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["round"] + 1, state["gparams"], state["velocity"]

    def run(self, on_round: Callable[[RoundLog], None] | None = None,
            checkpoint: str | Path | None = None) -> list[RoundLog]:
        """Run all rounds. With `checkpoint`, state is saved every `train.checkpoint_every` rounds and a later
        call with the same path resumes after the last saved round."""
        ids = [c.client_id for c in self.data.clients]
        ckpt = Path(checkpoint) if checkpoint else None
        start = 0
        if ckpt is not None and ckpt.exists():
            start, gparams, velocity = self._load_checkpoint(ckpt)
            print(f"[{self.cfg.name} s{self.cfg.seed}] resumed from checkpoint at round {start}", flush=True)
        else:
            gparams = get_params(self.model)
            self.strategy.initialize(gparams, len(ids), [len(c.train) for c in self.data.clients],
                                     [p.numel() for p in self.model.parameters()])
            velocity = None
        t0 = time.perf_counter()
        every = self.cfg.train.checkpoint_every
        for rnd in range(start, self.cfg.train.rounds):
            for sc in self.scenarios:
                sc.before_round(rnd, self.data)
            set_params(self.model, gparams)
            pre = {cid: evaluate(self.model, getattr(self.data.clients[cid], self.strategy.pre_eval_split), cid)
                   for cid in self.strategy.configure_eval(rnd, ids)}
            ins_map = self.strategy.configure_round(rnd, gparams, ids, pre)
            if self.strategy.needs_loss_before:
                for ins in ins_map.values():
                    ins.config["need_loss_before"] = True
            results = []
            for cid, ins in ins_map.items():
                # CPU generator: batch permutations and noise are drawn on the CPU, so a seed gives the same
                # draws on any device; tensors are moved to the model's device where needed.
                gen = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + rnd * 1009 + cid)
                res = self.client_algo.fit(self.model, self.data.clients[cid], ins, self.states[cid], gen)
                for sc in self.scenarios:
                    res = sc.after_fit(rnd, res, gparams)
                results.append(res)
            new = self.strategy.aggregate(rnd, gparams, results)
            if self.cfg.train.server_momentum > 0:
                step = new - gparams
                velocity = step if velocity is None else self.cfg.train.server_momentum * velocity + step
                new = gparams + velocity
            gparams = new
            set_params(self.model, gparams)
            if not self._eval_round(rnd):
                if ckpt is not None and every and (rnd + 1) % every == 0:
                    self._save_checkpoint(ckpt, rnd, gparams, velocity)
                continue
            train_loss = float(np.mean([r.metrics.get("train_loss", np.nan) for r in results]))
            log = self._log(rnd, sorted(ins_map), train_loss, t0)
            t0 = time.perf_counter()
            self.strategy.on_round_end(log)
            if on_round:
                on_round(log)
            if not math.isfinite(log.global_loss):
                # NaN/inf never recovers: stop instead of burning the remaining rounds (summary records the round)
                self.diverged_at = rnd + 1
                print(f"[{self.cfg.name} s{self.cfg.seed}] diverged (loss {log.global_loss}) at round {rnd + 1}; "
                      "stopping", flush=True)
                break
            if ckpt is not None and every and (rnd + 1) % every == 0:
                self._save_checkpoint(ckpt, rnd, gparams, velocity)
        t0 = time.perf_counter()
        if self.cfg.train.rounds == 0:
            self._log(0, [], None, t0)  # the loaded model before any post-processing
        if self.logs:
            self.logs[-1].train_evals = [evaluate(self.model, c.train, c.client_id, argmax_rule)
                                         for c in self.data.clients]
        # Post-processing gets CPU copies of the client data; predict_proba returns CPU probabilities.
        pp_clients = self.data.clients if self.device.type == "cpu" else [c.to("cpu") for c in self.data.clients]
        if self.strategy.postprocess(gparams, self.model, pp_clients):
            self.strategy.info["postprocessed"] = True
            log = self._log(self.cfg.train.rounds, [], None, t0)
            if on_round:
                on_round(log)
        self.final_params = gparams
        return self.logs

    def save(self, out_dir: str | Path | None = None) -> Path:
        out = Path(out_dir or Path(self.cfg.output_dir) / self.cfg.name / f"seed{self.cfg.seed}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(self.cfg.model_dump_json(indent=2), encoding="utf-8")
        with open(out / "rounds.jsonl", "w", encoding="utf-8") as f:
            for log in self.logs:
                f.write(log.model_dump_json() + "\n")
        return out


def load_logs(run_dir: str | Path) -> list[RoundLog]:
    with open(Path(run_dir) / "rounds.jsonl", encoding="utf-8") as f:
        return [RoundLog.model_validate(json.loads(line)) for line in f if line.strip()]
