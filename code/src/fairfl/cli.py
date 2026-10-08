from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from fairfl.config import RunConfig
from fairfl.models.registry import list_models
from fairfl.registry import list_strategies
from fairfl.train.engine import RunResult, run

app = typer.Typer(add_completion=False)


@app.command()
def train(
    config: Annotated[Optional[str], typer.Option("--config", help="YAML base config")] = None,
    set_: Annotated[Optional[list[str]], typer.Option("--set", help="dotted override, e.g. train.lr=0.05")] = None,
    out: Annotated[Optional[str], typer.Option(help="write final client_acc + history JSON here")] = None,
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, help="-v per-client detail too (default: per-round)"),
):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = RunConfig.from_yaml_and_overrides(config, set_ or [])

    mode = "centralised" if cfg.is_centralised else f"federated ({cfg.dataset.num_clients} clients)"
    typer.echo(f"[fairfl] {mode} | strategy={cfg.strategy.name} model={cfg.model.name} "
               f"rounds={cfg.train.rounds} alpha={cfg.dataset.alpha} lr={cfg.train.lr}")

    result: RunResult | None = None
    for item in run(cfg):
        if isinstance(item, RunResult):
            result = item

    typer.echo("--- final per-client accuracy ---")
    for c, acc in sorted(result.client_acc.items()):
        typer.echo(f"client {c}: {acc:.4f}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": cfg.model_dump(),
            "history": [h.__dict__ for h in result.history],
            "client_acc": result.client_acc,
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        typer.echo(f"saved -> {out}")


@app.command("strategies")
def strategies_cmd():
    for name in list_strategies():
        typer.echo(name)


@app.command("models")
def models_cmd():
    for name in list_models():
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
