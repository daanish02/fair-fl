from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from fairfl.config import RunConfig
from fairfl.train.engine import RunResult, run

app = typer.Typer(add_completion=False)


@app.command()
def train(
    num_clients: int = typer.Option(10, help="0 = centralised (no clients)"),
    rounds: int = typer.Option(30),
    local_epochs: int = typer.Option(5),
    alpha: float = typer.Option(0.5, help="Dirichlet concentration"),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(64),
    seed: int = typer.Option(0),
    data_root: str = typer.Option("data"),
    device: str = typer.Option("auto"),
    config: str = typer.Option(None, help="YAML file; CLI flags override its values"),
    out: str = typer.Option(None, help="write final client_acc + history JSON here"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, help="-v per-client detail too (default: per-round)"),
):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if config:
        cfg = RunConfig.from_yaml(config)
    else:
        cfg = RunConfig(
            num_clients=num_clients, rounds=rounds, local_epochs=local_epochs, alpha=alpha,
            lr=lr, batch_size=batch_size, seed=seed, data_root=data_root, device=device,
        )

    mode = "centralised" if cfg.is_centralised else f"federated ({cfg.num_clients} clients)"
    typer.echo(f"[fairfl] {mode} | rounds={cfg.rounds} alpha={cfg.alpha} lr={cfg.lr}")

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


def main():
    app()


if __name__ == "__main__":
    main()
