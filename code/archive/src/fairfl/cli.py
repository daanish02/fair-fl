from __future__ import annotations

from typing import Annotated, Optional

import typer

app = typer.Typer(add_completion=False, help="Fair federated learning testbed.")


@app.command()
def run(
    config: str,
    seed: Annotated[Optional[int], typer.Option(help="Override the config seed.")] = None,
    set_: Annotated[Optional[list[str]], typer.Option("--set", help="Dotted override, e.g. train.rounds=5")] = None,
    out: Annotated[Optional[str], typer.Option(help="Output directory.")] = None,
    quiet: bool = False,
) -> None:
    """Run one experiment from a YAML config."""
    from fairfl.core.config import ExperimentConfig
    from fairfl.experiments.runner import apply_overrides, run_experiment

    cfg = ExperimentConfig.from_yaml(config)
    if set_:
        cfg = apply_overrides(cfg, set_)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    path = run_experiment(cfg, out, verbose=not quiet)
    typer.echo(f"saved to {path}")


@app.command()
def batch(
    suite: str,
    workers: Annotated[int, typer.Option(help="Parallel processes.")] = 3,
    threads: Annotated[int, typer.Option(help="Torch threads per process.")] = 4,
    root: Annotated[str, typer.Option(help="Output root.")] = "runs",
) -> None:
    """Run a suite YAML of configs x seeds in parallel, skipping finished runs."""
    from fairfl.experiments.batch import run_suite

    run_suite(suite, workers, threads, root)


@app.command("compare")
def compare_cmd() -> None:
    """Write results/reproduction/published_vs_ours.csv and print it."""
    from fairfl.experiments.compare import compare

    for r in compare():
        typer.echo(f"{r['status']:8s} {r['run_name']:32s} {r['metric']:14s} paper {r['published']:>8s} ours {r['ours']:>8s} {r['diff']}")


@app.command("strategies")
def strategies() -> None:
    """List registered strategies."""
    from fairfl.core.registry import list_strategies

    for name in list_strategies():
        typer.echo(name)


def main() -> None:
    app()
