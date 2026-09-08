"""Command-line interface for the parts inflation prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from parts_inflation.config import load_config, project_root
from parts_inflation.ingest import profile_sources
from parts_inflation.pipeline import init_config, run_pipeline, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Client-specific parts inflation estimation and forecasting",
)
console = Console()


def _default_input() -> Path:
    return project_root() / "data" / "raw"


def _default_config() -> Path:
    return project_root() / "config" / "model_config.xlsx"


def _default_output() -> Path:
    return project_root() / "outputs"


@app.command("init-config")
def init_config_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir", help="Directory of PO workbooks"),
    output: Path = typer.Option(_default_config(), "--output", help="Path for model_config.xlsx"),
) -> None:
    """Generate an editable model_config.xlsx from discovered Bucket/Description pairs."""
    path = init_config(input_dir, output)
    console.print(f"[green]Wrote config[/green] {path}")


@app.command("profile")
def profile_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
) -> None:
    """Profile source workbooks (row counts, date ranges, fingerprints)."""
    setup_logging(_default_output())
    df = profile_sources(input_dir)
    console.print(df.to_string(index=False))
    console.print(f"\nTotal rows: {df['nrows'].sum():,}")


@app.command("validate")
def validate_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
    config: Path = typer.Option(_default_config(), "--config"),
) -> None:
    """Validate inputs and configuration without writing forecasts."""
    from parts_inflation.clean import clean_po_lines
    from parts_inflation.classify import apply_scope_and_category
    from parts_inflation.diagnostics import compare_to_expected_profile, profile_summary
    from parts_inflation.ingest import load_all_po_lines

    setup_logging(_default_output())
    cfg = load_config(config, create_if_missing=True)
    raw, infos, warnings = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, cfg)
    classified, mapping = apply_scope_and_category(cleaned, cfg)
    summary = profile_summary(classified, infos)
    cmp = compare_to_expected_profile(summary)
    console.print(cmp.to_string(index=False))
    console.print(f"Scope mapping rows: {len(mapping)}")
    console.print(f"Model-eligible rows: {int(classified['model_eligible'].sum()):,}")
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")


@app.command("backtest")
def backtest_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
    config: Path = typer.Option(_default_config(), "--config"),
    fast_mode: bool = typer.Option(False, "--fast-mode"),
) -> None:
    """Run rolling-origin backtests only and print summary metrics."""
    from parts_inflation.aggregate import aggregate_same_day
    from parts_inflation.backtest import run_backtests
    from parts_inflation.classify import apply_scope_and_category
    from parts_inflation.clean import clean_po_lines
    from parts_inflation.ingest import load_all_po_lines

    setup_logging(_default_output())
    cfg = load_config(config, cli_overrides={"fast_mode": fast_mode}, create_if_missing=True)
    raw, _, _ = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, cfg)
    classified, _ = apply_scope_and_category(cleaned, cfg)
    bt = run_backtests(classified, cfg)
    console.print(bt.summary.to_string(index=False) if bt.summary is not None else "No results")
    console.print(f"Selected: {bt.selected_model}")
    console.print(bt.selection_rationale)


@app.command("run")
def run_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
    config: Path = typer.Option(_default_config(), "--config"),
    output_dir: Path = typer.Option(_default_output(), "--output-dir"),
    target_date: Optional[str] = typer.Option(None, "--target-date", help="YYYY-MM-DD"),
    base_date: Optional[str] = typer.Option(None, "--base-date", help="YYYY-MM-DD"),
    scope_mode: Optional[str] = typer.Option(None, "--scope-mode"),
    fast_mode: bool = typer.Option(False, "--fast-mode"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    rebuild_cache: bool = typer.Option(False, "--rebuild-cache"),
    skip_backtest: bool = typer.Option(False, "--skip-backtest"),
    bootstrap_iterations: Optional[int] = typer.Option(None, "--bootstrap-iterations"),
) -> None:
    """Run the full estimation, forecast, and Excel report pipeline."""
    overrides = {}
    if bootstrap_iterations is not None:
        overrides["bootstrap_iterations"] = bootstrap_iterations
    try:
        out = run_pipeline(
            input_dir=input_dir,
            config_path=config,
            output_dir=output_dir,
            target_date=target_date,
            base_date=base_date,
            scope_mode=scope_mode,
            fast_mode=fast_mode,
            no_cache=no_cache,
            rebuild_cache=rebuild_cache,
            skip_backtest=skip_backtest,
            cli_overrides=overrides,
        )
        console.print(f"[green]Results written to[/green] {out}")
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
