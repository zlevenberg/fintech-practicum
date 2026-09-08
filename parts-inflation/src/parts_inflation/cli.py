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
    help="Glenair direct-cost realized, committed, and forecast inflation",
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
    cfg = load_config(
        config,
        cli_overrides={
            "scope_mode": "direct_costs",
            "include_open_orders_as_prices": False,
            "include_open_orders_in_weights": False,
        },
        create_if_missing=True,
    )
    raw, infos, warnings = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, cfg)
    classified, mapping = apply_scope_and_category(cleaned, cfg)
    summary = profile_summary(classified, infos)
    cmp = compare_to_expected_profile(summary)
    console.print(cmp.to_string(index=False))
    console.print(f"Scope mapping rows: {len(mapping)}")
    console.print(f"Model-eligible rows: {int(classified['model_eligible'].sum()):,}")
    console.print(f"Approved direct-cost rows: {int(classified['in_direct_costs'].sum()):,}")
    console.print(
        f"Realized direct-cost rows: "
        f"{int((classified['in_direct_costs'] & classified['realized_spend'].fillna(0).gt(0)).sum()):,}"
    )
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
    """Run v2 rolling-origin fixed-basket composite backtests."""
    import pandas as pd

    from parts_inflation.classify import apply_cutoff_category_labels, apply_scope_and_category
    from parts_inflation.clean import clean_po_lines
    from parts_inflation.config import resolve_dates
    from parts_inflation.ingest import load_all_po_lines
    from parts_inflation.v2_model import build_realized_daily, run_composite_backtest

    setup_logging(_default_output())
    cfg = load_config(
        config,
        cli_overrides={
            "fast_mode": fast_mode,
            "scope_mode": "direct_costs",
            "include_open_orders_as_prices": False,
            "include_open_orders_in_weights": False,
        },
        create_if_missing=True,
    )
    raw, _, _ = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, cfg)
    classified, _ = apply_scope_and_category(cleaned, cfg)
    latest = pd.to_datetime(classified["effective_historical_date"]).max().date()
    base, _ = resolve_dates(cfg.controls, latest)
    classified = apply_cutoff_category_labels(classified, pd.Timestamp(base))
    daily = build_realized_daily(classified, cfg, pd.Timestamp(base))
    _, summary, selected = run_composite_backtest(daily, classified, cfg, pd.Timestamp(base))
    console.print(summary.to_string(index=False) if not summary.empty else "No results")
    console.print(f"Selected: {selected}")


@app.command("run")
def run_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
    config: Path = typer.Option(_default_config(), "--config"),
    output_dir: Path = typer.Option(_default_output(), "--output-dir"),
    target_date: Optional[str] = typer.Option(None, "--target-date", help="YYYY-MM-DD"),
    base_date: Optional[str] = typer.Option(None, "--base-date", help="YYYY-MM-DD"),
    scope_mode: Optional[str] = typer.Option(
        None, "--scope-mode", help="Official v2 accepts direct_costs only"
    ),
    fast_mode: bool = typer.Option(False, "--fast-mode"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    rebuild_cache: bool = typer.Option(False, "--rebuild-cache"),
    skip_backtest: bool = typer.Option(False, "--skip-backtest"),
    bootstrap_iterations: Optional[int] = typer.Option(None, "--bootstrap-iterations"),
) -> None:
    """Run v2 realized history, committed-cost signal, and 1/2/3-year forecasts."""
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


@app.command("propose-mappings")
def propose_mappings_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir"),
    config: Path = typer.Option(_default_config(), "--config"),
    output_dir: Path = typer.Option(_default_output(), "--output-dir"),
) -> None:
    """Export conservative, unofficial part-family candidates for client review."""
    from parts_inflation.classify import apply_scope_and_category
    from parts_inflation.clean import clean_po_lines
    from parts_inflation.ingest import load_all_po_lines
    from parts_inflation.matching import propose_part_family_candidates

    cfg = load_config(config, create_if_missing=True)
    raw, _, _ = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, cfg)
    classified, _ = apply_scope_and_category(cleaned, cfg)
    candidates = propose_part_family_candidates(classified)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "part_family_candidates.csv"
    candidates.to_csv(path, index=False)
    console.print(f"[green]Mapping candidates written to[/green] {path}")


@app.command("historical-actuals")
def historical_actuals_cmd(
    input_dir: Path = typer.Option(_default_input(), "--input-dir", help="Directory of PO workbooks"),
    config: Path = typer.Option(_default_config(), "--config"),
    output_dir: Path = typer.Option(_default_output(), "--output-dir"),
    scope: str = typer.Option(
        "physical_inputs",
        "--scope",
        help="Headline scope: physical_inputs|inventory_only|all_po_lines",
    ),
    winsor_lower: Optional[float] = typer.Option(None, "--winsor-lower"),
    winsor_upper: Optional[float] = typer.Option(None, "--winsor-upper"),
    weight_cap_quantile: Optional[float] = typer.Option(None, "--weight-cap-quantile"),
    min_pair_gap_days: Optional[int] = typer.Option(None, "--min-pair-gap-days"),
    include_open_orders: str = typer.Option(
        "false",
        "--include-open-orders",
        help="true|false — keep open orders as price observations",
    ),
    fast: bool = typer.Option(False, "--fast", help="Faster development path (same formulas)"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Run the retained legacy historical sensitivity workflow."""
    from parts_inflation.historical_pipeline import run_historical_actuals

    try:
        include = str(include_open_orders).strip().lower() in {"true", "1", "yes", "y"}
        payload = run_historical_actuals(
            input_dir=input_dir,
            config_path=config,
            output_dir=output_dir,
            scope=scope,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            weight_cap_quantile=weight_cap_quantile,
            min_pair_gap_days=min_pair_gap_days,
            include_open_orders=include,
            fast=fast,
            no_cache=no_cache,
        )
        excel = payload.get("output_paths", {}).get("excel")
        console.print(f"[green]Historical actuals written to[/green] {excel}")
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        raise typer.Exit(code=1) from exc

if __name__ == "__main__":
    app()
