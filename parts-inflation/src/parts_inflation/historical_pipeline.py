"""Historical actuals pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation import __version__
from parts_inflation.clean import CLEANING_VERSION, clean_po_lines
from parts_inflation.config import ResolvedConfig, ScopeMode, load_config, project_root
from parts_inflation.diagnostics import compare_to_expected_profile, profile_summary
from parts_inflation.historical_index import (
    IndexSettings,
    category_comparisons,
    compute_scope_comparisons,
    spend_reconciliation,
)
from parts_inflation.historical_periods import (
    DEFAULT_YTD_END,
    assign_fiscal_year,
)
from parts_inflation.historical_regression import RegressionSettings, run_repeat_purchase_regression
from parts_inflation.historical_report import write_historical_outputs
from parts_inflation.historical_scope import apply_historical_scope, scope_disclosure
from parts_inflation.ingest import combined_fingerprint, load_all_po_lines
from parts_inflation.pipeline import setup_logging

logger = logging.getLogger(__name__)

SCOPES = ("inventory_only", "physical_inputs", "all_po_lines")


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean: {v!r}")


def run_historical_actuals(
    input_dir: Path,
    config_path: Path,
    output_dir: Path,
    scope: str = "physical_inputs",
    winsor_lower: Optional[float] = None,
    winsor_upper: Optional[float] = None,
    weight_cap_quantile: Optional[float] = None,
    min_pair_gap_days: Optional[int] = None,
    include_open_orders: bool = True,
    fast: bool = False,
    no_cache: bool = False,
    ytd_end: Optional[date] = None,
) -> dict[str, Any]:
    """
    Run Historical Actual Inflation analysis for all scopes; emphasize ``scope`` in summary.
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)
    logger.info("Historical actuals starting (v%s)", __version__)

    ytd_end = ytd_end or DEFAULT_YTD_END
    overrides: dict[str, Any] = {
        "include_open_orders_as_prices": include_open_orders,
        "include_open_orders_in_weights": include_open_orders,
        "scope_mode": scope,
        "fast_mode": fast,
        "cache_enabled": not no_cache,
    }
    if not Path(config_path).exists():
        from parts_inflation.pipeline import init_config

        init_config(Path(input_dir), Path(config_path))

    config = load_config(Path(config_path), cli_overrides=overrides, create_if_missing=True)

    # Precedence: explicit CLI arg → config historical_* → code default
    winsor_lower = float(
        winsor_lower if winsor_lower is not None else config.controls.historical_winsor_lower
    )
    winsor_upper = float(
        winsor_upper if winsor_upper is not None else config.controls.historical_winsor_upper
    )
    weight_cap_quantile = float(
        weight_cap_quantile
        if weight_cap_quantile is not None
        else config.controls.historical_weight_cap_quantile
    )
    min_pair_gap_days = int(
        min_pair_gap_days
        if min_pair_gap_days is not None
        else config.controls.historical_min_pair_gap_days
    )

    # Fingerprints before any processing (for unchanged-input check)
    raw_paths = sorted(Path(input_dir).glob("*.xlsx")) + sorted(Path(input_dir).glob("*.xlsm"))
    pre_hashes = {p.name: _file_sha256(p) for p in raw_paths}

    stage = time.time()
    raw, infos, ingest_warnings = load_all_po_lines(Path(input_dir))
    logger.info("Ingested %s rows from %s files in %.1fs", len(raw), len(infos), time.time() - stage)

    cleaned = clean_po_lines(raw, config)
    cleaned = assign_fiscal_year(cleaned, "po_date")
    scoped = apply_historical_scope(cleaned, config)

    index_settings = IndexSettings(
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        weight_cap_quantile=weight_cap_quantile,
    )
    reg_settings = RegressionSettings(
        min_pair_gap_days=min_pair_gap_days,
        weight_cap_quantile=weight_cap_quantile,
    )

    scope_summaries = []
    scope_details = []
    chains = {}
    for sc in SCOPES:
        logger.info("Computing scope comparisons: %s", sc)
        summary, detail, chain = compute_scope_comparisons(scoped, sc, index_settings, ytd_end=ytd_end)
        scope_summaries.append(summary)
        if not detail.empty:
            scope_details.append(detail)
        chains[sc] = chain

    scope_results = pd.concat(scope_summaries, ignore_index=True) if scope_summaries else pd.DataFrame()
    matched_detail = pd.concat(scope_details, ignore_index=True) if scope_details else pd.DataFrame()

    logger.info("Computing category sensitivity (physical inputs)")
    category_results = category_comparisons(scoped, index_settings, ytd_end=ytd_end)

    logger.info("Spend reconciliation")
    spend_recon = spend_reconciliation(scoped)

    logger.info("Repeat-purchase Huber regression")
    regression = run_repeat_purchase_regression(scoped, reg_settings)

    # Method sensitivity table from physical scope summaries
    phys = scope_results.loc[scope_results["scope"] == "physical_inputs"].copy()
    method_rows = []
    for _, r in phys.iterrows():
        for method, col in [
            ("Robust Capped Törnqvist (headline)", "headline_robust_capped_tornqvist"),
            ("Winsorized Törnqvist", "winsorized_tornqvist"),
            ("Raw Törnqvist", "raw_tornqvist"),
            ("Capped geometric-spend", "capped_geometric_spend"),
            ("Equal-part winsorized", "equal_part_winsorized"),
            ("Equal-part raw", "equal_part_raw"),
            ("Similar-quantity equal winsorized", "similar_quantity_equal_winsorized"),
        ]:
            if col in r:
                method_rows.append(
                    {
                        "comparison": r["comparison"],
                        "method": method,
                        "inflation_rate": r[col],
                        "matched_parts": r.get("matched_parts"),
                        "coverage_a": r.get("coverage_a"),
                        "coverage_b": r.get("coverage_b"),
                    }
                )
    method_sensitivity = pd.DataFrame(method_rows)

    disclosure = scope_disclosure(scoped)
    summary_prof = profile_summary(scoped, infos)
    # Prefer fiscal-period part counts over file-based proxy
    summary_prof = _enrich_profile(scoped, summary_prof)
    profile_cmp = compare_to_expected_profile(summary_prof)

    warnings: list[str] = list(ingest_warnings)
    # Open-order note for FY2026
    fy26 = scoped.loc[scoped["fiscal_year"] == 2026]
    open_fy26 = (
        (fy26["qty_received"].fillna(0) <= 0)
        & (fy26["qty_ordered"].fillna(0) > 0)
        & (fy26["po_value"].fillna(0) > 0)
    )
    n_open = int(open_fy26.sum())
    if abs(n_open - 12339) > 500:
        warnings.append(f"FY2026 open-order-like rows={n_open} (expected ~12,339)")
    else:
        warnings.append(f"FY2026 rows with Qty Received=0, Qty Ordered>0, PO Value>0: {n_open}")

    # Post-run input integrity
    post_hashes = {p.name: _file_sha256(p) for p in raw_paths}
    if pre_hashes != post_hashes:
        warnings.append("CRITICAL: raw input file hashes changed during run")
    else:
        logger.info("Raw input files unchanged (hash check passed)")

    run_ts = datetime.now()
    payload = {
        "run_timestamp": run_ts,
        "headline_scope": scope,
        "ytd_end": ytd_end,
        "scope_results": scope_results,
        "category_results": category_results,
        "method_sensitivity": method_sensitivity,
        "regression_results": regression["grid"],
        "matched_part_detail": matched_detail,
        "spend_reconciliation": spend_recon,
        "scope_disclosure": disclosure,
        "chains": chains,
        "regression": regression,
        "profile_summary": summary_prof,
        "profile_comparison": profile_cmp,
        "infos": infos,
        "warnings": warnings,
        "settings": {
            "winsor_lower": winsor_lower,
            "winsor_upper": winsor_upper,
            "weight_cap_quantile": weight_cap_quantile,
            "min_pair_gap_days": min_pair_gap_days,
            "include_open_orders": include_open_orders,
            "fast": fast,
        },
        "log_path": str(log_path),
        "version": __version__,
        "platform": platform.platform(),
        "python": sys.version,
        "elapsed_seconds": time.time() - t0,
        "input_hashes": pre_hashes,
    }

    paths = write_historical_outputs(output_dir, payload, config)
    payload["output_paths"] = paths

    _print_terminal_summary(payload, paths)
    return payload


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _enrich_profile(scoped: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    """Count parts across fiscal periods (more accurate than source-file proxy)."""
    use = scoped.loc[scoped["PartKey"].notna() & scoped["fiscal_year"].notna()]
    part_fy = use.groupby("PartKey")["fiscal_year"].nunique()
    summary["parts_in_ge_2_fiscal_periods"] = int((part_fy >= 2).sum())
    summary["parts_in_ge_3_fiscal_periods"] = int((part_fy >= 3).sum())
    summary["parts_in_all_4_fiscal_periods"] = int((part_fy >= 4).sum())
    # Keep file-based keys for existing compare_to_expected_profile compatibility
    return summary


def _print_terminal_summary(payload: dict[str, Any], paths: dict[str, Path]) -> None:
    from rich.console import Console

    console = Console()
    console.print("\n[bold]Historical Actual Inflation[/bold]")
    console.print(f"Headline scope: {payload['headline_scope']}")
    for info in payload["infos"]:
        console.print(
            f"  {info.path.name}: {info.nrows:,} rows  "
            f"{info.min_date.date() if info.min_date is not None else '?'} → "
            f"{info.max_date.date() if info.max_date is not None else '?'}"
        )

    chain = payload["chains"].get("physical_inputs", {})
    console.print(
        f"Physical Robust Capped Törnqvist: "
        f"FY23→24={_pct(chain.get('fy2023_to_fy2024'))}  "
        f"FY24→25={_pct(chain.get('fy2024_to_fy2025'))}  "
        f"annualized={_pct(chain.get('annualized_fy2023_to_fy2025'))}  "
        f"FY26 YTD YoY={_pct(chain.get('fy2026_ytd_yoy'))}"
    )
    reg = payload["regression"]
    console.print(
        f"Repeat-purchase Huber: weighted={_pct(reg.get('weighted_annual_rate_no_trim'))}  "
        f"unweighted={_pct(reg.get('unweighted_annual_rate_no_trim'))}  "
        f"pairs={reg.get('n_pairs'):,}  "
        f"γ={reg.get('gamma_no_trim_weighted'):.4f}  "
        f"2×qty≈{_pct(reg.get('qty_doubling_no_trim_weighted'))}"
    )
    console.print(f"Warnings: {len(payload['warnings'])}")
    for k, p in paths.items():
        console.print(f"  {k}: {p}")


def _pct(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"
