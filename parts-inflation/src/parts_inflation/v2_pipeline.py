"""End-to-end v2 direct-cost inflation pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation import __version__
from parts_inflation.benchmarks import compute_all_benchmarks
from parts_inflation.classify import apply_cutoff_category_labels, apply_scope_and_category
from parts_inflation.clean import clean_po_lines
from parts_inflation.committed import build_committed_cost_indicator
from parts_inflation.config import ResolvedConfig, load_config, resolve_dates
from parts_inflation.ingest import load_all_po_lines
from parts_inflation.matching import propose_part_family_candidates
from parts_inflation.repeat_sales import fit_repeat_sales
from parts_inflation.v2_model import (
    bootstrap_forecasts,
    build_bucket_weights,
    build_forecasts,
    build_pairs,
    build_realized_daily,
    historical_index_tables,
    run_composite_backtest,
)
from parts_inflation.v2_report import write_v2_outputs


logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_config(config: ResolvedConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config.controls.model_dump(mode="json"), sort_keys=True).encode())
    for frame in [
        config.scope_mapping,
        config.category_mapping,
        config.part_overrides,
        config.planned_basket,
    ]:
        if frame is not None and not frame.empty:
            digest.update(frame.fillna("").astype(str).to_csv(index=False).encode())
    return digest.hexdigest()


def _data_quality(
    cleaned: pd.DataFrame,
    classified: pd.DataFrame,
    daily: pd.DataFrame,
    pairs: pd.DataFrame,
    ingest_warnings: list[str],
) -> pd.DataFrame:
    metrics: list[dict] = []

    def add(issue: str, mask: pd.Series, value_col: str = "po_value", note: str = "") -> None:
        use = cleaned.loc[mask.fillna(False)]
        values = pd.to_numeric(use.get(value_col, pd.Series(0, index=use.index)), errors="coerce").fillna(0)
        metrics.append(
            {"issue": issue, "row_count": int(len(use)), "value": float(values.sum()), "note": note}
        )

    add("raw_rows", pd.Series(True, index=cleaned.index))
    add("approved_direct_cost_rows", classified["in_direct_costs"])
    add("realized_direct_cost_rows", classified["in_direct_costs"] & classified["realized_spend"].fillna(0).gt(0), "realized_spend")
    add("open_or_partially_open_direct_cost_rows", classified["in_direct_costs"] & classified["has_open_commitment"], "remaining_open_spend")
    add("missing_part_key", cleaned["PartKey"].isna())
    add("invalid_or_missing_date", cleaned["effective_historical_date"].isna())
    add("duplicate_looking_rows", cleaned["is_exact_duplicate"], note="Preserved in official run; sensitivity required")
    add("po_value_reconciliation_failure", cleaned["po_value"].notna() & ~cleaned["po_value_matches_ordered"])
    add("received_extension_reconciliation_failure", cleaned["extension_received"].notna() & ~np.isclose(cleaned["received_reconciliation_ratio"], 1.0, rtol=1e-4, atol=1e-4))
    add("historical_cost_fallback", cleaned["historical_price_source"].eq("cost_fallback"), "realized_spend")
    metrics.extend(
        [
            {"issue": "realized_part_days", "row_count": int(len(daily)), "value": float(daily.get("spend", pd.Series(dtype=float)).sum()), "note": ""},
            {"issue": "repeat_pairs", "row_count": int(len(pairs)), "value": float(pairs.get("spend_b", pd.Series(dtype=float)).sum()), "note": "Consecutive observed purchases only"},
            {"issue": "extreme_pairs", "row_count": int(pairs.get("extreme_flag", pd.Series(dtype=bool)).sum()), "value": float(pairs.loc[pairs.get("extreme_flag", pd.Series(False, index=pairs.index)), "spend_b"].sum()) if not pairs.empty else 0.0, "note": "Downweighted, not used to remove whole parts"},
        ]
    )
    for warning in ingest_warnings:
        metrics.append({"issue": "ingest_warning", "row_count": np.nan, "value": np.nan, "note": warning})
    basis = (
        cleaned.groupby("po_likely_basis_factor", dropna=False)
        .size().reset_index(name="row_count")
    )
    for _, row in basis.iterrows():
        metrics.append(
            {"issue": f"po_reconciliation_basis_{row['po_likely_basis_factor']}", "row_count": int(row["row_count"]), "value": np.nan, "note": "Observed PO Value/(Cost*Qty Ordered) factor"}
        )
    return pd.DataFrame(metrics)


def _exclusions(classified: pd.DataFrame) -> pd.DataFrame:
    work = classified.copy()
    work["v2_exclusion_reason"] = np.select(
        [
            ~work["in_direct_costs"].fillna(False),
            work["PartKey"].isna(),
            work["effective_historical_date"].isna(),
            work["qty_received"].fillna(0).le(0),
            work["historical_unit_price"].fillna(0).le(0),
        ],
        [
            "outside_client_approved_direct_cost_scope",
            "missing_part_key",
            "invalid_or_missing_date",
            "no_received_quantity_realized_history",
            "invalid_realized_unit_price",
        ],
        default="included_in_realized_candidate_pool",
    )
    work["audit_value"] = np.where(
        work["v2_exclusion_reason"].eq("no_received_quantity_realized_history"),
        work["remaining_open_spend"].fillna(0),
        work["realized_spend"].fillna(work["po_value"]).fillna(0),
    )
    return (
        work.groupby("v2_exclusion_reason", as_index=False)
        .agg(row_count=("v2_exclusion_reason", "size"), value=("audit_value", "sum"))
        .sort_values("value", ascending=False)
    )


def _coverage_with_current_basket(
    coverage: pd.DataFrame,
    classified: pd.DataFrame,
    pairs: pd.DataFrame,
    base_date: pd.Timestamp,
) -> pd.DataFrame:
    paired = set(pairs["PartKey"].astype(str)) if not pairs.empty else set()
    base = pd.Timestamp(base_date)
    start = base - pd.Timedelta(days=365)
    current = classified.loc[
        classified["in_direct_costs"].fillna(False)
        & classified["realized_spend"].fillna(0).gt(0)
        & pd.to_datetime(classified["effective_historical_date"]).gt(start)
        & pd.to_datetime(classified["effective_historical_date"]).le(base)
    ].copy()
    rows: list[dict] = []
    for category, group in current.groupby("direct_cost_category", sort=True):
        total = float(group["realized_spend"].sum())
        matched = float(group.loc[group["PartKey"].astype(str).isin(paired), "realized_spend"].sum())
        rows.append(
            {"category": category, "current_realized_spend": total, "matched_current_spend": matched, "matched_spend_coverage": matched / total if total > 0 else np.nan}
        )
    current_cov = pd.DataFrame(rows)
    return coverage.merge(current_cov, on="category", how="outer")


def run_v2_pipeline(
    input_dir: Path,
    config_path: Path,
    output_dir: Path,
    target_date: Optional[str] = None,
    base_date: Optional[str] = None,
    scope_mode: Optional[str] = None,
    fast_mode: Optional[bool] = None,
    no_cache: bool = False,
    rebuild_cache: bool = False,
    skip_backtest: bool = False,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> Path:
    del no_cache, rebuild_cache  # v2 currently recomputes from immutable inputs every run.
    started = time.time()
    overrides = dict(cli_overrides or {})
    if base_date:
        overrides["base_date"] = base_date
    if target_date:
        overrides["target_date"] = target_date
    if scope_mode and scope_mode != "direct_costs":
        raise ValueError("The official v2 pipeline requires --scope-mode direct_costs")
    if fast_mode is not None:
        overrides["fast_mode"] = fast_mode
    # V2 invariants: the official headline is the approved six-bucket direct-
    # cost scope and realized history never treats an unreceived PO as actual.
    overrides.update(
        {
            "scope_mode": "direct_costs",
            "include_open_orders_as_prices": False,
            "include_open_orders_in_weights": False,
            "forecast_horizons_months": "12,24,36",
        }
    )
    config = load_config(Path(config_path), cli_overrides=overrides, create_if_missing=True)

    raw, infos, ingest_warnings = load_all_po_lines(Path(input_dir))
    input_hashes = {info.path.name: _hash_file(info.path) for info in infos}
    cleaned = clean_po_lines(raw, config)
    classified, scope_mapping = apply_scope_and_category(cleaned, config)
    latest = pd.to_datetime(classified["effective_historical_date"]).max().date()
    base_d, _ = resolve_dates(config.controls, latest)
    base_ts = pd.Timestamp(base_d)
    classified = apply_cutoff_category_labels(classified, base_ts)
    # From this point forward the working frame is a true as-of snapshot. This
    # governs diagnostics and mapping candidates as well as model estimation.
    classified = classified.loc[
        pd.to_datetime(classified["effective_historical_date"]).le(base_ts)
    ].copy()
    cleaned = classified.copy()

    daily = build_realized_daily(classified, config, base_ts)
    pairs = build_pairs(daily, config)
    if len(pairs) < 50:
        raise RuntimeError(f"Only {len(pairs)} valid repeat pairs; official estimation unavailable")
    model = fit_repeat_sales(pairs, config)
    if not model.converged and config.controls.fail_on_nonconvergence:
        raise RuntimeError("Identified repeat-sales model did not converge; no official output was published")

    bucket_weights = build_bucket_weights(classified, base_ts, config)
    historical_monthly, historical_rates, coverage = historical_index_tables(
        model, bucket_weights, pairs, base_ts
    )
    coverage = _coverage_with_current_basket(coverage, classified, pairs, base_ts)
    benchmarks = compute_all_benchmarks(
        daily,
        config.controls.benchmark_winsor_lower,
        config.controls.benchmark_winsor_upper,
    )
    committed_detail, committed_summary = build_committed_cost_indicator(
        classified,
        base_ts,
        minimum_gap_days=config.controls.historical_min_pair_gap_days,
        winsor_lower=config.controls.benchmark_winsor_lower,
        winsor_upper=config.controls.benchmark_winsor_upper,
    )
    backtest_detail, backtest_summary, selected_method = (
        (pd.DataFrame(), pd.DataFrame(), "mean_reversion")
        if skip_backtest
        else run_composite_backtest(daily, classified, config, base_ts)
    )
    forecasts = build_forecasts(
        model,
        pairs,
        bucket_weights,
        base_ts,
        config,
        committed_summary=committed_summary,
        forced_method=selected_method,
    )
    bootstrap_summary, bootstrap_meta = bootstrap_forecasts(
        pairs,
        bucket_weights,
        base_ts,
        config,
        selected_method,
        committed_summary,
    )
    composite_forecast = forecasts.composite_forecast.merge(
        bootstrap_summary, on="forecast_year", how="left"
    )
    bucket_forecast = forecasts.bucket_forecast.copy()
    bucket_forecast["series"] = bucket_forecast["category"]
    forecast_all = pd.concat(
        [composite_forecast.assign(category="Combined Direct Costs"), bucket_forecast],
        ignore_index=True,
        sort=False,
    )

    mapping_candidates = propose_part_family_candidates(classified)
    dq = _data_quality(cleaned, classified, daily, pairs, ingest_warnings)
    exclusions = _exclusions(classified)
    historical_ttm = historical_rates.loc[
        historical_rates["period_type"].eq("trailing_12_months")
    ].copy() if not historical_rates.empty else pd.DataFrame()

    warnings: list[str] = list(ingest_warnings) + list(model.warnings)
    min_coverage = config.controls.minimum_matched_spend_coverage
    low_coverage = coverage.loc[coverage["matched_spend_coverage"].fillna(0).lt(min_coverage)]
    if not low_coverage.empty:
        warnings.append(
            "Matched-spend coverage below threshold for: "
            + ", ".join(low_coverage["category"].dropna().astype(str))
        )
    if "receipt_date" not in raw.columns and classified["historical_date_proxy"].all():
        warnings.append("Receipt date unavailable; PO date is used as the realized-timing proxy")
    if "vendor_id" not in classified.columns or classified.get("vendor_id", pd.Series(dtype=object)).isna().all():
        warnings.append("Vendor identifier unavailable; vendor-level inflation and concentration cannot be measured")
    if bootstrap_meta.get("success_rate", 0.0) < 0.80:
        warnings.append("Fewer than 80% of bootstrap refits converged")

    final_input_hashes = {info.path.name: _hash_file(info.path) for info in infos}
    if final_input_hashes != input_hashes:
        raise RuntimeError("A source workbook changed while the pipeline was running; rerun from a stable snapshot")

    ttm_direct = np.nan
    if not historical_ttm.empty:
        match = historical_ttm.loc[historical_ttm["series"].eq("Combined Direct Costs")]
        if not match.empty:
            ttm_direct = float(match.iloc[-1]["inflation_rate"])
    year1 = float(composite_forecast.loc[composite_forecast["forecast_year"].eq(1), "annual_rate"].iloc[0])
    cumulative3 = float(composite_forecast.loc[composite_forecast["forecast_year"].eq(3), "cumulative_escalation"].iloc[0])
    overall_committed = committed_summary.loc[
        committed_summary["category"].eq("Overall Direct Costs")
    ] if not committed_summary.empty else pd.DataFrame()
    committed_rate = float(overall_committed.iloc[0]["annualized_rate_signal"]) if not overall_committed.empty else np.nan
    committed_coverage = float(overall_committed.iloc[0]["matched_open_value_coverage"]) if not overall_committed.empty else np.nan

    run_id = datetime.now(timezone.utc).strftime("v2_%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / run_id
    pair_columns = [
        "PartKey", "match_tier", "approved_category", "date_a", "date_b",
        "price_a", "price_b", "qty_a", "qty_b", "spend_a", "spend_b",
        "y", "delta_days", "delta_years", "x_q", "price_ratio", "extreme_flag",
    ]
    run_information = pd.DataFrame(
        [
            {"item": "run_id", "value": run_id},
            {"item": "run_timestamp_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"item": "package_version", "value": __version__},
            {"item": "python_version", "value": sys.version.split()[0]},
            {"item": "platform", "value": platform.platform()},
            {"item": "base_date", "value": str(base_d)},
            {"item": "selected_forecast_method", "value": selected_method},
            {"item": "model_converged", "value": model.converged},
            {"item": "pairs", "value": len(pairs)},
            {"item": "elapsed_seconds", "value": round(time.time() - started, 2)},
        ]
    )
    summary = {
        "run_id": run_id,
        "status": "succeeded" if model.converged else "diagnostic_only_nonconverged",
        "executive_metrics": {
            "Scope": "COS, Inventory, Operating Supplies, Production Supplies, Production Aids, Small Tooling",
            "Data through": str(base_d),
            "Realized TTM direct-cost inflation": ttm_direct,
            "Committed-cost annualized signal": committed_rate,
            "Committed matched-open-value coverage": committed_coverage,
            "Forecast year 1 inflation": year1,
            "Forecast cumulative 3-year escalation": cumulative3,
            "Selected forecast method": selected_method,
            "Repeat-purchase pairs": int(len(pairs)),
            "Parts with repeat pairs": int(pairs["PartKey"].nunique()),
            "Model converged": bool(model.converged),
            "Bootstrap success rate": bootstrap_meta.get("success_rate"),
            "Warnings / limitations": "; ".join(warnings) if warnings else "None",
        },
        "warnings": warnings,
        "bootstrap": bootstrap_meta,
        "input_hashes": input_hashes,
        "config_hash": _hash_config(config),
    }
    resolved_config = config.controls.model_dump(mode="json")
    tables = {
        "historical_monthly": historical_monthly,
        "historical_rates": historical_rates,
        "historical_ttm": historical_ttm,
        "forecast_all": forecast_all,
        "forecast_monthly": forecasts.monthly_paths,
        "forecast_methods": forecasts.method_rows,
        "committed_detail": committed_detail,
        "committed_summary": committed_summary,
        "coverage": coverage.merge(bucket_weights, left_on="category", right_on="direct_cost_category", how="outer"),
        "backtest_detail": backtest_detail,
        "backtest_summary": backtest_summary,
        "data_quality": dq,
        "exclusions": exclusions,
        "pair_audit": pairs[[c for c in pair_columns if c in pairs.columns]],
        "part_family_candidates": mapping_candidates,
        "scope_mapping": scope_mapping,
        "benchmark_monthly": benchmarks.get("tornqvist_monthly", pd.DataFrame()),
        "benchmark_quarterly": benchmarks.get("tornqvist_quarterly", pd.DataFrame()),
        "run_information": run_information,
    }
    manifest = {
        "run_id": run_id,
        "status": summary["status"],
        "input_hashes": input_hashes,
        "config_hash": summary["config_hash"],
        "model": "identified_interval_repeat_sales_v2",
        "selected_forecast_method": selected_method,
        "seed": config.controls.random_seed,
    }
    paths = write_v2_outputs(
        run_dir,
        {"summary": summary, "resolved_config": resolved_config, "tables": tables, "manifest": manifest},
    )
    logger.info("V2 pipeline completed in %.1fs: %s", time.time() - started, paths["excel"])
    return paths["excel"]
