"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation import __version__
from parts_inflation.aggregate import aggregate_for_scope, aggregate_same_day
from parts_inflation.backtest import run_backtests
from parts_inflation.benchmarks import compute_all_benchmarks, last_price_forecast
from parts_inflation.classify import apply_scope_and_category
from parts_inflation.clean import CLEANING_VERSION, clean_po_lines
from parts_inflation.config import (
    ResolvedConfig,
    invent_initial_scope_mapping,
    load_config,
    project_root,
    resolve_dates,
    write_config_workbook,
)
from parts_inflation.diagnostics import (
    build_data_quality_table,
    compare_to_expected_profile,
    profile_summary,
)
from parts_inflation.forecast import build_forecast_candidates, select_forecast_by_backtest
from parts_inflation.hierarchy import compute_part_residuals, multiplier_category, multiplier_part
from parts_inflation.ingest import combined_fingerprint, load_all_po_lines, profile_sources
from parts_inflation.matched_pairs import build_adjacent_pairs, flag_extreme_pairs
from parts_inflation.repeat_sales import fit_repeat_sales, select_lambdas_by_inner_backtest
from parts_inflation.report import write_results_workbook
from parts_inflation.uncertainty import bootstrap_composite_and_parts, widen_interval

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"parts_inflation_{ts}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear existing handlers for idempotent runs
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


def _cache_paths(cache_dir: Path, fp: str) -> dict[str, Path]:
    return {
        "cleaned": cache_dir / f"cleaned_{fp}.parquet",
        "classified": cache_dir / f"classified_{fp}.parquet",
        "daily": cache_dir / f"daily_{fp}.parquet",
        "pairs": cache_dir / f"pairs_{fp}.parquet",
        "meta": cache_dir / f"meta_{fp}.json",
    }


def init_config(input_dir: Path, output: Path) -> Path:
    raw, infos, warnings = load_all_po_lines(input_dir)
    cleaned = clean_po_lines(raw, load_config(create_if_missing=True))  # temporary defaults
    # Use invent mapping from cleaned bucket/description
    pairs = (
        cleaned.groupby(["Bucket", "Description"], dropna=False).size().reset_index(name="n")
    )
    mapping = invent_initial_scope_mapping(pairs)
    return write_config_workbook(output, scope_mapping=mapping)


def run_pipeline(
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
    t0 = time.time()
    timings: dict[str, float] = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)
    logger.info("Parts inflation pipeline starting (v%s)", __version__)

    overrides = dict(cli_overrides or {})
    if target_date:
        overrides["target_date"] = target_date
    if base_date:
        overrides["base_date"] = base_date
    if scope_mode:
        overrides["scope_mode"] = scope_mode
    if fast_mode is not None:
        overrides["fast_mode"] = fast_mode

    stage = time.time()
    # Seed config if missing using raw profile
    if not Path(config_path).exists():
        logger.info("Config missing; initializing from input data")
        init_config(Path(input_dir), Path(config_path))

    config = load_config(Path(config_path), cli_overrides=overrides, create_if_missing=True)
    raw, infos, ingest_warnings = load_all_po_lines(Path(input_dir))
    timings["ingest"] = time.time() - stage

    fp = combined_fingerprint(infos, CLEANING_VERSION + str(config.controls.model_dump()))
    cache_dir = project_root() / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir, fp)

    stage = time.time()
    use_cache = config.controls.cache_enabled and not no_cache and not rebuild_cache
    if use_cache and paths["classified"].exists() and paths["pairs"].exists():
        logger.info("Loading cached cleaned/classified/pairs (%s)", fp)
        classified = pd.read_parquet(paths["classified"])
        daily = pd.read_parquet(paths["daily"])
        pairs = pd.read_parquet(paths["pairs"])
        cleaned = classified  # already classified
        for frame in (classified, daily):
            if "po_date" in frame.columns:
                frame["po_date"] = pd.to_datetime(frame["po_date"], errors="coerce")
        for col in ("date_a", "date_b"):
            if col in pairs.columns:
                pairs[col] = pd.to_datetime(pairs[col], errors="coerce")
        from parts_inflation.classify import ensure_scope_mapping

        scope_mapping = ensure_scope_mapping(config, classified)
    else:
        cleaned = clean_po_lines(raw, config)
        classified, scope_mapping = apply_scope_and_category(cleaned, config)
        # Persist updated scope mapping into config if newly invented empty was filled
        if config.scope_mapping is None or config.scope_mapping.empty:
            write_config_workbook(Path(config_path), scope_mapping=scope_mapping, controls=config.controls)
            config = load_config(Path(config_path), cli_overrides=overrides)
            classified, scope_mapping = apply_scope_and_category(cleaned, config)

        daily = aggregate_same_day(classified, config)
        pairs = build_adjacent_pairs(daily)
        pairs = flag_extreme_pairs(
            pairs,
            config.controls.extreme_ratio_low,
            config.controls.extreme_ratio_high,
            config.controls.extreme_ratio_max_days,
        )
        if config.controls.cache_enabled and not no_cache:
            def _to_parquet(df: pd.DataFrame, path: Path) -> None:
                out = df.copy()
                for col in out.columns:
                    if out[col].dtype == object:
                        out[col] = out[col].map(
                            lambda x: None
                            if x is None or (isinstance(x, float) and pd.isna(x))
                            else str(x)
                        )
                out.to_parquet(path, index=False)

            _to_parquet(classified, paths["classified"])
            _to_parquet(daily, paths["daily"])
            _to_parquet(pairs, paths["pairs"])
            paths["meta"].write_text(json.dumps({"fingerprint": fp, "cleaning": CLEANING_VERSION}))
    timings["clean_classify_aggregate"] = time.time() - stage

    latest = pd.to_datetime(classified["po_date"]).max().date()
    base_d, target_d = resolve_dates(config.controls, latest)
    base_ts, target_ts = pd.Timestamp(base_d), pd.Timestamp(target_d)
    horizon_months = (target_ts - base_ts).days / 30.4375
    long_warn = (
        f"Horizon {horizon_months:.1f} months exceeds {config.controls.long_horizon_warning_months}; "
        "treat as scenario"
        if horizon_months > config.controls.long_horizon_warning_months
        else ""
    )

    # Benchmarks for selected scope + sensitivity scopes
    stage = time.time()
    logger.info("Computing benchmarks")
    benchmarks = compute_all_benchmarks(
        daily,
        config.controls.benchmark_winsor_lower,
        config.controls.benchmark_winsor_upper,
    )
    scope_sensitivity_rows = []
    for scope_name, col in [
        ("inventory_only", "in_inventory_only"),
        ("physical_inputs", "in_physical_inputs"),
        ("all_po_lines", "in_all_po_lines"),
        ("physical_plus_needs_review", "in_physical_plus_needs_review"),
    ]:
        dly = aggregate_for_scope(classified, col, config)
        if dly.empty:
            scope_sensitivity_rows.append(
                {"scope": scope_name, "status": "no_data", "match_count": 0}
            )
            continue
        bm = compute_all_benchmarks(
            dly, config.controls.benchmark_winsor_lower, config.controls.benchmark_winsor_upper
        )
        for freq, matched_key, tq_key in [
            ("monthly", "matched_monthly", "tornqvist_monthly"),
            ("quarterly", "matched_quarterly", "tornqvist_quarterly"),
            ("fiscal_year", "matched_fiscal_year", "tornqvist_fiscal_year"),
        ]:
            matched = bm.get(matched_key, pd.DataFrame())
            tq = bm.get(tq_key, pd.DataFrame())
            last_pct = (
                float(matched["capped_spend_weight_geom"].iloc[-1]) if not matched.empty else np.nan
            )
            last_tq = (
                float(tq["pct_change"].dropna().iloc[-1])
                if not tq.empty and "pct_change" in tq.columns and tq["pct_change"].notna().any()
                else np.nan
            )
            match_count = int(matched["match_count"].iloc[-1]) if not matched.empty and "match_count" in matched.columns else 0
            matched_spend = (
                float(matched["matched_spend"].iloc[-1])
                if not matched.empty and "matched_spend" in matched.columns
                else np.nan
            )
            scope_sensitivity_rows.append(
                {
                    "scope": scope_name,
                    "frequency": freq,
                    "status": "ok",
                    "part_days": len(dly),
                    "distinct_parts": dly["PartKey"].nunique(),
                    "latest_matched_spend_geom": last_pct,
                    "latest_tornqvist": last_tq,
                    "match_count": match_count,
                    "matched_spend": matched_spend,
                    "total_spend": float(dly["spend"].fillna(0).sum()),
                }
            )
    scope_sensitivity = pd.DataFrame(scope_sensitivity_rows)

    # Extremes with/without sensitivity on selected-scope quarterly matched/Törnqvist
    extremes_rows = []
    pairs_ext = flag_extreme_pairs(
        pairs,
        config.controls.extreme_ratio_low,
        config.controls.extreme_ratio_high,
        config.controls.extreme_ratio_max_days,
    ) if not pairs.empty else pairs
    for label, mask_fn in [
        ("with_extremes", lambda p: p),
        ("without_extremes", lambda p: p.loc[~p["extreme_flag"]] if "extreme_flag" in p.columns else p),
    ]:
        sub = mask_fn(pairs_ext)
        if sub.empty:
            extremes_rows.append({"variant": label, "status": "no_data"})
            continue
        # Build a pseudo period-price frame from pair endpoints for latest changes
        from parts_inflation.benchmarks import matched_part_log_changes, period_part_prices

        pp = period_part_prices(daily, "Q")
        if label == "without_extremes" and not pairs_ext.empty and "extreme_flag" in pairs_ext.columns:
            extreme_parts = set(pairs_ext.loc[pairs_ext["extreme_flag"], "PartKey"])
            # Drop parts that ever had an extreme adjacent pair in the window for sensitivity
            pp_f = pp.loc[~pp["PartKey"].isin(extreme_parts)]
        else:
            pp_f = pp
        matched = matched_part_log_changes(
            pp_f, config.controls.benchmark_winsor_lower, config.controls.benchmark_winsor_upper
        )
        from parts_inflation.benchmarks import tornqvist_index

        tq = tornqvist_index(pp_f)
        extremes_rows.append(
            {
                "variant": label,
                "status": "ok",
                "n_period_part_rows": len(pp_f),
                "latest_matched_spend_geom": float(matched["capped_spend_weight_geom"].iloc[-1])
                if not matched.empty
                else np.nan,
                "latest_tornqvist": float(tq["pct_change"].dropna().iloc[-1])
                if not tq.empty and tq["pct_change"].notna().any()
                else np.nan,
                "extreme_pair_count": int(pairs_ext["extreme_flag"].sum())
                if not pairs_ext.empty and "extreme_flag" in pairs_ext.columns
                else 0,
            }
        )
    extremes_sensitivity = pd.DataFrame(extremes_rows)
    timings["benchmarks"] = time.time() - stage

    # Hierarchical model with lambda grid selection
    stage = time.time()
    logger.info("Selecting hierarchical regularization via inner backtest")
    selected_lambdas, lambda_grid_table = select_lambdas_by_inner_backtest(
        pairs, config, progress=lambda m: logger.info(m)
    )
    # Persist selected lambdas onto controls for Controls Used visibility
    for k, v in selected_lambdas.items():
        key = f"lambda_{k}" if not k.startswith("lambda_") else k
        if hasattr(config.controls, key):
            setattr(config.controls, key, float(v))
            if key in config.sources:
                config.sources[key].value = float(v)
                config.sources[key].source = "InnerBacktest"
                config.sources[key].description = (
                    config.sources[key].description + " (selected by inner rolling WAPE)"
                )
    logger.info("Fitting hierarchical repeat-sales model")
    model = fit_repeat_sales(pairs, config, lambdas_override=selected_lambdas)
    part_hier = compute_part_residuals(pairs, model, config)
    best_fc, fc_table = select_forecast_by_backtest(
        model.delta0, model.months, config.controls.horizon_months()
    )
    cands = build_forecast_candidates(model.delta0, model.months, target_ts, base_ts)
    cand = next((c for c in cands if c.name == best_fc), cands[0])
    timings["hierarchical"] = time.time() - stage

    # Backtests
    stage = time.time()
    if skip_backtest:
        from parts_inflation.backtest import BacktestResult

        bt = BacktestResult(
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "model": "hierarchical",
                        "horizon_months": 12,
                        "n": 0,
                        "WAPE": np.nan,
                        "note": "skipped",
                    }
                ]
            ),
            "hierarchical",
            "Backtest skipped; using hierarchical model",
        )
    else:
        logger.info("Running rolling backtests")
        bt = run_backtests(
            classified, config, progress=lambda m: logger.info(m), lambdas_override=selected_lambdas
        )
    timings["backtest"] = time.time() - stage

    selected_model = bt.selected_model
    # Weights: trailing 12m PO value
    stage = time.time()
    weight_start = base_ts - pd.DateOffset(months=12)
    wdf = classified.loc[
        classified["model_eligible"]
        & classified["included_for_weights"]
        & (pd.to_datetime(classified["po_date"]) > weight_start)
        & (pd.to_datetime(classified["po_date"]) <= base_ts)
    ]
    if not config.planned_basket.empty and "PartKey" in config.planned_basket.columns:
        pb = config.planned_basket.copy()
        pb["PartKey"] = pb["PartKey"].astype(str).str.strip().str.upper()
        pb["ExpectedQuantity"] = pd.to_numeric(pb["ExpectedQuantity"], errors="coerce")
        planned = pb.dropna(subset=["PartKey", "ExpectedQuantity"])
    else:
        planned = pd.DataFrame()

    last = last_price_forecast(daily, base_ts)
    last = last.merge(
        part_hier[
            ["PartKey", "category", "source", "shrunk_residual", "lambda_shrink", "quality", "n_pairs", "span_days"]
        ],
        on="PartKey",
        how="left",
    )
    last["category"] = last["category"].fillna(last["approved_category"])
    last["source"] = last["source"].fillna("overall")
    last["shrunk_residual"] = last["shrunk_residual"].fillna(0.0)
    last["lambda_shrink"] = last["lambda_shrink"].fillna(0.0)
    last["quality"] = last["quality"].fillna(0.0)

    # Apply ManualCurrentPrice as latest observed price at base date
    mcp_map = (
        classified.dropna(subset=["PartKey"])
        .groupby("PartKey", as_index=False)["manual_current_price"]
        .max()
        if "manual_current_price" in classified.columns
        else pd.DataFrame(columns=["PartKey", "manual_current_price"])
    )
    last = last.merge(mcp_map, on="PartKey", how="left")
    has_mcp = last["manual_current_price"].notna() & (last["manual_current_price"] > 0)
    last.loc[has_mcp, "latest_price"] = last.loc[has_mcp, "manual_current_price"]
    last.loc[has_mcp, "latest_date"] = base_ts
    last.loc[has_mcp, "price_override_reason"] = "ManualCurrentPrice"
    if "price_override_reason" not in last.columns:
        last["price_override_reason"] = ""

    # Estimated base-date price via bridging (category-cached)
    cat_bridge_cache: dict[tuple, float] = {}
    base_prices = []
    staleness = []
    for _, r in last.iterrows():
        d_i = pd.Timestamp(r["latest_date"])
        p_i = float(r["latest_price"])
        if d_i >= base_ts:
            base_prices.append(p_i)
            staleness.append(0)
            continue
        cat = str(r.get("category") or r.get("approved_category") or "overall")
        key = (cat, d_i.normalize())
        if key not in cat_bridge_cache:
            cat_bridge_cache[key] = multiplier_category(model, cat, d_i, base_ts, None, None)
        resid = float(r.get("shrunk_residual", 0.0) or 0.0)
        years_i = (base_ts - d_i).days / 365.25
        m = cat_bridge_cache[key] * float(np.exp(resid * years_i))
        base_prices.append(p_i * m)
        staleness.append(int((base_ts - d_i).days))
    last["estimated_base_price"] = base_prices
    last["price_staleness_days"] = staleness

    # Manual future qty map
    mfq_map = (
        classified.dropna(subset=["PartKey"])
        .groupby("PartKey", as_index=False)["manual_future_quantity"]
        .max()
        if "manual_future_quantity" in classified.columns
        else pd.DataFrame(columns=["PartKey", "manual_future_quantity"])
    )
    last = last.merge(mfq_map, on="PartKey", how="left")

    # Composite weights (fixed basket q*)
    if not planned.empty:
        last = last.merge(planned[["PartKey", "ExpectedQuantity"]], on="PartKey", how="left")
        last["q_star"] = last["ExpectedQuantity"]
    else:
        spend_w = (
            wdf.groupby("PartKey", as_index=False)["po_value"]
            .sum()
            .rename(columns={"po_value": "trail_spend"})
        )
        last = last.merge(spend_w, on="PartKey", how="left")
        last["trail_spend"] = last["trail_spend"].fillna(0.0)
        last["q_star"] = np.where(
            last["estimated_base_price"] > 0,
            last["trail_spend"] / last["estimated_base_price"],
            last["qty"].fillna(0),
        )
    # ManualFutureQuantity overrides assumed future qty; also fills q_star if missing
    has_mfq = last["manual_future_quantity"].notna() & (last["manual_future_quantity"] > 0)
    last["q_future"] = last["q_star"]
    last.loc[has_mfq, "q_future"] = last.loc[has_mfq, "manual_future_quantity"]
    last.loc[has_mfq & (last["q_star"].isna() | (last["q_star"] <= 0)), "q_star"] = last.loc[
        has_mfq & (last["q_star"].isna() | (last["q_star"] <= 0)), "manual_future_quantity"
    ]

    last["base_spend_weight_num"] = last["q_star"].fillna(0) * last["estimated_base_price"].fillna(0)
    total_base = last["base_spend_weight_num"].sum()
    last["composite_weight"] = (
        last["base_spend_weight_num"] / total_base if total_base > 0 else 0.0
    )

    # Uncertainty / forecasts from hierarchical challenger
    logger.info("Computing part-cluster bootstrap uncertainty")
    weights = last.set_index("PartKey")["composite_weight"]
    weights = weights[weights > 0]
    unc = bootstrap_composite_and_parts(
        daily,
        pairs,
        last,
        weights,
        base_ts,
        target_ts,
        config,
        selected_forecast=best_fc,
        model=model,
        part_hier=part_hier,
        lambdas_override=selected_lambdas,
    )

    # Annualized rates for non-hierarchical per-part bridging
    matched_q = benchmarks.get("matched_quarterly", pd.DataFrame())
    tq = benchmarks.get("tornqvist_quarterly", pd.DataFrame())
    matched_ann = 0.0
    tq_ann = 0.0
    if not matched_q.empty and "capped_spend_weight_geom" in matched_q.columns:
        matched_ann = np.log1p(float(matched_q["capped_spend_weight_geom"].tail(4).mean())) * 4.0
    if not tq.empty and "pct_change" in tq.columns and tq["pct_change"].notna().any():
        tq_ann = np.log1p(float(tq["pct_change"].dropna().tail(4).mean())) * 4.0
    from parts_inflation.backtest import _cagr_multiplier

    cagr_m_full = _cagr_multiplier(daily, base_ts, target_ts)
    years_bt = max((target_ts - base_ts).days / 365.25, 1e-6)

    def _per_part_bridge_muls(model_name: str) -> dict[str, float]:
        muls = {}
        for _, r in last.iterrows():
            part = r["PartKey"]
            d_i = pd.Timestamp(r["latest_date"])
            years_i = max((target_ts - d_i).days / 365.25, 0.0)
            if model_name == "last_price":
                muls[part] = 1.0
            elif model_name == "matched_part":
                muls[part] = float(np.exp(matched_ann * years_i))
            elif model_name == "tornqvist":
                muls[part] = float(np.exp(tq_ann * years_i))
            elif model_name == "overall_cagr":
                # Scale full-horizon CAGR to part-specific horizon from d_i
                muls[part] = float(cagr_m_full ** (years_i / years_bt)) if years_bt > 0 else 1.0
            elif model_name == "category_benchmark":
                cat = str(r.get("category") or "overall")
                muls[part] = multiplier_category(
                    model, cat, d_i, target_ts, cand.monthly_rates, cand.months
                )
            else:
                muls[part] = unc["point_muls"].get(part, 1.0)
        return muls

    if selected_model in {
        "last_price",
        "matched_part",
        "tornqvist",
        "overall_cagr",
        "category_benchmark",
    }:
        point_muls = _per_part_bridge_muls(selected_model)
        ww = weights.reindex(point_muls.keys()).fillna(0)
        if ww.sum() > 0:
            ww = ww / ww.sum()
            point_comp = float(sum(ww[p] * point_muls[p] for p in ww.index))
        else:
            point_comp = 1.0
        # Use hierarchical bootstrap relative width around the selected point
        rel_lo = unc["composite_p10"] / max(unc["point_composite"], 1e-9)
        rel_hi = unc["composite_p90"] / max(unc["point_composite"], 1e-9)
        unc = dict(unc)
        unc["point_muls"] = point_muls
        unc["point_composite"] = point_comp
        unc["composite_p50"] = point_comp
        unc["composite_p10"] = point_comp * rel_lo
        unc["composite_p90"] = point_comp * rel_hi
        unc["part_p50"] = dict(point_muls)
        unc["part_p10"] = {p: m * rel_lo for p, m in point_muls.items()}
        unc["part_p90"] = {p: m * rel_hi for p, m in point_muls.items()}
        unc["method"] = f"selected_{selected_model}_per_part_bridge"
        last["source"] = selected_model
    elif selected_model == "blended":
        hier_muls = unc["point_muls"]
        matched_muls = _per_part_bridge_muls("matched_part")
        point_muls = {
            p: 0.5 * hier_muls.get(p, 1.0) + 0.5 * matched_muls.get(p, 1.0)
            for p in set(hier_muls) | set(matched_muls)
        }
        ww = weights.reindex(point_muls.keys()).fillna(0)
        ww = ww / ww.sum() if ww.sum() > 0 else ww
        point_comp = float(sum(ww[p] * point_muls[p] for p in ww.index)) if len(ww) else 1.0
        rel_lo = unc["composite_p10"] / max(unc["point_composite"], 1e-9)
        rel_hi = unc["composite_p90"] / max(unc["point_composite"], 1e-9)
        unc = dict(unc)
        unc["point_muls"] = point_muls
        unc["point_composite"] = point_comp
        unc["composite_p50"] = point_comp
        unc["composite_p10"] = point_comp * rel_lo
        unc["composite_p90"] = point_comp * rel_hi
        unc["part_p50"] = dict(point_muls)
        unc["part_p10"] = {p: m * rel_lo for p, m in point_muls.items()}
        unc["part_p90"] = {p: m * rel_hi for p, m in point_muls.items()}
        unc["method"] = "selected_blended_per_part_bridge"
        last["source"] = "blended"

    timings["uncertainty"] = time.time() - stage

    # Purchasing-cost projection (separate from fixed-basket inflation)
    # C(T0) = sum q_future * p_base; C(T) = sum q_future * p_base * M_i
    p50m_all = unc["part_p50"]
    c_base = float(
        (
            last["q_future"].fillna(0) * last["estimated_base_price"].fillna(0)
        ).sum()
    )
    c_target = 0.0
    for _, r in last.iterrows():
        part = r["PartKey"]
        m = p50m_all.get(part, unc["point_muls"].get(part, 1.0))
        c_target += float(r.get("q_future") or 0) * float(r.get("estimated_base_price") or 0) * float(m)
    purchasing_cost_change = (c_target / c_base - 1.0) if c_base > 0 else np.nan
    fixed_basket_cost_change = unc["composite_p50"] - 1.0

    # Part forecasts table — prioritize composite-weighted parts, cap huge exports
    part_rows = []
    p10m = unc["part_p10"]
    p50m = unc["part_p50"]
    p90m = unc["part_p90"]
    last_sorted = last.sort_values("composite_weight", ascending=False)
    max_parts = 20000 if not config.controls.fast_mode else 8000
    last_export = last_sorted.head(max_parts)
    # Source-line traces from latest classified observation
    trace = (
        classified.dropna(subset=["PartKey"])
        .sort_values("po_date")
        .groupby("PartKey", as_index=False)
        .tail(1)[
            [
                c
                for c in ["PartKey", "source_file", "source_sheet", "source_row_number", "Description 1", "Description 2", "raw_part_number"]
                if c in classified.columns
            ]
        ]
    )
    last_export = last_export.merge(trace, on="PartKey", how="left")

    for _, r in last_export.iterrows():
        part = r["PartKey"]
        m50 = p50m.get(part, unc["point_muls"].get(part, 1.0))
        m10 = p10m.get(part, m50 * 0.98)
        m90 = p90m.get(part, m50 * 1.02)
        m10, m50, m90 = widen_interval(
            m10, m50, m90, str(r["source"]), float(r["price_staleness_days"]), horizon_months
        )
        base_p = float(r["estimated_base_price"])
        part_rows.append(
            {
                "PartKey": part,
                "raw_part_number": r.get("raw_part_number"),
                "Description 1": r.get("Description 1"),
                "Description 2": r.get("Description 2"),
                "category": r.get("category"),
                "latest_observed_price": r["latest_price"],
                "latest_observed_date": r["latest_date"],
                "estimated_base_date_price": base_p,
                "price_staleness_days": r["price_staleness_days"],
                "assumed_quantity_q_star": r.get("q_star"),
                "future_quantity_q_T": r.get("q_future"),
                "target_price_p10": base_p * m10,
                "target_price_p50": base_p * m50,
                "target_price_p90": base_p * m90,
                "multiplier_p10": m10,
                "multiplier_p50": m50,
                "multiplier_p90": m90,
                "model_source": r["source"],
                "history_pairs": r.get("n_pairs"),
                "span_days": r.get("span_days"),
                "shrinkage_weight": r.get("lambda_shrink"),
                "quality": r.get("quality"),
                "composite_weight": r.get("composite_weight"),
                "source_file": r.get("source_file"),
                "source_sheet": r.get("source_sheet"),
                "source_row_number": r.get("source_row_number"),
                "price_override_reason": r.get("price_override_reason", ""),
            }
        )
    part_forecasts = pd.DataFrame(part_rows)

    # Category results from bootstrap category samples
    cat_p10 = unc.get("category_p10", {})
    cat_p50 = unc.get("category_p50", {})
    cat_p90 = unc.get("category_p90", {})
    cat_rows = []
    for cat in sorted(set(last["category"].dropna().astype(str))):
        m50 = cat_p50.get(
            cat,
            multiplier_category(model, cat, base_ts, target_ts, cand.monthly_rates, cand.months),
        )
        m10 = cat_p10.get(cat, m50 * 0.98)
        m90 = cat_p90.get(cat, m50 * 1.02)
        cat_rows.append(
            {
                "category": cat,
                "pair_count": int(pairs.loc[pairs["approved_category"] == cat].shape[0])
                if not pairs.empty
                else 0,
                "spend_trailing": float(
                    wdf.loc[wdf["approved_category"] == cat, "po_value"].sum()
                )
                if "approved_category" in wdf.columns
                else np.nan,
                "multiplier_p10": m10,
                "multiplier_p50": m50,
                "multiplier_p90": m90,
                "stability_flag": "ok"
                if (pairs["approved_category"] == cat).sum() >= config.controls.category_min_pairs
                else "sparse",
            }
        )
    category_results = pd.DataFrame(cat_rows)

    # Historical index: monthly, quarterly, annual series
    hist_frames = []
    for freq, matched_key, tq_key, fish_key in [
        ("monthly", "matched_monthly", "tornqvist_monthly", "fisher_monthly"),
        ("quarterly", "matched_quarterly", "tornqvist_quarterly", "fisher_quarterly"),
        ("fiscal_year", "matched_fiscal_year", "tornqvist_fiscal_year", "fisher_fiscal_year"),
    ]:
        tq = benchmarks.get(tq_key, pd.DataFrame()).copy()
        matched = benchmarks.get(matched_key, pd.DataFrame()).copy()
        fish = benchmarks.get(fish_key, pd.DataFrame()).copy()
        if tq.empty and matched.empty:
            continue
        frame = tq.copy() if not tq.empty else matched.copy()
        frame["frequency"] = freq
        if not matched.empty:
            cols = [c for c in ["period", "capped_spend_weight_geom", "match_count", "matched_spend", "equal_weight_geom", "median_pct"] if c in matched.columns]
            frame = frame.merge(matched[cols], on="period", how="outer", suffixes=("", "_matched"))
        if not fish.empty and "fisher" in fish.columns:
            frame = frame.merge(fish[["period", "fisher", "status"]].rename(columns={"status": "fisher_status"}), on="period", how="left")
        # Hierarchical overall index aligned to months when available
        if freq == "monthly" and model.n_pairs_used and len(model.delta0):
            hier_idx = np.cumprod(np.concatenate([[1.0], np.exp(model.delta0)]))
            hier_df = pd.DataFrame(
                {
                    "period": [str(m) for m in model.months],
                    "hierarchical_index": hier_idx[1:],
                    "hierarchical_log_change": model.delta0,
                }
            )
            frame = frame.merge(hier_df, on="period", how="left")
        hist_frames.append(frame)
    historical_index = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()

    # Chart data: prefer quarterly Törnqvist
    hist_q = benchmarks.get("tornqvist_quarterly", pd.DataFrame()).copy()
    chart_rows = []
    if not hist_q.empty and "index" in hist_q.columns:
        for _, r in hist_q.iterrows():
            if pd.isna(r.get("index")):
                continue
            chart_rows.append(
                {
                    "period": r["period"],
                    "index": r["index"],
                    "p10": r["index"],
                    "p50": r["index"],
                    "p90": r["index"],
                }
            )
        last_idx = float(hist_q["index"].dropna().iloc[-1])
    else:
        last_idx = 1.0
    chart_rows.append(
        {
            "period": f"Forecast {target_d}",
            "index": last_idx * unc["composite_p50"],
            "p10": last_idx * unc["composite_p10"],
            "p50": last_idx * unc["composite_p50"],
            "p90": last_idx * unc["composite_p90"],
        }
    )
    hist_chart = pd.DataFrame(chart_rows)
    cat_chart = category_results[["category", "multiplier_p50"]].copy() if not category_results.empty else pd.DataFrame()

    # Dashboard metrics
    years = max((target_ts - base_ts).days / 365.25, 1e-6)
    ann = unc["composite_p50"] ** (1 / years) - 1
    pct_part = float((last.loc[last["source"] == "part", "composite_weight"].sum()))
    pct_cat = float((last.loc[last["source"] == "category", "composite_weight"].sum()))
    pct_ov = float((last.loc[last["source"] == "overall", "composite_weight"].sum()))
    matched_cov = np.nan
    if not matched_q.empty and "matched_spend" in matched_q.columns:
        latest_end = pd.to_datetime(matched_q["period_end"].iloc[-1]) if "period_end" in matched_q.columns else None
        if latest_end is not None:
            window = daily.loc[
                (pd.to_datetime(daily["po_date"]) > latest_end - pd.Timedelta(days=92))
                & (pd.to_datetime(daily["po_date"]) <= latest_end)
            ]
            denom = float(window["spend"].fillna(0).sum())
            numer = float(matched_q["matched_spend"].iloc[-1])
            matched_cov = numer / denom if denom > 0 else np.nan

    controls_used = pd.DataFrame(
        [
            {
                "key": k,
                "value": str(s.value),
                "source": s.source,
                "description": s.description,
            }
            for k, s in config.sources.items()
        ]
    )
    # Append selected lambda keys explicitly
    for k, v in selected_lambdas.items():
        controls_used = pd.concat(
            [
                controls_used,
                pd.DataFrame(
                    [
                        {
                            "key": f"selected_lambda_{k}",
                            "value": str(v),
                            "source": "InnerBacktest",
                            "description": "Selected by inner rolling WAPE grid search",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    dq = build_data_quality_table(
        cleaned,
        classified,
        pairs,
        infos,
        ingest_warnings + model.warnings,
        base_date=base_ts,
        last_prices=last,
    )
    summary = profile_summary(classified, infos)
    profile_cmp = compare_to_expected_profile(summary)

    backtests_sheet = bt.summary.copy() if bt.summary is not None else pd.DataFrame()
    if not backtests_sheet.empty:
        backtests_sheet["selected"] = backtests_sheet["model"] == selected_model
        backtests_sheet["selection_rationale"] = bt.selection_rationale

    run_information = pd.DataFrame(
        [
            {"item": "run_timestamp", "value": datetime.now().isoformat(timespec="seconds")},
            {"item": "package_version", "value": __version__},
            {"item": "python_version", "value": sys.version.split()[0]},
            {"item": "platform", "value": platform.platform()},
            {"item": "log_file", "value": str(log_path)},
            {"item": "selected_model", "value": selected_model},
            {"item": "selection_rationale", "value": bt.selection_rationale},
            {"item": "forecast_method", "value": best_fc},
            {"item": "uncertainty_method", "value": unc.get("method")},
            {"item": "selected_lambdas", "value": json.dumps(selected_lambdas)},
            {"item": "fingerprint", "value": fp},
            {"item": "total_rows", "value": summary["total_rows"]},
            {"item": "distinct_parts", "value": summary["distinct_parts"]},
            {"item": "adjacent_pairs", "value": len(pairs)},
            {"item": "model_warnings", "value": "; ".join(model.warnings)},
            {"item": "fixed_basket_cost_change_p50", "value": fixed_basket_cost_change},
            {"item": "purchasing_cost_change_p50", "value": purchasing_cost_change},
            {"item": "elapsed_seconds", "value": round(time.time() - t0, 2)},
        ]
        + [{"item": f"timing_{k}", "value": round(v, 2)} for k, v in timings.items()]
        + [
            {
                "item": f"input_{info.path.name}",
                "value": f"rows={info.nrows}; size={info.size}; mtime={info.mtime}; fp={info.fingerprint}; "
                f"{info.min_date}..{info.max_date}",
            }
            for info in infos
        ]
    )

    try:
        import importlib.metadata as md

        for pkg in ["pandas", "numpy", "scipy", "openpyxl", "xlsxwriter", "pyarrow", "statsmodels", "scikit-learn"]:
            try:
                run_information = pd.concat(
                    [
                        run_information,
                        pd.DataFrame([{"item": f"dep_{pkg}", "value": md.version(pkg)}]),
                    ],
                    ignore_index=True,
                )
            except Exception:
                pass
    except Exception:
        pass

    # Attach extremes sensitivity into scope sensitivity sheet area via separate payload
    scope_sensitivity = pd.concat(
        [
            scope_sensitivity.assign(section="scope"),
            extremes_sensitivity.assign(section="extremes", scope="selected_scope"),
        ],
        ignore_index=True,
        sort=False,
    )

    payload = {
        "dashboard": {
            "scope": config.controls.scope_mode.value,
            "data_period": f"{summary['min_date']} to {summary['max_date']}",
            "base_date": str(base_d),
            "target_date": str(target_d),
            "composite_p10": unc["composite_p10"],
            "composite_p50": unc["composite_p50"],
            "composite_p90": unc["composite_p90"],
            "annualized_p50": ann,
            "cost_change_p50": fixed_basket_cost_change,
            "fixed_basket_inflation_p50": fixed_basket_cost_change,
            "purchasing_cost_change_p50": purchasing_cost_change,
            "matched_spend_coverage": matched_cov,
            "pct_weight_part": pct_part,
            "pct_weight_category": pct_cat,
            "pct_weight_overall": pct_ov,
            "selected_model": selected_model,
            "selection_rationale": bt.selection_rationale,
            "forecast_method": best_fc,
            "uncertainty_method": unc.get("method"),
            "long_horizon_warning": long_warn or "None",
        },
        "controls_used": controls_used,
        "scope_sensitivity": scope_sensitivity,
        "historical_index": historical_index,
        "historical_index_chart": hist_chart,
        "category_comparison_chart": cat_chart,
        "category_results": category_results,
        "part_forecasts": part_forecasts,
        "backtests": backtests_sheet,
        "scope_mapping": scope_mapping,
        "data_quality": dq,
        "run_information": run_information,
        "profile_comparison": profile_cmp,
        "forecast_selection": fc_table,
        "lambda_grid": lambda_grid_table,
    }

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / f"parts_inflation_results_{ts}.xlsx"
    write_results_workbook(out_path, payload)
    logger.info("Pipeline complete in %.1fs -> %s", time.time() - t0, out_path)
    return out_path
