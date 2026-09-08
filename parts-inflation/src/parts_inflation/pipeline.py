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
from parts_inflation.repeat_sales import fit_repeat_sales
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
        matched = bm.get("matched_quarterly", pd.DataFrame())
        tq = bm.get("tornqvist_quarterly", pd.DataFrame())
        last_pct = float(matched["capped_spend_weight_geom"].iloc[-1]) if not matched.empty else np.nan
        last_tq = float(tq["pct_change"].dropna().iloc[-1]) if not tq.empty and tq["pct_change"].notna().any() else np.nan
        scope_sensitivity_rows.append(
            {
                "scope": scope_name,
                "status": "ok",
                "part_days": len(dly),
                "distinct_parts": dly["PartKey"].nunique(),
                "latest_quarter_matched_spend_geom": last_pct,
                "latest_quarter_tornqvist": last_tq,
                "total_spend": float(dly["spend"].fillna(0).sum()),
            }
        )
    scope_sensitivity = pd.DataFrame(scope_sensitivity_rows)
    timings["benchmarks"] = time.time() - stage

    # Hierarchical model
    stage = time.time()
    logger.info("Fitting hierarchical repeat-sales model")
    model = fit_repeat_sales(pairs, config)
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
        bt = run_backtests(classified, config, progress=lambda m: logger.info(m))
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

    # Estimated base-date price via bridging (category-cached)
    from parts_inflation.hierarchy import multiplier_category

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

    # Composite weights
    if not planned.empty:
        last = last.merge(planned[["PartKey", "ExpectedQuantity"]], on="PartKey", how="left")
        last["q_star"] = last["ExpectedQuantity"].fillna(last["qty"])
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

    last["base_spend_weight_num"] = last["q_star"].fillna(0) * last["estimated_base_price"].fillna(0)
    total_base = last["base_spend_weight_num"].sum()
    last["composite_weight"] = (
        last["base_spend_weight_num"] / total_base if total_base > 0 else 0.0
    )

    # Uncertainty / forecasts from hierarchical challenger
    logger.info("Computing uncertainty bootstrap")
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
    )

    # If backtests select a simpler model, override published composite/part multipliers
    if selected_model == "last_price":
        unc = dict(unc)
        unc["composite_p10"] = 1.0
        unc["composite_p50"] = 1.0
        unc["composite_p90"] = 1.0
        unc["point_composite"] = 1.0
        unc["point_muls"] = {p: 1.0 for p in unc.get("point_muls", {})}
        unc["part_p10"] = {p: 1.0 for p in unc.get("part_p10", {})}
        unc["part_p50"] = {p: 1.0 for p in unc.get("part_p50", {})}
        unc["part_p90"] = {p: 1.0 for p in unc.get("part_p90", {})}
        unc["method"] = "selected_last_price_zero_inflation"
        last["forecast_source"] = "last_price"
        # Keep hierarchical fallback labels for weight-mix diagnostics
        last["model_source"] = last["source"]
    elif selected_model in {"matched_part", "tornqvist", "overall_cagr"}:
        # Approximate selected benchmark as constant annualized rate from latest quarterly series
        matched_q = benchmarks.get("matched_quarterly", pd.DataFrame())
        tq = benchmarks.get("tornqvist_quarterly", pd.DataFrame())
        years = max((target_ts - base_ts).days / 365.25, 1e-6)
        if selected_model == "matched_part" and not matched_q.empty:
            qrate = float(matched_q["capped_spend_weight_geom"].iloc[-1])
            m = (1 + qrate) ** (years * 4)
        elif selected_model == "tornqvist" and not tq.empty and tq["pct_change"].notna().any():
            qrate = float(tq["pct_change"].dropna().iloc[-1])
            m = (1 + qrate) ** (years * 4)
        else:
            m = float(unc["point_composite"])
        unc = dict(unc)
        unc["composite_p50"] = m
        unc["composite_p10"] = m * 0.95
        unc["composite_p90"] = m * 1.05
        unc["point_composite"] = m
        unc["point_muls"] = {p: m for p in unc.get("point_muls", {})}
        unc["part_p10"] = {p: m * 0.95 for p in unc.get("part_p10", {})}
        unc["part_p50"] = {p: m for p in unc.get("part_p50", {})}
        unc["part_p90"] = {p: m * 1.05 for p in unc.get("part_p90", {})}
        unc["method"] = f"selected_{selected_model}"
        last["source"] = selected_model

    timings["uncertainty"] = time.time() - stage

    # Part forecasts table — prioritize composite-weighted parts, cap huge exports
    part_rows = []
    p10m = unc["part_p10"]
    p50m = unc["part_p50"]
    p90m = unc["part_p90"]
    last_sorted = last.sort_values("composite_weight", ascending=False)
    max_parts = 20000 if not config.controls.fast_mode else 8000
    last_export = last_sorted.head(max_parts)
    years_bt = max((target_ts - base_ts).days / 365.25, 0.0)
    cat_fwd_cache: dict[str, float] = {}
    for _, r in last_export.iterrows():
        part = r["PartKey"]
        cat = str(r.get("category") or "overall")
        if cat not in cat_fwd_cache:
            cat_fwd_cache[cat] = multiplier_category(
                model, cat, base_ts, target_ts, cand.monthly_rates, cand.months
            )
        resid = float(r.get("shrunk_residual", 0.0) or 0.0)
        m_point = cat_fwd_cache[cat] * float(np.exp(resid * years_bt))
        m50 = p50m.get(part, unc["point_muls"].get(part, m_point))
        m10 = p10m.get(part, m50 * 0.98)
        m90 = p90m.get(part, m50 * 1.02)
        m10, m50, m90 = widen_interval(
            m10, m50, m90, str(r["source"]), float(r["price_staleness_days"]), horizon_months
        )
        base_p = float(r["estimated_base_price"])
        part_rows.append(
            {
                "PartKey": part,
                "category": r.get("category"),
                "latest_observed_price": r["latest_price"],
                "latest_observed_date": r["latest_date"],
                "estimated_base_date_price": base_p,
                "price_staleness_days": r["price_staleness_days"],
                "assumed_quantity": r.get("q_star"),
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
            }
        )
    part_forecasts = pd.DataFrame(part_rows)

    # Merge descriptions from classified
    desc = (
        classified.dropna(subset=["PartKey"])
        .sort_values("po_date")
        .groupby("PartKey", as_index=False)
        .tail(1)[["PartKey", "Description 1", "Description 2", "raw_part_number"]]
    )
    part_forecasts = part_forecasts.drop(columns=["Description 1"], errors="ignore").merge(
        desc, on="PartKey", how="left"
    )

    # Category results
    cat_rows = []
    for cat in sorted(set(last["category"].dropna().astype(str))):
        m50 = multiplier_category(
            model, cat, base_ts, target_ts, cand.monthly_rates, cand.months
        )
        # Bootstrap approx via overall composite noise scale
        spread = max(unc["composite_p90"] - unc["composite_p10"], 0.02)
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
                "multiplier_p10": m50 - 0.5 * spread,
                "multiplier_p50": m50,
                "multiplier_p90": m50 + 0.5 * spread,
                "stability_flag": "ok"
                if (pairs["approved_category"] == cat).sum() >= config.controls.category_min_pairs
                else "sparse",
            }
        )
    category_results = pd.DataFrame(cat_rows)

    # Historical index sheet
    hist_q = benchmarks.get("tornqvist_quarterly", pd.DataFrame()).copy()
    matched_q = benchmarks.get("matched_quarterly", pd.DataFrame()).copy()
    historical_index = hist_q
    if not matched_q.empty:
        historical_index = hist_q.merge(
            matched_q[["period", "capped_spend_weight_geom", "match_count", "matched_spend"]],
            on="period",
            how="outer",
            suffixes=("", "_matched"),
        )

    # Chart data: historical index + forecast fan
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
    # Append forecast points at target
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
        # Coverage vs total spend in the latest matched quarter's window
        latest_end = pd.to_datetime(matched_q["period_end"].iloc[-1]) if "period_end" in matched_q.columns else None
        if latest_end is not None:
            q_start = latest_end - pd.offsets.QuarterEnd(0) + pd.Timedelta(days=1) - pd.offsets.QuarterBegin(0)
            # Approximate quarter window: last 90 days before period end
            window = daily.loc[
                (pd.to_datetime(daily["po_date"]) > latest_end - pd.Timedelta(days=92))
                & (pd.to_datetime(daily["po_date"]) <= latest_end)
            ]
            denom = float(window["spend"].fillna(0).sum())
            numer = float(matched_q["matched_spend"].iloc[-1])
            matched_cov = numer / denom if denom > 0 else np.nan
        else:
            matched_cov = float(matched_q["match_count"].iloc[-1])

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

    dq = build_data_quality_table(cleaned, classified, pairs, infos, ingest_warnings + model.warnings)
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
            {"item": "fingerprint", "value": fp},
            {"item": "total_rows", "value": summary["total_rows"]},
            {"item": "distinct_parts", "value": summary["distinct_parts"]},
            {"item": "adjacent_pairs", "value": len(pairs)},
            {"item": "model_warnings", "value": "; ".join(model.warnings)},
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

    # Dependency versions
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
            "cost_change_p50": unc["composite_p50"] - 1.0,
            "matched_spend_coverage": matched_cov,
            "pct_weight_part": pct_part,
            "pct_weight_category": pct_cat,
            "pct_weight_overall": pct_ov,
            "selected_model": selected_model,
            "forecast_method": best_fc,
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
    }

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / f"parts_inflation_results_{ts}.xlsx"
    write_results_workbook(out_path, payload)
    logger.info("Pipeline complete in %.1fs -> %s", time.time() - t0, out_path)
    return out_path
