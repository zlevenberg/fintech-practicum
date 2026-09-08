"""V2 realized index, fixed basket, forecasting, backtesting, and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from parts_inflation.aggregate import aggregate_same_day
from parts_inflation.config import CompositeWeighting, ResolvedConfig
from parts_inflation.forecast import (
    ForecastCandidate,
    build_forecast_candidates,
    select_forecast_by_backtest,
)
from parts_inflation.matched_pairs import build_adjacent_pairs, flag_extreme_pairs
from parts_inflation.repeat_sales import RepeatSalesResult, fit_repeat_sales


FORECAST_METHODS = ("trailing_12m_mean", "ewma", "mean_reversion", "damped_holt")


@dataclass
class ForecastOutputs:
    bucket_forecast: pd.DataFrame
    composite_forecast: pd.DataFrame
    monthly_paths: pd.DataFrame
    method_rows: pd.DataFrame


def last_complete_month_end(base_date: pd.Timestamp) -> pd.Timestamp:
    """Return the latest fully observed calendar month at a cutoff."""
    base = pd.Timestamp(base_date).normalize()
    if base.is_month_end:
        return base
    return base.to_period("M").start_time - pd.Timedelta(days=1)


def build_realized_daily(
    classified: pd.DataFrame,
    config: ResolvedConfig,
    base_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a realized-only daily panel through the forecast origin."""
    cutoff = pd.Timestamp(base_date)
    use = classified.loc[
        classified["in_direct_costs"].fillna(False)
        & classified["historical_unit_price"].fillna(0).gt(0)
        & classified["qty_received"].fillna(0).gt(0)
        & pd.to_datetime(classified["effective_historical_date"]).le(cutoff)
    ].copy()
    if use.empty:
        return pd.DataFrame()
    use["po_date"] = pd.to_datetime(use["effective_historical_date"])
    use["price"] = use["historical_unit_price"]
    use["qty"] = use["qty_received"]
    use["po_value"] = use["realized_spend"]
    use["model_eligible"] = True
    use["included_for_pricing"] = True
    use["is_open_order"] = False
    return aggregate_same_day(use, config)


def build_pairs(daily: pd.DataFrame, config: ResolvedConfig) -> pd.DataFrame:
    pairs = build_adjacent_pairs(daily)
    if pairs.empty:
        return pairs
    pairs = pairs.loc[
        pairs["delta_days"].ge(config.controls.historical_min_pair_gap_days)
    ].reset_index(drop=True)
    return flag_extreme_pairs(
        pairs,
        config.controls.extreme_ratio_low,
        config.controls.extreme_ratio_high,
        config.controls.extreme_ratio_max_days,
    )


def latest_complete_fiscal_year(base_date: pd.Timestamp) -> int:
    d = pd.Timestamp(base_date)
    if d.month == 9 and d.day == 30:
        return d.year
    current_fy = d.year + 1 if d.month >= 10 else d.year
    return current_fy - 1


def _fiscal_year(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return pd.Series(np.where(d.dt.month >= 10, d.dt.year + 1, d.dt.year), index=dates.index)


def build_bucket_weights(
    classified: pd.DataFrame,
    base_date: pd.Timestamp,
    config: ResolvedConfig,
) -> pd.DataFrame:
    """Create fixed bucket weights from a planned basket or latest complete FY."""
    base = pd.Timestamp(base_date)
    planned = config.planned_basket.copy() if config.planned_basket is not None else pd.DataFrame()
    if config.controls.composite_weighting == CompositeWeighting.planned_basket and not planned.empty:
        planned.columns = [str(c).strip() for c in planned.columns]
        if "PartKey" not in planned.columns:
            raise ValueError("Planned Basket requires PartKey")
        planned["PartKey"] = planned["PartKey"].astype(str).str.strip().str.upper()
        meta = (
            classified.loc[
                classified["in_direct_costs"].fillna(False)
                & classified["PartKey"].notna()
                & pd.to_datetime(classified["effective_historical_date"]).le(base)
            ]
            .sort_values("effective_historical_date")
            .groupby("PartKey", as_index=False)
            .tail(1)[["PartKey", "direct_cost_category", "historical_unit_price"]]
        )
        basket = planned.merge(meta, on="PartKey", how="left")
        expected_spend = pd.to_numeric(
            basket.get("ExpectedSpend", pd.Series(np.nan, index=basket.index)),
            errors="coerce",
        )
        expected_qty = pd.to_numeric(
            basket.get("ExpectedQuantity", pd.Series(np.nan, index=basket.index)),
            errors="coerce",
        )
        basket["basket_value"] = expected_spend.where(
            expected_spend.gt(0), expected_qty * basket["historical_unit_price"]
        )
        source = "planned_basket"
        fy = np.nan
    else:
        fy = latest_complete_fiscal_year(base)
        eligible = classified.loc[
            classified["in_direct_costs"].fillna(False)
            & classified["realized_spend"].fillna(0).gt(0)
            & pd.to_datetime(classified["effective_historical_date"]).le(base)
        ].copy()
        eligible["fiscal_year"] = _fiscal_year(eligible["effective_historical_date"])
        basket = eligible.loc[eligible["fiscal_year"].eq(fy)].copy()
        if basket.empty:
            basket = eligible.loc[
                pd.to_datetime(eligible["effective_historical_date"]).gt(base - pd.Timedelta(days=365))
            ].copy()
            source = "trailing_12m_realized_spend_proxy"
        else:
            source = f"FY{fy}_realized_spend_proxy"
        basket["basket_value"] = basket["realized_spend"]

    weights = (
        basket.dropna(subset=["direct_cost_category"])
        .groupby("direct_cost_category", as_index=False)
        .agg(basket_value=("basket_value", "sum"), basket_rows=("direct_cost_category", "size"))
    )
    weights = weights.loc[weights["basket_value"].gt(0)].copy()
    total = float(weights["basket_value"].sum())
    if total <= 0:
        raise ValueError("No positive direct-cost basket value is available")
    weights["weight"] = weights["basket_value"] / total
    weights["basket_source"] = source
    weights["base_fiscal_year"] = fy
    return weights.sort_values("direct_cost_category").reset_index(drop=True)


def historical_index_tables(
    model: RepeatSalesResult,
    bucket_weights: pd.DataFrame,
    pairs: pd.DataFrame,
    base_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return monthly index levels, FY/TTM rates, and category coverage."""
    if not model.months or len(model.delta0) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    weight_map = bucket_weights.set_index("direct_cost_category")["weight"].to_dict()
    rows: list[dict] = []
    bucket_levels: dict[str, np.ndarray] = {}
    overall_levels = np.exp(np.cumsum(model.delta0))
    overall_levels = 100.0 * overall_levels / overall_levels[0]
    for category in sorted(weight_map):
        rates = model.category_monthly(category)
        levels = np.exp(np.cumsum(rates))
        levels = 100.0 * levels / levels[0]
        bucket_levels[category] = levels
        for month, rate, level in zip(model.months, rates, levels):
            rows.append(
                {
                    "month": str(month),
                    "month_end": month.to_timestamp("M"),
                    "series": category,
                    "monthly_log_change": float(rate),
                    "monthly_rate": float(np.expm1(rate)),
                    "index": float(level),
                }
            )
    aligned_weights = {c: w for c, w in weight_map.items() if c in bucket_levels}
    weight_sum = sum(aligned_weights.values())
    if weight_sum <= 0:
        composite = overall_levels
    else:
        composite = sum((w / weight_sum) * bucket_levels[c] for c, w in aligned_weights.items())
    for month, level in zip(model.months, composite):
        rows.append(
            {
                "month": str(month), "month_end": month.to_timestamp("M"),
                "series": "Combined Direct Costs", "monthly_log_change": np.nan,
                "monthly_rate": np.nan, "index": float(level),
            }
        )
    for month, rate, level in zip(model.months, model.delta0, overall_levels):
        rows.append(
            {
                "month": str(month), "month_end": month.to_timestamp("M"),
                "series": "Overall Direct Fit", "monthly_log_change": float(rate),
                "monthly_rate": float(np.expm1(rate)), "index": float(level),
            }
        )
    monthly = pd.DataFrame(rows).sort_values(["series", "month_end"])
    complete_month_end = last_complete_month_end(base_date)
    monthly["period_status"] = np.where(
        monthly["month_end"].le(complete_month_end), "complete", "partial"
    )

    rate_rows: list[dict] = []
    for series, group in monthly.groupby("series", sort=True):
        group = group.sort_values("month_end").set_index("month_end")
        for year in sorted(group.index.year.unique()):
            end = pd.Timestamp(year, 9, 30)
            prev = pd.Timestamp(year - 1, 9, 30)
            cur_rows = group.loc[group.index <= end]
            prev_rows = group.loc[group.index <= prev]
            if cur_rows.empty or prev_rows.empty:
                continue
            cur_date = cur_rows.index[-1]
            prev_date = prev_rows.index[-1]
            if cur_date.year != year or cur_date.month != 9 or prev_date.month != 9:
                continue
            rate_rows.append(
                {
                    "series": series,
                    "period": f"FY{year}",
                    "period_type": "complete_fiscal_year",
                    "start_date": prev_date,
                    "end_date": cur_date,
                    "inflation_rate": float(cur_rows.iloc[-1]["index"] / prev_rows.iloc[-1]["index"] - 1),
                }
            )
        base = complete_month_end
        prior = base - pd.DateOffset(years=1)
        cur_rows = group.loc[group.index <= base]
        prev_rows = group.loc[group.index <= prior]
        if not cur_rows.empty and not prev_rows.empty:
            rate_rows.append(
                {
                    "series": series,
                    "period": f"TTM through {base.date()}",
                    "period_type": "trailing_12_months",
                    "start_date": prev_rows.index[-1],
                    "end_date": cur_rows.index[-1],
                    "inflation_rate": float(cur_rows.iloc[-1]["index"] / prev_rows.iloc[-1]["index"] - 1),
                }
            )
    rates = pd.DataFrame(rate_rows)

    coverage_rows: list[dict] = []
    total_spend = float(pairs["spend_b"].fillna(0).clip(lower=0).sum()) if not pairs.empty else 0.0
    for category in sorted(weight_map):
        group = pairs.loc[pairs["approved_category"].astype(str).eq(category)]
        spend = float(group["spend_b"].fillna(0).clip(lower=0).sum()) if not group.empty else 0.0
        coverage_rows.append(
            {
                "category": category,
                "pair_count": int(len(group)),
                "parts_with_pairs": int(group["PartKey"].nunique()) if not group.empty else 0,
                "ending_pair_spend": spend,
                "share_of_pair_spend": spend / total_spend if total_spend > 0 else np.nan,
                "basket_weight": weight_map.get(category, 0.0),
                "model_source": "category" if category in model.categories else "overall_fallback",
            }
        )
    return monthly, rates, pd.DataFrame(coverage_rows)


def _candidate_by_name(
    rates: np.ndarray,
    months: list[pd.Period],
    base_date: pd.Timestamp,
    horizon_months: int,
    method: Optional[str] = None,
) -> tuple[ForecastCandidate, str, pd.DataFrame]:
    target = pd.Timestamp(base_date) + pd.DateOffset(months=horizon_months)
    selected, table = select_forecast_by_backtest(rates, months, [12])
    if method:
        selected = method
    candidates = build_forecast_candidates(rates, months, target, pd.Timestamp(base_date))
    candidate = next((c for c in candidates if c.name == selected), candidates[0])
    return candidate, selected, table


def build_forecasts(
    model: RepeatSalesResult,
    pairs: pd.DataFrame,
    bucket_weights: pd.DataFrame,
    base_date: pd.Timestamp,
    config: ResolvedConfig,
    committed_summary: Optional[pd.DataFrame] = None,
    forced_method: Optional[str] = None,
) -> ForecastOutputs:
    """Build interpretable annual and cumulative bucket/composite forecasts."""
    horizon = 36
    weight_map = bucket_weights.set_index("direct_cost_category")["weight"].to_dict()
    committed = (
        committed_summary.set_index("category") if committed_summary is not None and not committed_summary.empty else pd.DataFrame()
    )
    complete_month_end = last_complete_month_end(base_date)
    complete_mask = np.array([m.to_timestamp("M") <= complete_month_end for m in model.months])
    overall_history = model.delta0[complete_mask]
    history_months = [m for m, keep in zip(model.months, complete_mask) if keep]
    _, overall_method, overall_selection = _candidate_by_name(
        overall_history,
        history_months,
        base_date, horizon, forced_method
    )
    overall_long_run = float(np.mean(overall_history[-36:]) * 12) if len(overall_history) else 0.0
    method_rows = overall_selection.copy()
    if not method_rows.empty:
        method_rows["series"] = "Overall Direct Fit"
        method_rows["selected"] = method_rows["method"].eq(overall_method)

    annual_rows: list[dict] = []
    monthly_rows: list[dict] = []
    cumulative_by_category: dict[str, list[float]] = {}
    for category in sorted(weight_map):
        history = model.category_monthly(category)[complete_mask]
        candidate, method, selection = _candidate_by_name(
            history, history_months, base_date, horizon, forced_method
        )
        if not selection.empty:
            selection = selection.copy()
            selection["series"] = category
            selection["selected"] = selection["method"].eq(method)
            method_rows = pd.concat([method_rows, selection], ignore_index=True)
        internal_year1 = float(np.sum(candidate.monthly_rates[:12]))
        uncapped_internal_year1 = internal_year1
        annual_floor = config.controls.forecast_annual_log_floor
        annual_cap = config.controls.forecast_annual_log_cap
        internal_year1 = float(np.clip(internal_year1, annual_floor, annual_cap))
        long_run = float(np.mean(history[-36:]) * 12) if len(history) else overall_long_run
        # Shrink long-run category level toward overall according to repeat-pair support.
        n_pairs = int(pairs["approved_category"].astype(str).eq(category).sum())
        b = n_pairs / (n_pairs + max(float(config.controls.category_min_pairs), 1.0))
        long_run = b * long_run + (1.0 - b) * overall_long_run
        long_run = float(np.clip(long_run, annual_floor, annual_cap))
        year1 = internal_year1
        overlay_weight = 0.0
        committed_signal = np.nan
        if not committed.empty and category in committed.index:
            row = committed.loc[category]
            coverage = float(row.get("matched_open_value_coverage", 0.0) or 0.0)
            committed_signal = float(row.get("annualized_log_signal", np.nan))
            if np.isfinite(committed_signal) and coverage > 0:
                committed_signal = float(
                    np.clip(committed_signal, annual_floor, annual_cap)
                )
                overlay_weight = min(
                    config.controls.committed_overlay_max_weight,
                    config.controls.committed_overlay_max_weight
                    * coverage / max(config.controls.committed_full_weight_coverage, 1e-9),
                )
                year1 = overlay_weight * committed_signal + (1.0 - overlay_weight) * internal_year1
        rho = config.controls.forecast_mean_reversion_rho
        annual_logs = [
            year1,
            long_run + rho * (year1 - long_run),
            long_run + rho**2 * (year1 - long_run),
        ]
        annual_logs = [float(np.clip(x, annual_floor, annual_cap)) for x in annual_logs]
        cumulative = 1.0
        cumulative_by_category[category] = []
        for year, annual_log in enumerate(annual_logs, 1):
            annual_multiplier = float(np.exp(annual_log))
            cumulative *= annual_multiplier
            cumulative_by_category[category].append(cumulative)
            annual_rows.append(
                {
                    "category": category,
                    "forecast_year": year,
                    "annual_log_rate": annual_log,
                    "annual_rate": annual_multiplier - 1.0,
                    "annual_multiplier": annual_multiplier,
                    "cumulative_multiplier": cumulative,
                    "cumulative_escalation": cumulative - 1.0,
                    "selected_internal_method": method,
                    "uncapped_internal_year1_log_rate": uncapped_internal_year1,
                    "forecast_cap_applied": bool(
                        not np.isclose(uncapped_internal_year1, internal_year1)
                        or np.isclose(annual_log, annual_floor)
                        or np.isclose(annual_log, annual_cap)
                    ),
                    "pair_count": n_pairs,
                    "category_shrinkage_weight": b,
                    "committed_log_signal": committed_signal,
                    "committed_overlay_weight": overlay_weight if year == 1 else 0.0,
                }
            )
        # Monthly display path exactly reconciles to annual logs.
        for month_number in range(1, 37):
            year = (month_number - 1) // 12
            monthly_log = annual_logs[year] / 12.0
            monthly_rows.append(
                {
                    "category": category,
                    "forecast_month": month_number,
                    "month": str(pd.Timestamp(base_date).to_period("M") + month_number),
                    "monthly_log_rate": monthly_log,
                    "monthly_rate": float(np.expm1(monthly_log)),
                }
            )

    composite_rows: list[dict] = []
    previous = 1.0
    for year in range(1, 4):
        cumulative = sum(
            weight_map[category] * cumulative_by_category[category][year - 1]
            for category in cumulative_by_category
        )
        annual_multiplier = cumulative / previous
        composite_rows.append(
            {
                "series": "Combined Direct Costs",
                "forecast_year": year,
                "annual_rate": annual_multiplier - 1.0,
                "annual_multiplier": annual_multiplier,
                "cumulative_multiplier": cumulative,
                "cumulative_escalation": cumulative - 1.0,
                "basket_weight_coverage": sum(weight_map[c] for c in cumulative_by_category),
            }
        )
        previous = cumulative
    return ForecastOutputs(
        bucket_forecast=pd.DataFrame(annual_rows),
        composite_forecast=pd.DataFrame(composite_rows),
        monthly_paths=pd.DataFrame(monthly_rows),
        method_rows=method_rows,
    )


def _actual_matched_basket(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_months: int,
    config: Optional[ResolvedConfig] = None,
) -> pd.DataFrame:
    target = cutoff + pd.DateOffset(months=horizon_months)
    before = daily.loc[pd.to_datetime(daily["po_date"]).le(cutoff)].copy()
    after = daily.loc[
        pd.to_datetime(daily["po_date"]).ge(target - pd.Timedelta(days=90))
        & pd.to_datetime(daily["po_date"]).le(target + pd.Timedelta(days=90))
    ].copy()
    if before.empty or after.empty:
        return pd.DataFrame()
    base = before.sort_values("po_date").groupby("PartKey", as_index=False).tail(1)
    after["_target_distance_days"] = (
        pd.to_datetime(after["po_date"]) - target
    ).abs().dt.days
    actual = (
        after.sort_values(["PartKey", "_target_distance_days", "po_date"])
        .groupby("PartKey", as_index=False)
        .head(1)
    )
    out = base[["PartKey", "price", "spend", "approved_category"]].merge(
        actual[["PartKey", "price"]], on="PartKey", suffixes=("_base", "_actual")
    )
    out = out.loc[out["price_base"].gt(0) & out["price_actual"].gt(0)].copy()
    out["raw_actual_multiplier"] = out["price_actual"] / out["price_base"]
    log_relative = np.log(out["raw_actual_multiplier"])
    winsor_lower = config.controls.benchmark_winsor_lower if config else 0.01
    winsor_upper = config.controls.benchmark_winsor_upper if config else 0.99
    absolute_low = config.controls.extreme_ratio_low if config else 0.25
    absolute_high = config.controls.extreme_ratio_high if config else 4.0
    q_low, q_high = log_relative.quantile([winsor_lower, winsor_upper])
    lower = max(float(q_low), float(np.log(absolute_low)))
    upper = min(float(q_high), float(np.log(absolute_high)))
    out["actual_multiplier"] = np.exp(log_relative.clip(lower=lower, upper=upper))
    out["actual_extreme_flag"] = out["raw_actual_multiplier"].lt(absolute_low) | out[
        "raw_actual_multiplier"
    ].gt(absolute_high)
    out["weight"] = out["spend"].fillna(out["price_base"]).clip(lower=0)
    if not out.empty and out["weight"].sum() > 0:
        cap = float(out["weight"].quantile(0.95))
        out["weight"] = out["weight"].clip(upper=cap)
        out["pre_robust_weight"] = out["weight"] / out["weight"].sum()
        out.loc[out["actual_extreme_flag"], "weight"] *= 0.25
        out["weight"] /= out["weight"].sum()
    return out


def run_composite_backtest(
    daily: pd.DataFrame,
    classified: pd.DataFrame,
    config: ResolvedConfig,
    base_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Rolling-origin model selection using one composite error per cutoff."""
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame(), "trailing_12m_mean"
    min_date = pd.to_datetime(daily["po_date"]).min() + pd.DateOffset(months=18)
    last_cutoff = pd.Timestamp(base_date) - pd.DateOffset(months=12)
    cutoffs = pd.date_range(min_date, last_cutoff, freq="QE")
    cutoffs = cutoffs[-2:] if config.controls.fast_mode else cutoffs[-8:]
    detail: list[dict] = []
    for cutoff in cutoffs:
        train_daily = daily.loc[pd.to_datetime(daily["po_date"]).le(cutoff)].copy()
        train_pairs = build_pairs(train_daily, config)
        if len(train_pairs) < 50:
            continue
        model = fit_repeat_sales(train_pairs, config)
        if not model.converged:
            continue
        actual = _actual_matched_basket(daily, cutoff, 12, config)
        if actual.empty or actual["weight"].sum() <= 0:
            continue
        actual_composite = float(np.sum(actual["weight"] * actual["actual_multiplier"]))
        # Basket weights are cutoff-only; future records cannot affect them.
        weights = build_bucket_weights(classified.loc[pd.to_datetime(classified["effective_historical_date"]).le(cutoff)], cutoff, config)
        for method in FORECAST_METHODS:
            forecast = build_forecasts(model, train_pairs, weights, cutoff, config, forced_method=method)
            bucket_map = forecast.bucket_forecast.loc[
                forecast.bucket_forecast["forecast_year"].eq(1)
            ].set_index("category")["annual_multiplier"].to_dict()
            predicted = float(
                np.sum(
                    actual["weight"]
                    * actual["approved_category"].map(bucket_map).fillna(
                        np.exp(np.sum(model.delta0[-12:]))
                    )
                )
            )
            detail.append(
                {
                    "cutoff": cutoff,
                    "horizon_months": 12,
                    "method": method,
                    "matched_parts": int(len(actual)),
                    "actual_extreme_weight_share": float(
                        actual.loc[actual["actual_extreme_flag"], "pre_robust_weight"].sum()
                    ),
                    "actual_multiplier": actual_composite,
                    "predicted_multiplier": predicted,
                    "absolute_log_error": abs(np.log(predicted) - np.log(actual_composite)),
                    "percentage_point_error": abs((predicted - 1) - (actual_composite - 1)),
                    "signed_error": predicted - actual_composite,
                }
            )
    detail_df = pd.DataFrame(detail)
    if detail_df.empty:
        return detail_df, pd.DataFrame(), "trailing_12m_mean"
    summary = (
        detail_df.groupby("method", as_index=False)
        .agg(
            mean_absolute_log_error=("absolute_log_error", "mean"),
            mean_percentage_point_error=("percentage_point_error", "mean"),
            bias=("signed_error", "mean"),
            cutoffs=("cutoff", "nunique"),
            mean_matched_parts=("matched_parts", "mean"),
        )
        .sort_values(["mean_absolute_log_error", "mean_percentage_point_error", "method"])
    )
    selected = str(summary.iloc[0]["method"])
    summary["selected"] = summary["method"].eq(selected)
    return detail_df, summary, selected


def bootstrap_forecasts(
    pairs: pd.DataFrame,
    bucket_weights: pd.DataFrame,
    base_date: pd.Timestamp,
    config: ResolvedConfig,
    selected_method: str,
    committed_summary: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    """Entity-cluster bootstrap that refits the selected v2 estimator."""
    if pairs.empty:
        return pd.DataFrame(), {"success_rate": 0.0, "attempts": 0}
    rng = np.random.default_rng(config.controls.random_seed)
    entities = pairs["PartKey"].astype(str).unique()
    groups = pairs.groupby(pairs["PartKey"].astype(str), sort=False).indices
    attempts = max(2, int(config.controls.bootstrap_iterations))
    samples: list[dict] = []
    for iteration in range(attempts):
        if iteration == 0:
            sample = pairs.copy()
        else:
            draws = rng.choice(entities, size=len(entities), replace=True)
            chunks = []
            for draw_no, entity in enumerate(draws):
                chunk = pairs.iloc[groups[entity]].copy()
                chunk["PartKey"] = f"{entity}__BOOT{draw_no}"
                chunks.append(chunk)
            sample = pd.concat(chunks, ignore_index=True)
        model = fit_repeat_sales(sample, config, max_iter_override=30 if config.controls.fast_mode else 60)
        if not model.converged:
            continue
        forecast = build_forecasts(
            model, sample, bucket_weights, base_date, config,
            committed_summary=committed_summary, forced_method=selected_method,
        )
        for _, row in forecast.composite_forecast.iterrows():
            samples.append(
                {
                    "iteration": iteration,
                    "forecast_year": int(row["forecast_year"]),
                    "cumulative_multiplier": float(row["cumulative_multiplier"]),
                }
            )
    sample_df = pd.DataFrame(samples)
    if sample_df.empty:
        return sample_df, {"success_rate": 0.0, "attempts": attempts}
    summary = (
        sample_df.groupby("forecast_year")["cumulative_multiplier"]
        .quantile([
            config.controls.confidence_lower_quantile,
            0.5,
            config.controls.confidence_upper_quantile,
        ])
        .unstack()
        .reset_index()
    )
    summary.columns = ["forecast_year", "lower", "median", "upper"]
    successes = int(sample_df["iteration"].nunique())
    return summary, {
        "success_rate": successes / attempts,
        "successful_fits": successes,
        "attempts": attempts,
        "method": "comparison_entity_cluster_bootstrap_refit",
    }
