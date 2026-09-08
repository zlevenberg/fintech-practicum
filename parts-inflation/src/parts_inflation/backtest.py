"""Rolling-origin backtests and model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from parts_inflation.aggregate import aggregate_same_day
from parts_inflation.benchmarks import (
    last_price_forecast,
    matched_part_log_changes,
    period_part_prices,
    tornqvist_index,
)
from parts_inflation.config import ResolvedConfig
from parts_inflation.forecast import build_forecast_candidates, select_forecast_by_backtest
from parts_inflation.hierarchy import compute_part_residuals, multiplier_part
from parts_inflation.matched_pairs import build_adjacent_pairs
from parts_inflation.repeat_sales import fit_repeat_sales

logger = logging.getLogger(__name__)


def wape(y_true: np.ndarray, y_pred: np.ndarray, qty: np.ndarray) -> float:
    denom = np.sum(qty * y_true)
    if denom <= 0:
        return float("nan")
    return float(np.sum(qty * np.abs(y_pred - y_true)) / denom)


def wale(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    mask = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    ww = w[mask]
    return float(np.sum(ww * np.abs(np.log(y_pred[mask]) - np.log(y_true[mask]))) / ww.sum())


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    mask = (y_true > 0) & (y_base > 0) & (y_pred > 0)
    if not mask.any():
        return float("nan")
    actual_dir = np.sign(y_true[mask] - y_base[mask])
    pred_dir = np.sign(y_pred[mask] - y_base[mask])
    return float(np.mean(actual_dir == pred_dir))


@dataclass
class BacktestResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    selected_model: str
    selection_rationale: str


def _cagr_multiplier(daily: pd.DataFrame, cutoff: pd.Timestamp, target: pd.Timestamp) -> float:
    d = daily.loc[daily["po_date"] <= cutoff]
    if d.empty:
        return 1.0
    pp = period_part_prices(d, "Q")
    tq = tornqvist_index(pp)
    if tq.empty:
        return 1.0
    ok = tq.loc[tq["status"].eq("ok")] if "status" in tq.columns else tq
    if len(ok) < 2 or "index" not in ok.columns:
        return 1.0
    years = max((cutoff - d["po_date"].min()).days / 365.25, 1 / 12)
    total = float(ok["index"].iloc[-1] / ok["index"].iloc[0])
    if total <= 0:
        return 1.0
    cagr = total ** (1 / years) - 1
    horizon_years = max((target - cutoff).days / 365.25, 0)
    return float((1 + cagr) ** horizon_years)


def _sample_future(future: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if len(future) <= max_n:
        return future
    # Prefer higher-spend observations
    f = future.copy()
    f["_w"] = f["spend"].fillna(f["price"] * f["qty"].fillna(1)).clip(lower=0)
    if f["_w"].sum() <= 0:
        return f.sample(n=max_n, random_state=seed)
    probs = f["_w"] / f["_w"].sum()
    idx = np.random.default_rng(seed).choice(f.index.to_numpy(), size=max_n, replace=False, p=probs.to_numpy())
    return f.loc[idx]


def run_backtests(
    classified: pd.DataFrame,
    config: ResolvedConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> BacktestResult:
    """Time-based rolling-origin validation without look-ahead."""
    ctrls = config.controls
    horizons = ctrls.horizon_months()
    if classified.loc[classified["model_eligible"]].empty:
        empty = pd.DataFrame()
        return BacktestResult(empty, empty, "last_price", "No eligible rows for backtest")

    daily_all = aggregate_same_day(classified, config)
    if daily_all.empty:
        empty = pd.DataFrame()
        return BacktestResult(empty, empty, "last_price", "No daily aggregates for backtest")

    daily_all["po_date"] = pd.to_datetime(daily_all["po_date"])
    min_date = daily_all["po_date"].min()
    max_date = daily_all["po_date"].max()
    start_cutoff = min_date + pd.DateOffset(months=18)
    if start_cutoff >= max_date - pd.DateOffset(months=min(horizons)):
        start_cutoff = min_date + pd.DateOffset(months=12)

    cutoffs = pd.date_range(
        start=start_cutoff, end=max_date - pd.DateOffset(months=min(horizons)), freq="QE"
    )
    if len(cutoffs) == 0:
        cutoffs = pd.DatetimeIndex([start_cutoff])

    max_eval = 800 if ctrls.fast_mode else 2500
    if ctrls.fast_mode:
        cutoffs = cutoffs[-2:] if len(cutoffs) > 2 else cutoffs

    detail_rows = []

    for cutoff in cutoffs:
        if progress:
            progress(f"Backtest cutoff {cutoff.date()}")
        train_daily = daily_all.loc[daily_all["po_date"] <= cutoff].copy()
        if train_daily.empty:
            continue
        train_pairs = build_adjacent_pairs(train_daily)
        hier = fit_repeat_sales(train_pairs, config)
        part_hier = compute_part_residuals(train_pairs, hier, config)
        part_hier_map = part_hier.set_index("PartKey") if not part_hier.empty else None

        last_prices = last_price_forecast(train_daily, cutoff)
        last_map = last_prices.set_index("PartKey")

        pp = period_part_prices(train_daily, "Q")
        matched = matched_part_log_changes(
            pp, ctrls.benchmark_winsor_lower, ctrls.benchmark_winsor_upper
        )
        recent_matched = (
            float(matched["capped_spend_weight_geom"].tail(4).mean())
            if not matched.empty
            else 0.0
        )
        tq = tornqvist_index(pp)
        if not tq.empty and "pct_change" in tq.columns and tq["pct_change"].notna().any():
            recent_tq = float(tq["pct_change"].dropna().tail(4).mean())
        else:
            recent_tq = 0.0
        # Convert average quarterly pct to annualized continuous rate for compounding
        matched_ann = np.log1p(recent_matched) * 4.0
        tq_ann = np.log1p(recent_tq) * 4.0

        best_fc, _ = select_forecast_by_backtest(hier.delta0, hier.months, [3, 6, 12])

        for horizon in horizons:
            target = cutoff + pd.DateOffset(months=horizon)
            future = daily_all.loc[
                (daily_all["po_date"] > cutoff) & (daily_all["po_date"] <= target)
            ]
            if future.empty:
                continue
            future = _sample_future(future, max_eval, ctrls.random_seed + int(horizon) + cutoff.year)

            # Vectorized base join
            fut = future.merge(
                last_map.reset_index()[["PartKey", "latest_price", "latest_date"]],
                on="PartKey",
                how="inner",
            )
            fut = fut.loc[pd.to_datetime(fut["latest_date"]) <= cutoff]
            if fut.empty:
                continue

            base_price = fut["latest_price"].astype(float).to_numpy()
            base_date = pd.to_datetime(fut["latest_date"])
            actual_price = fut["price"].astype(float).to_numpy()
            actual_date = pd.to_datetime(fut["po_date"])
            qty = fut["qty"].fillna(1).clip(lower=1e-9).astype(float).to_numpy()
            years = np.clip((actual_date - base_date).dt.days.to_numpy() / 365.25, 1e-6, None)

            # Single CAGR factor for this cutoff→horizon (avoid per-row index rebuilds)
            cagr_m = _cagr_multiplier(train_daily, cutoff, target)
            preds = {
                "last_price": base_price,
                "overall_cagr": base_price * cagr_m,
                "matched_part": base_price * np.exp(matched_ann * years),
                "tornqvist": base_price * np.exp(tq_ann * years),
            }

            # Hierarchical: precompute category multipliers for unique base/actual month pairs
            # then apply part residual analytically.
            cands = build_forecast_candidates(hier.delta0, hier.months, target, cutoff)
            cand = next((c for c in cands if c.name == best_fc), cands[0])
            from parts_inflation.hierarchy import multiplier_category

            cat_cache: dict[tuple, float] = {}
            hier_preds = np.empty(len(fut))
            sources: list[str] = []
            for i in range(len(fut)):
                part = fut.iloc[i]["PartKey"]
                bd = base_date.iloc[i]
                ad = actual_date.iloc[i]
                if part_hier_map is not None and part in part_hier_map.index:
                    prow = part_hier_map.loc[part]
                    cat = str(prow.get("category", "overall"))
                    key = (cat, bd.normalize(), ad.normalize())
                    if key not in cat_cache:
                        cat_cache[key] = multiplier_category(
                            hier, cat, bd, ad, cand.monthly_rates, cand.months
                        )
                    years_i = max((ad - bd).days / 365.25, 0.0)
                    resid = float(prow.get("shrunk_residual", 0.0) or 0.0)
                    m_i = cat_cache[key] * float(np.exp(resid * years_i))
                    src = prow.get("source", "category")
                    hier_preds[i] = base_price[i] * m_i
                    sources.append(src)
                else:
                    key = ("__overall__", bd.normalize(), ad.normalize())
                    if key not in cat_cache:
                        cat_cache[key] = multiplier_category(
                            hier, "overall", bd, ad, cand.monthly_rates, cand.months
                        )
                    hier_preds[i] = base_price[i] * cat_cache[key]
                    sources.append("overall")
            preds["hierarchical"] = hier_preds

            for model_name, pred in preds.items():
                for i in range(len(fut)):
                    detail_rows.append(
                        {
                            "cutoff": cutoff,
                            "horizon_months": horizon,
                            "PartKey": fut.iloc[i]["PartKey"],
                            "actual_date": actual_date.iloc[i],
                            "base_date": base_date.iloc[i],
                            "actual_price": actual_price[i],
                            "base_price": base_price[i],
                            "predicted_price": float(pred[i]),
                            "qty": qty[i],
                            "model": model_name,
                            "category": fut.iloc[i].get("approved_category"),
                            "fallback_source": sources[i]
                            if model_name == "hierarchical"
                            else model_name,
                        }
                    )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return BacktestResult(detail, pd.DataFrame(), "last_price", "No backtest evaluation rows")

    summary_rows = []
    for (model, horizon), grp in detail.groupby(["model", "horizon_months"]):
        yt = grp["actual_price"].to_numpy()
        yp = grp["predicted_price"].to_numpy()
        q = grp["qty"].to_numpy()
        w = q * yt
        summary_rows.append(
            {
                "model": model,
                "horizon_months": horizon,
                "n": len(grp),
                "WAPE": wape(yt, yp, q),
                "WALE": wale(yt, yp, w),
                "MdAPE": mdape(yt, yp),
                "directional_accuracy": directional_accuracy(
                    yt, yp, grp["base_price"].to_numpy()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    avg = summary.groupby("model")["WAPE"].mean().sort_values()
    order_complexity = {
        "last_price": 0,
        "overall_cagr": 1,
        "matched_part": 2,
        "tornqvist": 3,
        "hierarchical": 4,
    }
    best = str(avg.index[0])
    best_wape = float(avg.iloc[0])
    selected = best
    rationale = f"Lowest average WAPE: {best} ({best_wape:.4f})"
    material = ctrls.material_wape_improvement
    for cand in avg.index:
        if order_complexity.get(cand, 99) < order_complexity.get(best, 99):
            cand_wape = float(avg.loc[cand])
            if best_wape >= cand_wape * (1 - material):
                selected = cand
                rationale = (
                    f"Preferred simpler model {cand} (WAPE {cand_wape:.4f}) over {best} "
                    f"(WAPE {best_wape:.4f}); improvement below {material:.1%} material threshold"
                )
                break
    if ctrls.selected_model_mode.value != "best_backtest":
        selected = ctrls.selected_model_mode.value
        rationale = f"Forced by selected_model_mode={selected}"

    logger.info("Selected model: %s (%s)", selected, rationale)
    return BacktestResult(detail, summary, selected, rationale)
