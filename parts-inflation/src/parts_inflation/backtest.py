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
from parts_inflation.hierarchy import compute_part_residuals, multiplier_category
from parts_inflation.matched_pairs import build_adjacent_pairs
from parts_inflation.repeat_sales import fit_repeat_sales

logger = logging.getLogger(__name__)

COMPLEXITY_ORDER = {
    "last_price": 0,
    "overall_cagr": 1,
    "matched_part": 2,
    "tornqvist": 3,
    "category_benchmark": 4,
    "hierarchical": 5,
    "blended": 6,
}


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


def interval_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(p10) & np.isfinite(p90)
    if not mask.any():
        return float("nan")
    return float(np.mean((y_true[mask] >= p10[mask]) & (y_true[mask] <= p90[mask])))


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
    f = future.copy()
    f["_w"] = f["spend"].fillna(f["price"] * f["qty"].fillna(1)).clip(lower=0)
    if f["_w"].sum() <= 0:
        return f.sample(n=max_n, random_state=seed)
    probs = f["_w"] / f["_w"].sum()
    idx = np.random.default_rng(seed).choice(f.index.to_numpy(), size=max_n, replace=False, p=probs.to_numpy())
    return f.loc[idx]


def _composite_error(
    last_map: pd.DataFrame,
    fut: pd.DataFrame,
    pred_by_part: dict[str, float],
    actual_by_part: dict[str, float],
) -> float:
    """Fixed-basket composite multiplier error vs actual matched-basket change."""
    parts = [p for p in pred_by_part if p in actual_by_part and p in last_map.index]
    if not parts:
        return float("nan")
    base = last_map.loc[parts, "latest_price"].astype(float)
    # Weight by base spend using qty at cutoff if available else 1
    qty = last_map.loc[parts, "qty"].fillna(1).astype(float) if "qty" in last_map.columns else pd.Series(1.0, index=parts)
    w_num = (qty * base).clip(lower=0)
    if w_num.sum() <= 0:
        return float("nan")
    w = w_num / w_num.sum()
    m_hat = float(sum(w[p] * pred_by_part[p] / float(base[p]) for p in parts if base[p] > 0))
    m_act = float(sum(w[p] * actual_by_part[p] / float(base[p]) for p in parts if base[p] > 0))
    if m_act == 0:
        return float("nan")
    return abs(m_hat - m_act) / abs(m_act)


def run_backtests(
    classified: pd.DataFrame,
    config: ResolvedConfig,
    progress: Optional[Callable[[str], None]] = None,
    lambdas_override: Optional[dict[str, float]] = None,
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
    composite_rows = []

    for cutoff in cutoffs:
        if progress:
            progress(f"Backtest cutoff {cutoff.date()}")
        train_daily = daily_all.loc[daily_all["po_date"] <= cutoff].copy()
        if train_daily.empty:
            continue
        train_pairs = build_adjacent_pairs(train_daily)
        hier = fit_repeat_sales(train_pairs, config, lambdas_override=lambdas_override)
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
        matched_ann = np.log1p(recent_matched) * 4.0
        tq_ann = np.log1p(recent_tq) * 4.0

        best_fc, _ = select_forecast_by_backtest(hier.delta0, hier.months, [3, 6, 12])
        sigma = max(getattr(hier, "sigma", 0.05) or 0.05, 1e-4)

        for horizon in horizons:
            target = cutoff + pd.DateOffset(months=horizon)
            future = daily_all.loc[
                (daily_all["po_date"] > cutoff) & (daily_all["po_date"] <= target)
            ]
            if future.empty:
                continue
            future = _sample_future(future, max_eval, ctrls.random_seed + int(horizon) + cutoff.year)

            merge_cols = ["PartKey", "latest_price", "latest_date", "qty"]
            left_last = last_map.reset_index()
            merge_cols = [c for c in merge_cols if c in left_last.columns]
            fut = future.merge(
                left_last[merge_cols],
                on="PartKey",
                how="inner",
                suffixes=("_future", "_at_cutoff"),
            )
            fut = fut.loc[pd.to_datetime(fut["latest_date"]) <= cutoff]
            if fut.empty:
                continue

            # After merge: price from future; qty_future / qty_at_cutoff when both present
            price_col = "price_future" if "price_future" in fut.columns else "price"
            qty_fut_col = "qty_future" if "qty_future" in fut.columns else "qty"
            qty_cut_col = "qty_at_cutoff" if "qty_at_cutoff" in fut.columns else qty_fut_col

            base_price = fut["latest_price"].astype(float).to_numpy()
            base_date = pd.to_datetime(fut["latest_date"])
            actual_price = fut[price_col].astype(float).to_numpy()
            actual_date = pd.to_datetime(fut["po_date"])
            qty_cutoff = fut[qty_cut_col].fillna(1).clip(lower=1e-9).astype(float).to_numpy()
            qty_actual = fut[qty_fut_col].fillna(1).clip(lower=1e-9).astype(float).to_numpy()
            years = np.clip((actual_date - base_date).dt.days.to_numpy() / 365.25, 1e-6, None)

            cagr_m = _cagr_multiplier(train_daily, cutoff, target)
            preds = {
                "last_price": base_price.copy(),
                "overall_cagr": base_price * cagr_m,
                "matched_part": base_price * np.exp(matched_ann * years),
                "tornqvist": base_price * np.exp(tq_ann * years),
            }

            cands = build_forecast_candidates(hier.delta0, hier.months, target, cutoff)
            cand = next((c for c in cands if c.name == best_fc), cands[0])

            cat_cache: dict[tuple, float] = {}
            hier_preds = np.empty(len(fut))
            cat_preds = np.empty(len(fut))
            sources: list[str] = []
            p10 = np.empty(len(fut))
            p90 = np.empty(len(fut))
            for i in range(len(fut)):
                part = fut.iloc[i]["PartKey"]
                bd = base_date.iloc[i]
                ad = actual_date.iloc[i]
                if part_hier_map is not None and part in part_hier_map.index:
                    prow = part_hier_map.loc[part]
                    cat = str(prow.get("category", "overall"))
                    resid = float(prow.get("shrunk_residual", 0.0) or 0.0)
                    src = prow.get("source", "category")
                else:
                    cat = str(fut.iloc[i].get("approved_category", "overall"))
                    resid = 0.0
                    src = "overall"
                key = (cat, bd.normalize(), ad.normalize())
                if key not in cat_cache:
                    cat_cache[key] = multiplier_category(
                        hier, cat, bd, ad, cand.monthly_rates, cand.months
                    )
                years_i = max((ad - bd).days / 365.25, 0.0)
                m_cat = cat_cache[key]
                m_i = m_cat * float(np.exp(resid * years_i))
                hier_preds[i] = base_price[i] * m_i
                cat_preds[i] = base_price[i] * m_cat
                sources.append(src)
                # Predictive band from residual sigma
                band = float(np.exp(1.2816 * sigma * np.sqrt(max(years_i, 1e-3))))
                p10[i] = hier_preds[i] / band
                p90[i] = hier_preds[i] * band

            preds["category_benchmark"] = cat_preds
            preds["hierarchical"] = hier_preds
            preds["blended"] = 0.5 * hier_preds + 0.5 * preds["matched_part"]

            for model_name, pred in preds.items():
                for eval_mode, qty in [
                    ("price_only", qty_cutoff),
                    ("quantity_conditional", qty_actual),
                ]:
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
                                "qty": float(qty[i]),
                                "model": model_name,
                                "eval_mode": eval_mode,
                                "category": fut.iloc[i].get("approved_category"),
                                "fallback_source": sources[i]
                                if model_name in {"hierarchical", "category_benchmark", "blended"}
                                else model_name,
                                "pred_p10": float(p10[i]) if model_name == "hierarchical" else np.nan,
                                "pred_p90": float(p90[i]) if model_name == "hierarchical" else np.nan,
                            }
                        )

            # Composite fixed-basket error (price-only)
            for model_name, pred in preds.items():
                pred_map = {fut.iloc[i]["PartKey"]: float(pred[i]) for i in range(len(fut))}
                act_map = {fut.iloc[i]["PartKey"]: float(actual_price[i]) for i in range(len(fut))}
                # Collapse to last observation per part in window
                pred_part = {}
                act_part = {}
                for i in range(len(fut)):
                    p = fut.iloc[i]["PartKey"]
                    pred_part[p] = float(pred[i])
                    act_part[p] = float(actual_price[i])
                e_comp = _composite_error(last_map, fut, pred_part, act_part)
                composite_rows.append(
                    {
                        "cutoff": cutoff,
                        "horizon_months": horizon,
                        "model": model_name,
                        "E_composite": e_comp,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return BacktestResult(detail, pd.DataFrame(), "last_price", "No backtest evaluation rows")

    summary_rows = []
    # Primary selection uses price_only eval
    primary = detail.loc[detail["eval_mode"] == "price_only"]
    for (model, horizon), grp in primary.groupby(["model", "horizon_months"]):
        yt = grp["actual_price"].to_numpy()
        yp = grp["predicted_price"].to_numpy()
        q = grp["qty"].to_numpy()
        w = q * yt
        cov = (
            interval_coverage(yt, grp["pred_p10"].to_numpy(), grp["pred_p90"].to_numpy())
            if model == "hierarchical"
            else np.nan
        )
        # Matching quantity-conditional WAPE
        qc = detail.loc[
            (detail["eval_mode"] == "quantity_conditional")
            & (detail["model"] == model)
            & (detail["horizon_months"] == horizon)
        ]
        wape_qc = (
            wape(qc["actual_price"].to_numpy(), qc["predicted_price"].to_numpy(), qc["qty"].to_numpy())
            if not qc.empty
            else np.nan
        )
        e_comp = np.nan
        csub = [r for r in composite_rows if r["model"] == model and r["horizon_months"] == horizon]
        if csub:
            vals = [r["E_composite"] for r in csub if np.isfinite(r["E_composite"])]
            e_comp = float(np.mean(vals)) if vals else np.nan
        summary_rows.append(
            {
                "model": model,
                "horizon_months": horizon,
                "n": len(grp),
                "WAPE": wape(yt, yp, q),
                "WAPE_quantity_conditional": wape_qc,
                "WALE": wale(yt, yp, w),
                "MdAPE": mdape(yt, yp),
                "directional_accuracy": directional_accuracy(
                    yt, yp, grp["base_price"].to_numpy()
                ),
                "interval_coverage": cov,
                "E_composite": e_comp,
                "eval_mode": "price_only",
            }
        )
    summary = pd.DataFrame(summary_rows)

    avg = summary.groupby("model")["WAPE"].mean().sort_values()
    best = str(avg.index[0])
    best_wape = float(avg.iloc[0])
    selected = best
    rationale = f"Lowest average WAPE: {best} ({best_wape:.4f})"
    material = ctrls.material_wape_improvement
    # Prefer simplest model whose WAPE is within material of the best
    for cand in sorted(avg.index, key=lambda m: COMPLEXITY_ORDER.get(m, 99)):
        cand_wape = float(avg.loc[cand])
        if cand_wape <= best_wape * (1 + material) or best_wape >= cand_wape * (1 - material):
            if COMPLEXITY_ORDER.get(cand, 99) < COMPLEXITY_ORDER.get(best, 99):
                # Only switch if improvement of best over cand is below material threshold
                if best_wape >= cand_wape * (1 - material):
                    selected = cand
                    rationale = (
                        f"Preferred simpler model {cand} (WAPE {cand_wape:.4f}) over {best} "
                        f"(WAPE {best_wape:.4f}); improvement below {material:.1%} material threshold"
                    )
                    break
    # Re-evaluate: start from simplest; pick more complex only if it beats by material amount
    selected = "last_price"
    selected_wape = float(avg.loc["last_price"]) if "last_price" in avg.index else best_wape
    rationale = f"Baseline last_price WAPE={selected_wape:.4f}"
    for cand in sorted(avg.index, key=lambda m: COMPLEXITY_ORDER.get(m, 99)):
        cand_wape = float(avg.loc[cand])
        if COMPLEXITY_ORDER.get(cand, 99) <= COMPLEXITY_ORDER.get(selected, 99):
            if cand_wape < selected_wape:
                selected = cand
                selected_wape = cand_wape
                rationale = f"Lowest WAPE among equally-simple-or-simpler: {cand} ({cand_wape:.4f})"
            continue
        # More complex: require material improvement vs current selection
        if cand_wape <= selected_wape * (1 - material):
            # Also check composite error does not worsen materially if available
            sel_e = summary.loc[summary["model"] == selected, "E_composite"].mean()
            cand_e = summary.loc[summary["model"] == cand, "E_composite"].mean()
            if np.isfinite(sel_e) and np.isfinite(cand_e) and cand_e > sel_e * 1.05:
                continue
            rationale = (
                f"Selected {cand} (WAPE {cand_wape:.4f}) over {selected} "
                f"(WAPE {selected_wape:.4f}); relative improvement "
                f"{(selected_wape - cand_wape) / max(selected_wape, 1e-9):.1%} >= {material:.1%}"
            )
            selected = cand
            selected_wape = cand_wape

    if ctrls.selected_model_mode.value != "best_backtest":
        selected = ctrls.selected_model_mode.value
        rationale = f"Forced by selected_model_mode={selected}"

    logger.info("Selected model: %s (%s)", selected, rationale)
    return BacktestResult(detail, summary, selected, rationale)
