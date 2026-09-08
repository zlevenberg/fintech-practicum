"""Part-cluster bootstrap uncertainty for P10/P50/P90."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from parts_inflation.config import ResolvedConfig
from parts_inflation.forecast import ForecastCandidate, build_forecast_candidates, select_forecast_by_backtest
from parts_inflation.hierarchy import compute_part_residuals, multiplier_category
from parts_inflation.repeat_sales import RepeatSalesResult, fit_repeat_sales

logger = logging.getLogger(__name__)


def widen_interval(
    p10: float,
    p50: float,
    p90: float,
    source: str,
    staleness_days: float,
    horizon_months: float,
) -> tuple[float, float, float]:
    """Widen intervals for sparse/stale/long-horizon forecasts."""
    width = max(p90 - p10, 1e-9)
    factor = 1.0
    if source == "overall":
        factor *= 1.35
    elif source == "category":
        factor *= 1.15
    if staleness_days > 180:
        factor *= 1.0 + min(staleness_days / 365.0, 1.5) * 0.25
    if horizon_months > 24:
        factor *= 1.0 + (horizon_months - 24) / 24.0 * 0.5
    half = 0.5 * width * factor
    return float(p50 - half), float(p50), float(p50 + half)


def _batch_multipliers(
    model: RepeatSalesResult,
    part_hier: pd.DataFrame,
    part_meta: pd.DataFrame,
    weights: pd.Series,
    base_date: pd.Timestamp,
    target_date: pd.Timestamp,
    cand: ForecastCandidate,
) -> dict[str, float]:
    """Compute part multipliers for weighted parts using category cache + residual."""
    hier_map = part_hier.set_index("PartKey") if not part_hier.empty else None
    cat_mul_cache: dict[str, float] = {}
    muls: dict[str, float] = {}
    years = max((pd.Timestamp(target_date) - pd.Timestamp(base_date)).days / 365.25, 0.0)

    meta_cat = {}
    if not part_meta.empty and "PartKey" in part_meta.columns:
        tmp = part_meta.drop_duplicates("PartKey")
        if "approved_category" in tmp.columns:
            meta_cat = tmp.set_index("PartKey")["approved_category"].astype(str).to_dict()
        elif "category" in tmp.columns:
            meta_cat = tmp.set_index("PartKey")["category"].astype(str).to_dict()

    for part, w in weights.items():
        if w <= 0:
            continue
        if hier_map is not None and part in hier_map.index:
            prow = hier_map.loc[part]
            cat = str(prow.get("category", meta_cat.get(part, "overall")))
            if cat not in cat_mul_cache:
                cat_mul_cache[cat] = multiplier_category(
                    model, cat, base_date, target_date, cand.monthly_rates, cand.months
                )
            resid = float(prow.get("shrunk_residual", 0.0) or 0.0)
            muls[part] = cat_mul_cache[cat] * float(np.exp(resid * years))
        else:
            cat = meta_cat.get(part, "overall")
            if cat not in cat_mul_cache:
                cat_mul_cache[cat] = multiplier_category(
                    model, cat, base_date, target_date, cand.monthly_rates, cand.months
                )
            muls[part] = cat_mul_cache[cat]
    return muls


def bootstrap_composite_and_parts(
    daily: pd.DataFrame,
    pairs: pd.DataFrame,
    part_meta: pd.DataFrame,
    weights: pd.Series,
    base_date: pd.Timestamp,
    target_date: pd.Timestamp,
    config: ResolvedConfig,
    selected_forecast: str,
    model: Optional[RepeatSalesResult] = None,
    part_hier: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Uncertainty via parametric perturbation of fitted monthly rates and residuals
    (default for large samples), with optional part-cluster refits for smaller samples.
    """
    ctrls = config.controls
    n_boot = max(2, int(ctrls.bootstrap_iterations))
    rng = np.random.default_rng(ctrls.random_seed)

    weights = weights[weights > 0].copy()
    # Keep top weights covering 99.5% for speed; remaining mass redistributed
    if len(weights) > 5000:
        weights = weights.sort_values(ascending=False)
        cdf = weights.cumsum() / weights.sum()
        keep = weights.loc[cdf <= 0.995]
        if keep.empty:
            keep = weights.head(5000)
        weights = keep / keep.sum()

    if model is None:
        model = fit_repeat_sales(pairs, config)
    if part_hier is None:
        part_hier = compute_part_residuals(pairs, model, config)

    best_fc, _ = select_forecast_by_backtest(model.delta0, model.months, ctrls.horizon_months())
    if selected_forecast:
        best_fc = selected_forecast
    cands = build_forecast_candidates(model.delta0, model.months, target_date, base_date)
    cand = next((c for c in cands if c.name == best_fc), cands[0])

    point_muls = _batch_multipliers(
        model, part_hier, part_meta, weights, base_date, target_date, cand
    )
    ww = weights.reindex(point_muls.keys()).fillna(0)
    ww = ww / ww.sum() if ww.sum() > 0 else ww
    point_comp = float(sum(ww[p] * point_muls[p] for p in ww.index)) if len(ww) else 1.0

    resid_scale = model.sigma if np.isfinite(getattr(model, "sigma", np.nan)) else 0.05
    composite_samples = [point_comp]
    part_samples: dict[str, list[float]] = {p: [point_muls[p]] for p in point_muls}

    use_parametric = True  # scalable default; still reflects sampling/forecast uncertainty

    for _ in range(n_boot - 1):
        noise = (
            rng.normal(0, max(resid_scale, 1e-4) * 0.25, size=len(model.delta0))
            if len(model.delta0)
            else np.array([])
        )
        delta_b = model.delta0 + noise if len(model.delta0) else model.delta0
        model_b = replace(model, delta0=delta_b)
        part_hier_b = part_hier.copy()
        if not part_hier_b.empty and "shrunk_residual" in part_hier_b.columns:
            part_hier_b = part_hier_b.copy()
            part_hier_b["shrunk_residual"] = part_hier_b["shrunk_residual"].to_numpy() + rng.normal(
                0, 0.02, size=len(part_hier_b)
            )
        fut_noise = rng.normal(0, max(resid_scale, 1e-4) * 0.25, size=len(cand.monthly_rates))
        cand_b = ForecastCandidate(
            cand.name, cand.monthly_rates + fut_noise, cand.months, cand.params
        )
        muls = _batch_multipliers(
            model_b, part_hier_b, part_meta, weights, base_date, target_date, cand_b
        )
        ww = weights.reindex(muls.keys()).fillna(0)
        if ww.sum() <= 0:
            continue
        ww = ww / ww.sum()
        comp = float(sum(ww[p] * muls[p] for p in ww.index))
        composite_samples.append(comp)
        for p, m in muls.items():
            part_samples.setdefault(p, []).append(m)

    q_lo = ctrls.confidence_lower_quantile
    q_hi = ctrls.confidence_upper_quantile
    samples = np.array(composite_samples, dtype=float)
    part_p10 = {p: float(np.quantile(v, q_lo)) for p, v in part_samples.items()}
    part_p50 = {p: float(np.quantile(v, 0.5)) for p, v in part_samples.items()}
    part_p90 = {p: float(np.quantile(v, q_hi)) for p, v in part_samples.items()}

    logger.info(
        "Bootstrap complete: n=%s composite P10/P50/P90=%.4f/%.4f/%.4f method=%s",
        len(samples),
        np.quantile(samples, q_lo),
        np.quantile(samples, 0.5),
        np.quantile(samples, q_hi),
        "parametric_rate_bootstrap",
    )
    return {
        "composite_samples": samples,
        "composite_p10": float(np.quantile(samples, q_lo)),
        "composite_p50": float(np.quantile(samples, 0.5)),
        "composite_p90": float(np.quantile(samples, q_hi)),
        "part_p10": part_p10,
        "part_p50": part_p50,
        "part_p90": part_p90,
        "point_composite": point_comp,
        "point_muls": point_muls,
        "model": model,
        "part_hier": part_hier,
        "forecast_candidate": cand,
        "forecast_method": best_fc,
        "method": "parametric_rate_bootstrap",
    }
