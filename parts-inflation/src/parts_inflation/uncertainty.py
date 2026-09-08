"""Part-cluster bootstrap uncertainty for P10/P50/P90."""

from __future__ import annotations

import logging
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


def _resample_pairs_by_part(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample parts with replacement; keep all observations for each sampled part."""
    if pairs.empty:
        return pairs.iloc[0:0].copy()
    parts = pairs["PartKey"].astype(str).to_numpy()
    unique_parts = np.unique(parts)
    sampled = rng.choice(unique_parts, size=len(unique_parts), replace=True)
    # Build index lists per part once
    groups = pairs.groupby(pairs["PartKey"].astype(str), sort=False).indices
    chunks = []
    for i, part in enumerate(sampled):
        idx = groups.get(part)
        if idx is None or len(idx) == 0:
            continue
        chunk = pairs.iloc[idx].copy()
        # Disambiguate duplicate part draws
        chunk["PartKey"] = f"{part}__boot{i}"
        chunks.append(chunk)
    if not chunks:
        return pairs.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


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
    lambdas_override: Optional[dict[str, float]] = None,
) -> dict:
    """
    Part-cluster bootstrap: resample parts with replacement, keep all pairs for each
    sampled part, refit the hierarchical model, and recompute multipliers.
    """
    ctrls = config.controls
    n_boot = max(2, int(ctrls.bootstrap_iterations))
    rng = np.random.default_rng(ctrls.random_seed)

    weights = weights[weights > 0].copy()
    # Cap weight set for multiplier aggregation speed only (fit still uses full resampled pairs)
    if len(weights) > 5000:
        weights = weights.sort_values(ascending=False)
        cdf = weights.cumsum() / weights.sum()
        keep = weights.loc[cdf <= 0.995]
        if keep.empty:
            keep = weights.head(5000)
        weights = keep / keep.sum()

    if model is None:
        model = fit_repeat_sales(pairs, config, lambdas_override=lambdas_override)
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

    # Category point multipliers for interval samples
    cats = sorted(
        set(
            list(part_hier["category"].dropna().astype(str))
            if not part_hier.empty and "category" in part_hier.columns
            else []
        )
        | set(model.categories)
    )
    cat_point = {
        c: multiplier_category(model, c, base_date, target_date, cand.monthly_rates, cand.months)
        for c in cats
    }

    composite_samples = [point_comp]
    part_samples: dict[str, list[float]] = {p: [point_muls[p]] for p in point_muls}
    cat_samples: dict[str, list[float]] = {c: [cat_point[c]] for c in cats}

    # Forecast-method residual scale from rolling selection MAE if available
    fc_err = max(getattr(model, "sigma", 0.05) or 0.05, 1e-4) * 0.15

    for b in range(n_boot - 1):
        boot_pairs = _resample_pairs_by_part(pairs, rng)
        if boot_pairs.empty:
            continue
        # Bootstrap refits use fewer IRLS iterations (uncertainty sampling, same objective).
        boot_iter = 8 if ctrls.fast_mode else 15
        model_b = fit_repeat_sales(
            boot_pairs,
            config,
            lambdas_override=lambdas_override,
            max_iter_override=boot_iter,
        )
        if model_b.n_pairs_used == 0:
            continue
        # Recompute residuals on boot_pairs then map keys back.
        part_hier_b = compute_part_residuals(boot_pairs, model_b, config)
        if not part_hier_b.empty:
            part_hier_b = part_hier_b.copy()
            part_hier_b["PartKey"] = (
                part_hier_b["PartKey"].astype(str).str.replace(r"__boot\d+$", "", regex=True)
            )
            # Average duplicate draws of the same part
            agg = {
                "category": "first",
                "source": "first",
                "shrunk_residual": "mean",
                "lambda_shrink": "mean",
                "quality": "mean",
                "n_pairs": "sum",
                "span_days": "max",
            }
            agg = {k: v for k, v in agg.items() if k in part_hier_b.columns}
            part_hier_b = part_hier_b.groupby("PartKey", as_index=False).agg(agg)
        cands_b = build_forecast_candidates(model_b.delta0, model_b.months, target_date, base_date)
        # Prefer same forecast method name; add forecast-method error noise
        cand_b = next((c for c in cands_b if c.name == best_fc), cands_b[0] if cands_b else cand)
        fut_noise = rng.normal(0, fc_err, size=len(cand_b.monthly_rates))
        cand_b = ForecastCandidate(
            cand_b.name, cand_b.monthly_rates + fut_noise, cand_b.months, cand_b.params
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
        for c in cats:
            cat_samples.setdefault(c, []).append(
                multiplier_category(
                    model_b, c, base_date, target_date, cand_b.monthly_rates, cand_b.months
                )
            )

    q_lo = ctrls.confidence_lower_quantile
    q_hi = ctrls.confidence_upper_quantile
    samples = np.array(composite_samples, dtype=float)
    part_p10 = {p: float(np.quantile(v, q_lo)) for p, v in part_samples.items()}
    part_p50 = {p: float(np.quantile(v, 0.5)) for p, v in part_samples.items()}
    part_p90 = {p: float(np.quantile(v, q_hi)) for p, v in part_samples.items()}
    cat_p10 = {c: float(np.quantile(v, q_lo)) for c, v in cat_samples.items()}
    cat_p50 = {c: float(np.quantile(v, 0.5)) for c, v in cat_samples.items()}
    cat_p90 = {c: float(np.quantile(v, q_hi)) for c, v in cat_samples.items()}

    logger.info(
        "Bootstrap complete: n=%s composite P10/P50/P90=%.4f/%.4f/%.4f method=%s",
        len(samples),
        np.quantile(samples, q_lo),
        np.quantile(samples, 0.5),
        np.quantile(samples, q_hi),
        "part_cluster_bootstrap",
    )
    return {
        "composite_samples": samples,
        "composite_p10": float(np.quantile(samples, q_lo)),
        "composite_p50": float(np.quantile(samples, 0.5)),
        "composite_p90": float(np.quantile(samples, q_hi)),
        "part_p10": part_p10,
        "part_p50": part_p50,
        "part_p90": part_p90,
        "category_p10": cat_p10,
        "category_p50": cat_p50,
        "category_p90": cat_p90,
        "point_composite": point_comp,
        "point_muls": point_muls,
        "model": model,
        "part_hier": part_hier,
        "forecast_candidate": cand,
        "forecast_method": best_fc,
        "method": "part_cluster_bootstrap",
    }
