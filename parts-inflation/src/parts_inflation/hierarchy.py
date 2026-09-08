"""Part-level hierarchical shrinkage and fallback selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from parts_inflation.config import ResolvedConfig
from parts_inflation.repeat_sales import RepeatSalesResult, predict_pair_log_change

logger = logging.getLogger(__name__)


@dataclass
class PartHierarchyRow:
    PartKey: str
    category: str
    n_pairs: int
    span_days: float
    residual_rate: float
    lambda_shrink: float
    shrunk_residual: float
    quality: float
    source: str  # part | category | overall


def compute_part_residuals(
    pairs: pd.DataFrame,
    model: RepeatSalesResult,
    config: ResolvedConfig,
) -> pd.DataFrame:
    if pairs.empty or model.n_pairs_used == 0:
        return pd.DataFrame(
            columns=[
                "PartKey",
                "category",
                "n_pairs",
                "span_days",
                "residual_rate",
                "lambda_shrink",
                "shrunk_residual",
                "quality",
                "source",
            ]
        )

    yhat = predict_pair_log_change(model, pairs)
    e = pairs["y"].to_numpy() - yhat
    dt = pairs["delta_years"].to_numpy()
    # Reliability weight: inverse of extreme + qty comparability
    a = np.ones(len(pairs))
    if "extreme_flag" in pairs.columns:
        a = a * np.where(pairs["extreme_flag"].to_numpy(), 0.25, 1.0)
    if "qty_comparable" in pairs.columns:
        a = a * np.where(pairs["qty_comparable"].to_numpy(), 1.0, 0.7)
    # Downweight very short intervals
    a = a * np.clip(dt / (30 / 365.25), 0.1, 1.0)

    rows = []
    k = config.controls.part_shrinkage_k
    min_int = config.controls.part_min_intervals
    min_span = config.controls.part_min_span_days

    work = pairs.reset_index(drop=True)
    for part, grp in work.groupby("PartKey"):
        idx = grp.index.to_numpy()
        ee = e[idx]
        dtt = dt[idx]
        aa = a[idx]
        # annualized residual rate
        rate = float(np.sum(aa * ee / np.clip(dtt, 1e-6, None)) / np.sum(aa))
        n_eff = float(np.sum(aa))
        span_days = float(
            (
                pd.to_datetime(work.loc[idx, "date_b"]).max()
                - pd.to_datetime(work.loc[idx, "date_a"]).min()
            ).days
        )
        # Quality from dispersion and flags
        extreme_share = float(work.loc[idx, "extreme_flag"].mean()) if "extreme_flag" in work else 0.0
        qty_share = float(work.loc[idx, "qty_comparable"].mean()) if "qty_comparable" in work else 1.0
        quality = float(np.clip((1 - extreme_share) * (0.5 + 0.5 * qty_share), 0, 1))
        n_pairs = len(idx)
        cat = str(work.loc[idx[0], "approved_category"])

        if n_pairs >= min_int and span_days >= min_span:
            lam = (n_eff / (n_eff + k)) * min(span_days / 730.0, 1.0) * quality
            source = "part"
        else:
            lam = 0.0
            source = "category" if cat in model.categories else "overall"

        rows.append(
            {
                "PartKey": part,
                "category": cat,
                "n_pairs": n_pairs,
                "span_days": span_days,
                "residual_rate": rate,
                "lambda_shrink": lam,
                "shrunk_residual": lam * rate,
                "quality": quality,
                "source": source,
                "n_eff": n_eff,
            }
        )

    out = pd.DataFrame(rows)
    logger.info(
        "Part hierarchy: %s parts; part-level=%s category-fallback=%s",
        len(out),
        int((out["source"] == "part").sum()),
        int((out["source"] == "category").sum()),
    )
    return out


def month_fractions(date_a, date_b, months: list[pd.Period]) -> np.ndarray:
    from parts_inflation.matched_pairs import fractional_month_weights

    wm = fractional_month_weights(pd.Timestamp(date_a), pd.Timestamp(date_b))
    return np.array([wm.get(m, 0.0) for m in months], dtype=float)


def multiplier_category(
    model: RepeatSalesResult,
    category: str,
    date_a,
    date_b,
    future_monthly: Optional[np.ndarray] = None,
    future_months: Optional[list[pd.Period]] = None,
) -> float:
    """
    Historical months use fitted delta; months after last fitted month use future_monthly.
    """
    a = pd.Timestamp(date_a)
    b = pd.Timestamp(date_b)
    if b <= a:
        return 1.0
    if model.n_pairs_used == 0 or not model.months:
        # Fallback: no model — return 1.0 (zero inflation) rather than fabricate
        return 1.0

    hist_delta = model.category_monthly(category)
    # Trailing mean of category deviation û_c for future months
    u_hat = 0.0
    if category in model.categories and model.u.size:
        c_i = model.categories.index(category)
        u_series = model.u[c_i]
        if len(u_series):
            window = min(12, len(u_series))
            u_hat = float(np.mean(u_series[-window:]))

    log_sum = 0.0
    from parts_inflation.matched_pairs import fractional_month_weights

    wm = fractional_month_weights(a, b)
    for m, f in wm.items():
        if m in model.months:
            i = model.months.index(m)
            log_sum += f * hist_delta[i]
        elif future_monthly is not None and future_months is not None and m in future_months:
            j = future_months.index(m)
            # future_δ_c,m = future_δ_0,m + û_c
            log_sum += f * (float(future_monthly[j]) + u_hat)
        elif future_monthly is not None and len(future_monthly):
            log_sum += f * (float(future_monthly[-1]) + u_hat)
        else:
            log_sum += f * hist_delta[-1]
    return float(np.exp(log_sum))


def multiplier_part(
    model: RepeatSalesResult,
    part_row: pd.Series,
    date_a,
    date_b,
    future_monthly: Optional[np.ndarray] = None,
    future_months: Optional[list[pd.Period]] = None,
) -> tuple[float, str]:
    cat = part_row.get("category", "overall")
    m_c = multiplier_category(model, cat, date_a, date_b, future_monthly, future_months)
    years = (pd.Timestamp(date_b) - pd.Timestamp(date_a)).days / 365.25
    resid = float(part_row.get("shrunk_residual", 0.0) or 0.0)
    m_i = m_c * float(np.exp(resid * years))
    source = part_row.get("source", "category")
    if model.n_pairs_used == 0:
        source = "overall"
    return m_i, source
