"""Independent adjacent-pair Huber regression for historical actuals validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from parts_inflation.matched_pairs import build_adjacent_pairs

logger = logging.getLogger(__name__)

DEFAULT_TRIM_LEVELS = (0.0, 0.005, 0.01, 0.025)


@dataclass
class RegressionSettings:
    min_pair_gap_days: int = 30
    weight_cap_quantile: float = 0.95
    huber_epsilon: float = 1.35
    huber_alpha: float = 0.1
    huber_max_iter: int = 1000
    trim_levels: tuple[float, ...] = DEFAULT_TRIM_LEVELS


def aggregate_daily_physical(scoped: pd.DataFrame) -> pd.DataFrame:
    """
    Same-part same-day aggregation for physical inputs.

    - daily price: median Cost
    - daily quantity: median positive Qty Ordered
    - daily spend: sum of nonnegative PO Value
    - daily category: modal normalized Description
    """
    use = scoped.loc[scoped["in_physical_inputs"]].copy()
    if use.empty:
        return pd.DataFrame()

    use["po_date"] = pd.to_datetime(use["po_date"], errors="coerce").dt.normalize()
    use["price"] = pd.to_numeric(use["price"], errors="coerce")
    use = use.loc[use["po_date"].notna() & use["PartKey"].notna() & (use["price"] > 0)].copy()
    use["po_value_nonneg"] = pd.to_numeric(use["po_value"], errors="coerce").fillna(0).clip(lower=0)
    use["qty_pos"] = pd.to_numeric(use["qty_ordered"], errors="coerce")
    use.loc[~(use["qty_pos"] > 0), "qty_pos"] = np.nan

    def _mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else (s.iloc[0] if len(s) else np.nan)

    daily = (
        use.groupby(["PartKey", "po_date"], sort=False)
        .agg(
            price=("price", "median"),
            qty=("qty_pos", "median"),
            spend=("po_value_nonneg", "sum"),
            approved_category=("description_norm", _mode),
        )
        .reset_index()
    )
    daily = daily.loc[daily["price"] > 0].copy()
    return daily


def build_regression_pairs(
    daily: pd.DataFrame,
    min_gap_days: int = 30,
) -> pd.DataFrame:
    """
    Consecutive same-part pairs only; require positive prices/qty and min gap.
    """
    if daily.empty:
        return pd.DataFrame()
    pairs = build_adjacent_pairs(daily)
    if pairs.empty:
        return pairs
    pairs = pairs.loc[
        (pairs["price_a"] > 0)
        & (pairs["price_b"] > 0)
        & (pairs["qty_a"] > 0)
        & (pairs["qty_b"] > 0)
        & (pairs["delta_days"] >= min_gap_days)
    ].copy()
    pairs["y"] = np.log(pairs["price_b"] / pairs["price_a"])
    pairs["delta_t"] = pairs["delta_days"] / 365.25
    pairs["x_q"] = np.log(pairs["qty_b"] / pairs["qty_a"])
    pairs["a_j"] = pairs["y"] / pairs["delta_t"].clip(lower=1e-12)
    # Exposure weights
    va = pairs["spend_a"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    vb = pairs["spend_b"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pairs["b_j"] = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    return pairs.reset_index(drop=True)


def pair_regression_weights(
    pairs: pd.DataFrame,
    weight_cap_quantile: float = 0.95,
) -> np.ndarray:
    """
    w_j = b_j^cap / sqrt(N_i), then normalize to mean 1.
    """
    if pairs.empty:
        return np.array([])
    b = pairs["b_j"].to_numpy(dtype=float)
    cap = float(np.quantile(b, weight_cap_quantile)) if len(b) else 0.0
    b_cap = np.minimum(b, cap)
    n_i = pairs["n_pairs_part"].to_numpy(dtype=float)
    w = b_cap / np.sqrt(np.maximum(n_i, 1.0))
    mean_w = float(np.mean(w)) if len(w) else 1.0
    if mean_w <= 0 or not np.isfinite(mean_w):
        return np.ones(len(pairs))
    return w / mean_w


def _fit_huber(
    y: np.ndarray,
    X: np.ndarray,
    sample_weight: Optional[np.ndarray],
    settings: RegressionSettings,
) -> dict[str, Any]:
    model = HuberRegressor(
        fit_intercept=False,
        epsilon=settings.huber_epsilon,
        alpha=settings.huber_alpha,
        max_iter=settings.huber_max_iter,
    )
    try:
        if sample_weight is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=sample_weight)
        beta_t = float(model.coef_[0])
        gamma = float(model.coef_[1]) if X.shape[1] > 1 else float("nan")
        return {
            "beta_t": beta_t,
            "gamma": gamma,
            "annual_rate": float(np.expm1(beta_t)),
            "qty_doubling_effect": float(2.0**gamma - 1.0) if np.isfinite(gamma) else float("nan"),
            "converged": bool(getattr(model, "n_iter_", settings.huber_max_iter) < settings.huber_max_iter),
            "n_iter": int(getattr(model, "n_iter_", -1)),
            "status": "ok",
        }
    except Exception as exc:
        logger.warning("Huber regression failed: %s", exc)
        return {
            "beta_t": np.nan,
            "gamma": np.nan,
            "annual_rate": np.nan,
            "qty_doubling_effect": np.nan,
            "converged": False,
            "n_iter": -1,
            "status": f"failed: {exc}",
        }


def run_trim_grid(
    pairs: pd.DataFrame,
    settings: RegressionSettings,
) -> pd.DataFrame:
    """Weighted and unweighted Huber at each trim level."""
    if pairs.empty:
        return pd.DataFrame()

    rows = []
    a = pairs["a_j"].to_numpy(dtype=float)
    for trim in settings.trim_levels:
        if trim <= 0:
            mask = np.ones(len(pairs), dtype=bool)
        else:
            lo = float(np.quantile(a, trim))
            hi = float(np.quantile(a, 1.0 - trim))
            mask = (a >= lo) & (a <= hi)
        sub = pairs.loc[mask].copy()
        if sub.empty:
            continue
        y = sub["y"].to_numpy(dtype=float)
        X = np.column_stack([sub["delta_t"].to_numpy(dtype=float), sub["x_q"].to_numpy(dtype=float)])
        w = pair_regression_weights(sub, settings.weight_cap_quantile)

        unweighted = _fit_huber(y, X, None, settings)
        weighted = _fit_huber(y, X, w, settings)

        for kind, res in (("unweighted", unweighted), ("spend_weighted", weighted)):
            rows.append(
                {
                    "trim": trim,
                    "weighting": kind,
                    "n_pairs": int(len(sub)),
                    "n_parts": int(sub["PartKey"].nunique()),
                    "annual_rate": res["annual_rate"],
                    "beta_t": res["beta_t"],
                    "gamma": res["gamma"],
                    "qty_doubling_effect": res["qty_doubling_effect"],
                    "converged": res["converged"],
                    "n_iter": res["n_iter"],
                    "status": res["status"],
                }
            )
    return pd.DataFrame(rows)


def run_repeat_purchase_regression(
    scoped: pd.DataFrame,
    settings: Optional[RegressionSettings] = None,
) -> dict[str, Any]:
    settings = settings or RegressionSettings()
    daily = aggregate_daily_physical(scoped)
    pairs = build_regression_pairs(daily, settings.min_pair_gap_days)
    grid = run_trim_grid(pairs, settings)

    no_trim_w = grid.loc[(grid["trim"] == 0.0) & (grid["weighting"] == "spend_weighted")]
    no_trim_u = grid.loc[(grid["trim"] == 0.0) & (grid["weighting"] == "unweighted")]

    return {
        "daily": daily,
        "pairs": pairs,
        "grid": grid,
        "n_pairs": int(len(pairs)),
        "n_parts": int(pairs["PartKey"].nunique()) if not pairs.empty else 0,
        "weighted_annual_rate_no_trim": float(no_trim_w["annual_rate"].iloc[0]) if len(no_trim_w) else np.nan,
        "unweighted_annual_rate_no_trim": float(no_trim_u["annual_rate"].iloc[0]) if len(no_trim_u) else np.nan,
        "gamma_no_trim_weighted": float(no_trim_w["gamma"].iloc[0]) if len(no_trim_w) else np.nan,
        "qty_doubling_no_trim_weighted": float(no_trim_w["qty_doubling_effect"].iloc[0]) if len(no_trim_w) else np.nan,
    }
