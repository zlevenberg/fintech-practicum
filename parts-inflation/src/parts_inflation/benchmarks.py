"""Benchmark inflation indices: last-price, matched-part, Törnqvist, Fisher."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def last_price_forecast(daily: pd.DataFrame, base_date: pd.Timestamp) -> pd.DataFrame:
    """Latest observed price at or before base_date for each part."""
    d = daily.loc[pd.to_datetime(daily["po_date"]) <= pd.Timestamp(base_date)].copy()
    if d.empty:
        return pd.DataFrame(columns=["PartKey", "latest_price", "latest_date"])
    d = d.sort_values(["PartKey", "po_date"])
    last = d.groupby("PartKey", as_index=False).tail(1)
    return last.rename(columns={"price": "latest_price", "po_date": "latest_date"})[
        ["PartKey", "latest_price", "latest_date", "qty", "approved_category"]
    ]


def _period_label(freq: str, ts: pd.Timestamp) -> str:
    if freq == "M":
        return ts.to_period("M").strftime("%Y-%m")
    if freq == "Q":
        return str(ts.to_period("Q"))
    if freq == "Y":
        # Fiscal year ending Sep 30: FY starts Oct 1
        fy = ts.year + 1 if ts.month >= 10 else ts.year
        return f"FY{fy}"
    return str(ts)


def assign_period(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    d = daily.copy()
    d["po_date"] = pd.to_datetime(d["po_date"])
    if freq == "M":
        d["period"] = d["po_date"].dt.to_period("M").astype(str)
        d["period_end"] = d["po_date"].dt.to_period("M").dt.to_timestamp("M")
    elif freq == "Q":
        d["period"] = d["po_date"].dt.to_period("Q").astype(str)
        d["period_end"] = d["po_date"].dt.to_period("Q").dt.to_timestamp("Q")
    elif freq == "Y":
        fy = np.where(d["po_date"].dt.month >= 10, d["po_date"].dt.year + 1, d["po_date"].dt.year)
        d["period"] = [f"FY{int(y)}" for y in fy]
        d["period_end"] = pd.to_datetime([f"{int(y)}-09-30" for y in fy])
    else:
        raise ValueError(freq)
    return d


def period_part_prices(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Quantity-weighted mean price and total qty/spend per part-period."""
    d = assign_period(daily, freq)
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    valid = d["qty"].fillna(0).gt(0) & d["price"].notna() & d["price"].gt(0)
    d["_pq"] = np.where(valid, d["price"] * d["qty"], 0.0)
    d["_q"] = np.where(valid, d["qty"], 0.0)
    d["_spend"] = d["spend"].fillna(d["price"] * d["qty"].fillna(0))
    g = d.groupby(["PartKey", "period"], sort=False)
    agg = g.agg(
        pq=("_pq", "sum"),
        qty=("_q", "sum"),
        spend=("_spend", "sum"),
        price_median=("price", "median"),
        period_end=("period_end", "first"),
        approved_category=("approved_category", "first"),
    ).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        qw = np.where(agg["qty"] > 0, agg["pq"] / agg["qty"], np.nan)
    agg["price"] = np.where(np.isfinite(qw), qw, agg["price_median"])
    return agg.drop(columns=["pq", "price_median"])


def matched_part_log_changes(
    period_prices: pd.DataFrame,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """Adjacent-period matched-part log changes and aggregate measures."""
    if period_prices.empty:
        return pd.DataFrame()
    pp = period_prices.sort_values(["PartKey", "period_end"])
    g = pp.groupby("PartKey", sort=False)
    cur = pp.copy()
    cur["prev_period"] = g["period"].shift(1)
    cur["prev_price"] = g["price"].shift(1)
    cur["prev_qty"] = g["qty"].shift(1)
    cur["prev_spend"] = g["spend"].shift(1)
    cur["prev_period_end"] = g["period_end"].shift(1)
    m = cur.dropna(subset=["prev_price", "price"])
    m = m[(m["prev_price"] > 0) & (m["price"] > 0)]
    if m.empty:
        return pd.DataFrame()
    m["r"] = np.log(m["price"] / m["prev_price"])
    m["matched_spend"] = np.sqrt(m["spend"].fillna(0).clip(lower=0) * m["prev_spend"].fillna(0).clip(lower=0))

    results = []
    for period, grp in m.groupby("period"):
        r = grp["r"].to_numpy()
        w = grp["matched_spend"].to_numpy()
        if w.sum() <= 0:
            w = np.ones_like(r)
        q_l, q_u = np.quantile(r, [winsor_lower, winsor_upper])
        r_win = np.clip(r, q_l, q_u)
        # Cap spend at 95th percentile
        cap = np.quantile(w, 0.95) if len(w) > 1 else w.max()
        w_cap = np.minimum(w, cap)
        ew = float(np.mean(r))
        ew_win = float(np.mean(r_win))
        sw = float(np.average(r_win, weights=w_cap))
        results.append(
            {
                "period": period,
                "period_end": grp["period_end"].iloc[0],
                "match_count": len(grp),
                "matched_spend": float(grp["matched_spend"].sum()),
                "median_pct_change": float(np.expm1(np.median(r))),
                "equal_weight_geom": float(np.expm1(ew)),
                "winsor_equal_weight_geom": float(np.expm1(ew_win)),
                "capped_spend_weight_geom": float(np.expm1(sw)),
                "winsor_low": float(q_l),
                "winsor_high": float(q_u),
            }
        )
    return pd.DataFrame(results).sort_values("period_end")


def tornqvist_index(period_prices: pd.DataFrame) -> pd.DataFrame:
    """Törnqvist price index over adjacent periods for matched basket."""
    if period_prices.empty:
        return pd.DataFrame()
    pp = period_prices.sort_values(["PartKey", "period_end"])
    g = pp.groupby("PartKey", sort=False)
    cur = pp.copy()
    cur["prev_price"] = g["price"].shift(1)
    cur["prev_qty"] = g["qty"].shift(1)
    cur["prev_spend"] = g["spend"].shift(1)
    cur["prev_period"] = g["period"].shift(1)
    cur["prev_period_end"] = g["period_end"].shift(1)
    m = cur.dropna(subset=["prev_price", "price"])
    m = m[(m["prev_price"] > 0) & (m["price"] > 0)]
    if m.empty:
        return pd.DataFrame()

    rows = []
    index_level = 1.0
    for period, grp in m.groupby("period", sort=False):
        # Expenditure shares within matched basket
        spend_t = (grp["price"] * grp["qty"].fillna(0)).to_numpy()
        spend_tm1 = (grp["prev_price"] * grp["prev_qty"].fillna(0)).to_numpy()
        # Fallback to spend columns if qty missing
        if not np.isfinite(spend_t).any() or spend_t.sum() <= 0:
            spend_t = grp["spend"].fillna(0).to_numpy()
        if not np.isfinite(spend_tm1).any() or spend_tm1.sum() <= 0:
            spend_tm1 = grp["prev_spend"].fillna(0).to_numpy()
        if spend_t.sum() <= 0 or spend_tm1.sum() <= 0:
            rows.append(
                {
                    "period": period,
                    "period_end": grp["period_end"].iloc[0],
                    "index": index_level,
                    "pct_change": np.nan,
                    "match_count": len(grp),
                    "coverage_note": "insufficient spend for shares",
                    "status": "unavailable",
                }
            )
            continue
        s_t = spend_t / spend_t.sum()
        s_tm1 = spend_tm1 / spend_tm1.sum()
        s_bar = 0.5 * (s_t + s_tm1)
        r = np.log(grp["price"].to_numpy() / grp["prev_price"].to_numpy())
        g_t = float(np.sum(s_bar * r))
        index_level *= float(np.exp(g_t))
        rows.append(
            {
                "period": period,
                "period_end": grp["period_end"].iloc[0],
                "index": index_level,
                "log_change": g_t,
                "pct_change": float(np.expm1(g_t)),
                "match_count": len(grp),
                "matched_spend_t": float(spend_t.sum()),
                "matched_spend_tm1": float(spend_tm1.sum()),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def fisher_index(period_prices: pd.DataFrame) -> pd.DataFrame:
    """Fisher ideal index; returns unavailable when quantities are insufficient."""
    if period_prices.empty:
        return pd.DataFrame()
    pp = period_prices.sort_values(["PartKey", "period_end"])
    g = pp.groupby("PartKey", sort=False)
    cur = pp.copy()
    cur["prev_price"] = g["price"].shift(1)
    cur["prev_qty"] = g["qty"].shift(1)
    cur["prev_period_end"] = g["period_end"].shift(1)
    m = cur.dropna(subset=["prev_price", "price", "prev_qty", "qty"])
    m = m[(m["prev_price"] > 0) & (m["price"] > 0) & (m["prev_qty"] > 0) & (m["qty"] > 0)]
    if m.empty:
        return pd.DataFrame(
            [{"period": None, "status": "unavailable", "reason": "insufficient valid quantities"}]
        )

    rows = []
    index_level = 1.0
    for period, grp in m.groupby("period", sort=False):
        p_t = grp["price"].to_numpy()
        p_0 = grp["prev_price"].to_numpy()
        q_t = grp["qty"].to_numpy()
        q_0 = grp["prev_qty"].to_numpy()
        laspeyres_den = np.sum(p_0 * q_0)
        paasche_den = np.sum(p_0 * q_t)
        if laspeyres_den <= 0 or paasche_den <= 0:
            rows.append(
                {
                    "period": period,
                    "period_end": grp["period_end"].iloc[0],
                    "status": "unavailable",
                    "reason": "nonpositive denominator",
                    "match_count": len(grp),
                }
            )
            continue
        L = float(np.sum(p_t * q_0) / laspeyres_den)
        P = float(np.sum(p_t * q_t) / paasche_den)
        if L <= 0 or P <= 0:
            rows.append(
                {
                    "period": period,
                    "period_end": grp["period_end"].iloc[0],
                    "status": "unavailable",
                    "reason": "nonpositive L or P",
                    "match_count": len(grp),
                }
            )
            continue
        F = float(np.sqrt(L * P))
        index_level *= F
        rows.append(
            {
                "period": period,
                "period_end": grp["period_end"].iloc[0],
                "laspeyres": L,
                "paasche": P,
                "fisher": F,
                "index": index_level,
                "pct_change": F - 1.0,
                "match_count": len(grp),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def compute_all_benchmarks(
    daily: pd.DataFrame,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for freq, label in [("M", "monthly"), ("Q", "quarterly"), ("Y", "fiscal_year")]:
        pp = period_part_prices(daily, freq)
        out[f"period_prices_{label}"] = pp
        out[f"matched_{label}"] = matched_part_log_changes(pp, winsor_lower, winsor_upper)
        out[f"tornqvist_{label}"] = tornqvist_index(pp)
        out[f"fisher_{label}"] = fisher_index(pp)
        logger.info(
            "Benchmarks %s: periods=%s matched_rows=%s",
            label,
            pp["period"].nunique() if not pp.empty else 0,
            len(out[f"matched_{label}"]),
        )
    return out
