"""Adjacent matched-pair construction and fractional-month design matrix helpers."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_adjacent_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build consecutive same-part purchase pairs only (not all combinations).
    Expects columns: PartKey, po_date, price, qty, spend, approved_category.
    """
    if daily is None or daily.empty:
        return pd.DataFrame()

    d = daily.sort_values(["PartKey", "po_date"]).copy()
    d["po_date"] = pd.to_datetime(d["po_date"])
    g = d.groupby("PartKey", sort=False)
    prev = d.copy()
    prev[["prev_date", "prev_price", "prev_qty", "prev_spend"]] = g[
        ["po_date", "price", "qty", "spend"]
    ].shift(1)

    pairs = prev.dropna(subset=["prev_date", "prev_price", "price"]).copy()
    pairs = pairs[(pairs["prev_price"] > 0) & (pairs["price"] > 0)]
    pairs["y"] = np.log(pairs["price"] / pairs["prev_price"])
    pairs["delta_days"] = (pairs["po_date"] - pairs["prev_date"]).dt.days.astype(float)
    pairs = pairs[pairs["delta_days"] > 0]

    valid_q = (
        pairs["qty"].notna()
        & pairs["prev_qty"].notna()
        & (pairs["qty"] > 0)
        & (pairs["prev_qty"] > 0)
    )
    pairs["x_q"] = np.where(valid_q, np.log(pairs["qty"] / pairs["prev_qty"]), 0.0)
    pairs["qty_comparable"] = valid_q
    pairs["delta_years"] = pairs["delta_days"] / 365.25
    pairs["price_ratio"] = pairs["price"] / pairs["prev_price"]
    pairs = pairs.rename(
        columns={
            "po_date": "date_b",
            "prev_date": "date_a",
            "price": "price_b",
            "prev_price": "price_a",
            "qty": "qty_b",
            "prev_qty": "qty_a",
            "spend": "spend_b",
            "prev_spend": "spend_a",
        }
    )
    pairs["n_pairs_part"] = pairs.groupby("PartKey")["PartKey"].transform("size")
    logger.info("Built %s adjacent pairs across %s parts", len(pairs), pairs["PartKey"].nunique())
    return pairs.reset_index(drop=True)


def fractional_month_weights(date_a: pd.Timestamp, date_b: pd.Timestamp) -> dict[pd.Period, float]:
    """
    Fraction of each calendar month contained in (date_a, date_b].
    A full covered month has value 1; partial months get day fractions.
    """
    a = pd.Timestamp(date_a).normalize()
    b = pd.Timestamp(date_b).normalize()
    if b <= a:
        return {}
    weights: dict[pd.Period, float] = {}
    # Start the day after a
    start = a + pd.Timedelta(days=1)
    if start > b:
        return {}
    cur_month = start.to_period("M")
    end_month = b.to_period("M")
    while cur_month <= end_month:
        month_start = cur_month.to_timestamp()
        next_month = (cur_month + 1).to_timestamp()
        days_in_month = (next_month - month_start).days
        seg_start = max(start, month_start)
        seg_end = min(b, next_month - pd.Timedelta(days=1))
        if seg_start <= seg_end:
            days_in_seg = (seg_end - seg_start).days + 1
            weights[cur_month] = days_in_seg / days_in_month
        cur_month = cur_month + 1
    for k in list(weights):
        weights[k] = float(min(1.0, max(0.0, weights[k])))
    return weights


def month_coverage_matrix(
    pairs: pd.DataFrame, month_index: Optional[list[pd.Period]] = None
) -> tuple[np.ndarray, list[pd.Period]]:
    """Build dense D matrix of shape (n_pairs, n_months) with fractional month weights."""
    if pairs.empty:
        return np.zeros((0, 0)), []

    a = pd.to_datetime(pairs["date_a"]).dt.normalize()
    b = pd.to_datetime(pairs["date_b"]).dt.normalize()
    # Inclusive coverage starts day after a
    start = a + pd.Timedelta(days=1)
    valid = start <= b
    if month_index is None:
        if not valid.any():
            return np.zeros((len(pairs), 0)), []
        min_m = start[valid].min().to_period("M")
        max_m = b[valid].max().to_period("M")
        all_months = list(pd.period_range(min_m, max_m, freq="M"))
    else:
        all_months = list(month_index)

    if not all_months:
        return np.zeros((len(pairs), 0)), []

    n = len(pairs)
    M = len(all_months)
    D = np.zeros((n, M), dtype=np.float64)
    month_starts = np.array([m.to_timestamp().toordinal() for m in all_months], dtype=np.int64)
    next_starts = np.array(
        [(m + 1).to_timestamp().toordinal() for m in all_months], dtype=np.int64
    )
    days_in_month = (next_starts - month_starts).astype(np.float64)

    start_ord = np.array([pd.Timestamp(x).toordinal() for x in start], dtype=np.int64)
    end_ord = np.array([pd.Timestamp(x).toordinal() for x in b], dtype=np.int64)
    valid_np = valid.to_numpy()

    for j in range(M):
        ms = month_starts[j]
        ns = next_starts[j] - 1  # last day of month ordinal
        left = np.maximum(start_ord, ms)
        right = np.minimum(end_ord, ns)
        overlap = np.clip(right - left + 1, 0, None).astype(np.float64)
        D[:, j] = np.where(valid_np, overlap / days_in_month[j], 0.0)
        D[:, j] = np.clip(D[:, j], 0.0, 1.0)

    return D, all_months


def flag_extreme_pairs(
    pairs: pd.DataFrame,
    ratio_low: float = 0.25,
    ratio_high: float = 4.0,
    max_days: int = 548,
) -> pd.DataFrame:
    out = pairs.copy()
    short = out["delta_days"] <= max_days
    out["extreme_ratio"] = short & (
        (out["price_ratio"] < ratio_low) | (out["price_ratio"] > ratio_high)
    )
    ann = out["y"] / out["delta_years"].clip(lower=1 / 365.25)
    med = np.nanmedian(ann)
    mad = np.nanmedian(np.abs(ann - med)) + 1e-12
    out["ann_log_change"] = ann
    out["mad_z"] = 0.6745 * (ann - med) / mad
    out["extreme_mad"] = np.abs(out["mad_z"]) > 8
    out["extreme_flag"] = out["extreme_ratio"] | out["extreme_mad"]
    return out
