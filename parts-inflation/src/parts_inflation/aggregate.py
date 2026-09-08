"""Same-part same-day aggregation and period helpers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from parts_inflation.config import ResolvedConfig, SameDayAgg

logger = logging.getLogger(__name__)


def aggregate_same_day(df: pd.DataFrame, config: ResolvedConfig) -> pd.DataFrame:
    """
    Aggregate multiple observations for the same PartKey and date into one price/qty.
    Preserves line counts and price dispersion diagnostics.
    """
    use = df.loc[df["model_eligible"]].copy()
    if use.empty:
        return pd.DataFrame()

    use["po_date"] = pd.to_datetime(use["po_date"]).dt.normalize()
    mode = config.controls.same_day_price_aggregation

    use["_pq"] = use["price"] * use["qty"]
    use["_spend"] = use["po_value"].fillna(use["_pq"])
    valid_q = use["qty"].notna() & (use["qty"] > 0) & use["price"].notna() & (use["price"] > 0)
    use["_pq_valid"] = np.where(valid_q, use["_pq"], 0.0)
    use["_q_valid"] = np.where(valid_q, use["qty"], 0.0)

    g = use.groupby(["PartKey", "po_date"], sort=False)
    agg = g.agg(
        qty=("_q_valid", "sum"),
        pq=("_pq_valid", "sum"),
        spend=("_spend", "sum"),
        line_count=("price", "size"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        price_median=("price", "median"),
        is_open_order_any=("is_open_order", "any"),
    ).reset_index()

    # Category / descriptions: first non-null via groupby
    cat = (
        use.sort_values("po_date")
        .groupby(["PartKey", "po_date"], sort=False)
        .agg(
            approved_category=("approved_category", "first"),
            **{
                "Description 1": ("Description 1", "first"),
                "Description 2": ("Description 2", "first"),
            },
        )
        .reset_index()
    )
    agg = agg.merge(cat, on=["PartKey", "po_date"], how="left")

    if mode == SameDayAgg.quantity_weighted_mean:
        with np.errstate(divide="ignore", invalid="ignore"):
            qw = np.where(agg["qty"] > 0, agg["pq"] / agg["qty"], np.nan)
        use_median = ~np.isfinite(qw)
        price = np.where(use_median, agg["price_median"], qw)
        agg_method = np.where(use_median, "median", "quantity_weighted_mean")
    else:
        price = agg["price_median"].to_numpy()
        agg_method = np.full(len(agg), "median")

    agg["price"] = price
    agg["agg_method"] = agg_method
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["price_dispersion_cv"] = np.where(
            (agg["price_mean"] > 0) & agg["price_std"].notna(),
            agg["price_std"] / agg["price_mean"],
            0.0,
        )
    agg = agg.dropna(subset=["price"])
    agg = agg[agg["price"] > 0].copy()
    # Replace zero qty with NaN for downstream clarity when median fallback had no qty
    agg.loc[agg["qty"] <= 0, "qty"] = np.nan

    keep = [
        "PartKey",
        "po_date",
        "price",
        "qty",
        "spend",
        "line_count",
        "price_dispersion_cv",
        "price_min",
        "price_max",
        "agg_method",
        "approved_category",
        "Description 1",
        "Description 2",
        "is_open_order_any",
    ]
    out = agg[keep]
    logger.info(
        "Same-day aggregation: %s part-days from %s eligible lines",
        len(out),
        len(use),
    )
    return out


def aggregate_for_scope(df: pd.DataFrame, scope_col: str, config: ResolvedConfig) -> pd.DataFrame:
    """Aggregate using an alternate scope boolean column."""
    tmp = df.loc[df[scope_col]].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["model_eligible"] = True
    return aggregate_same_day(tmp, config)


def period_ends(freq: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if freq == "M":
        return pd.date_range(start=start, end=end, freq="ME")
    if freq == "Q":
        return pd.date_range(start=start, end=end, freq="QE")
    if freq == "Y":
        years = range(start.year - 1, end.year + 2)
        ends = [pd.Timestamp(y, 9, 30) for y in years]
        return pd.DatetimeIndex([d for d in ends if start <= d <= end + pd.Timedelta(days=370)])
    raise ValueError(freq)
