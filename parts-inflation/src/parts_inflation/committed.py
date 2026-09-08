"""Open-PO committed-cost signal, kept separate from realized inflation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_prior_prices(realized: pd.DataFrame, open_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    realized_groups = {
        str(key): grp.sort_values("effective_historical_date")
        for key, grp in realized.groupby("comparison_entity_id", sort=False)
    }
    for entity, group in open_rows.groupby("comparison_entity_id", sort=False):
        history = realized_groups.get(str(entity))
        if history is None or history.empty:
            out = group.copy()
            out["prior_realized_price"] = np.nan
            out["prior_realized_date"] = pd.NaT
            rows.append(out)
            continue
        dates = pd.to_datetime(history["effective_historical_date"]).to_numpy()
        prices = history["historical_unit_price"].to_numpy(float)
        pieces = []
        for _, row in group.iterrows():
            order_date = np.datetime64(pd.Timestamp(row["po_date"]))
            position = int(np.searchsorted(dates, order_date, side="left") - 1)
            rec = row.to_dict()
            if position >= 0:
                rec["prior_realized_price"] = float(prices[position])
                rec["prior_realized_date"] = pd.Timestamp(dates[position])
            else:
                rec["prior_realized_price"] = np.nan
                rec["prior_realized_date"] = pd.NaT
            pieces.append(rec)
        rows.append(pd.DataFrame(pieces))
    return pd.concat(rows, ignore_index=True) if rows else open_rows.iloc[0:0].copy()


def build_committed_cost_indicator(
    classified: pd.DataFrame,
    base_date: pd.Timestamp,
    minimum_gap_days: int = 30,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matched open-line detail and a category/overall summary."""
    if classified.empty:
        return pd.DataFrame(), pd.DataFrame()
    cutoff = pd.Timestamp(base_date)
    direct = classified.loc[classified["in_direct_costs"].fillna(False)].copy()
    realized = direct.loc[
        direct["historical_unit_price"].fillna(0).gt(0)
        & direct["qty_received"].fillna(0).gt(0)
        & pd.to_datetime(direct["effective_historical_date"]).le(cutoff)
    ].copy()
    open_rows = direct.loc[
        direct["has_open_commitment"].fillna(False)
        & direct["committed_unit_price"].fillna(0).gt(0)
        & pd.to_datetime(direct["po_date"]).le(cutoff)
    ].copy()
    if open_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched = _latest_prior_prices(realized, open_rows)
    matched["matched"] = matched["prior_realized_price"].fillna(0).gt(0)
    matched["gap_days"] = (
        pd.to_datetime(matched["po_date"]) - pd.to_datetime(matched["prior_realized_date"])
    ).dt.days
    matched["log_price_change"] = np.where(
        matched["matched"],
        np.log(matched["committed_unit_price"] / matched["prior_realized_price"]),
        np.nan,
    )
    matched["annualized_log_change"] = np.where(
        matched["matched"] & matched["gap_days"].ge(minimum_gap_days),
        matched["log_price_change"] / (matched["gap_days"] / 365.25),
        np.nan,
    )
    matched["open_weight"] = matched["remaining_open_spend"].clip(lower=0)

    total_open = float(open_rows["remaining_open_spend"].clip(lower=0).sum())
    summaries: list[dict] = []
    scopes = [("Overall Direct Costs", matched)] + list(
        matched.groupby("direct_cost_category", sort=True)
    )
    for category, group in scopes:
        eligible = group.loc[
            group["annualized_log_change"].notna() & group["open_weight"].gt(0)
        ].copy()
        group_total = float(group["open_weight"].sum())
        matched_value = float(eligible["open_weight"].sum())
        if eligible.empty:
            signal = np.nan
        else:
            lower, upper = eligible["annualized_log_change"].quantile(
                [winsor_lower, winsor_upper]
            )
            clipped = eligible["annualized_log_change"].clip(lower=lower, upper=upper)
            cap = float(eligible["open_weight"].quantile(0.95))
            weights = eligible["open_weight"].clip(upper=cap)
            signal = float(np.average(clipped, weights=weights))
        summaries.append(
            {
                "category": category,
                "open_rows": int(len(group)),
                "matched_rows": int(eligible.shape[0]),
                "open_value": group_total,
                "matched_open_value": matched_value,
                "matched_open_value_coverage": matched_value / group_total if group_total > 0 else np.nan,
                "annualized_log_signal": signal,
                "annualized_rate_signal": float(np.expm1(signal)) if np.isfinite(signal) else np.nan,
                "share_of_all_direct_open_value": group_total / total_open if total_open > 0 else np.nan,
                "status": "ok" if not eligible.empty else "insufficient_matches",
            }
        )
    keep = [
        "source_file", "source_sheet", "source_row_number", "PartKey",
        "comparison_entity_id", "direct_cost_category", "po_date",
        "committed_unit_price", "remaining_open_qty", "remaining_open_spend",
        "prior_realized_date", "prior_realized_price", "gap_days",
        "log_price_change", "annualized_log_change", "matched", "match_tier",
    ]
    return matched[[c for c in keep if c in matched.columns]], pd.DataFrame(summaries)
