"""Part-period aggregation, matched baskets, winsorization, and robust Törnqvist."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation.historical_periods import (
    PeriodWindow,
    chain_and_annualize,
    default_comparison_pairs,
    mask_window,
)
from parts_inflation.historical_scope import scope_column

logger = logging.getLogger(__name__)


@dataclass
class IndexSettings:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    weight_cap_quantile: float = 0.95
    qty_comparable_low: float = 0.5
    qty_comparable_high: float = 2.0
    no_change_abs_log: float = 1e-12
    low_reliability_min_matches: int = 100
    low_reliability_min_coverage: float = 0.20


def _mode_first(series: pd.Series) -> Any:
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]


def part_period_values(df: pd.DataFrame, window: PeriodWindow) -> pd.DataFrame:
    """
    Representative part-period values within ``window``.

    - price: median Cost
    - spend: sum of max(PO Value, 0)
    - qty: median positive Qty Ordered
    """
    m = mask_window(df["po_date"], window)
    use = df.loc[m].copy()
    if use.empty:
        return pd.DataFrame()

    use["price"] = pd.to_numeric(use["price"], errors="coerce")
    use = use.loc[np.isfinite(use["price"]) & (use["price"] > 0) & use["PartKey"].notna()].copy()
    if use.empty:
        return pd.DataFrame()

    use["po_value_nonneg"] = pd.to_numeric(use["po_value"], errors="coerce").fillna(0).clip(lower=0)
    use["qty_pos"] = pd.to_numeric(use["qty_ordered"], errors="coerce")
    use.loc[~(use["qty_pos"] > 0), "qty_pos"] = np.nan
    use["po_date"] = pd.to_datetime(use["po_date"], errors="coerce")
    if "raw_part_number" not in use.columns:
        use["raw_part_number"] = use["PartKey"]
    if "Description 1" not in use.columns:
        use["Description 1"] = ""
    if "hist_category" not in use.columns:
        use["hist_category"] = use["description_norm"]

    agg = (
        use.groupby("PartKey", sort=False)
        .agg(
            price=("price", "median"),
            spend=("po_value_nonneg", "sum"),
            qty=("qty_pos", "median"),
            line_count=("price", "size"),
            unique_price_count=("price", "nunique"),
            min_po_date=("po_date", "min"),
            max_po_date=("po_date", "max"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            price_dispersion=("price", lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0),
            hist_category=("hist_category", _mode_first),
            description_norm=("description_norm", "first"),
            desc1=("Description 1", _mode_first),
            raw_part_number=("raw_part_number", "first"),
        )
        .reset_index()
        .rename(columns={"desc1": "Description 1"})
    )
    agg["period_label"] = window.label
    agg["period_start"] = window.start
    agg["period_end"] = window.end
    return agg


def _scope_total_spend(df: pd.DataFrame, window: PeriodWindow) -> float:
    m = mask_window(df["po_date"], window)
    spend = pd.to_numeric(df.loc[m, "po_value"], errors="coerce").fillna(0).clip(lower=0)
    return float(spend.sum())


def build_matched_pair(
    period_a: pd.DataFrame,
    period_b: pd.DataFrame,
    scope_df: pd.DataFrame,
    window_a: PeriodWindow,
    window_b: PeriodWindow,
    comparison_label: str,
    settings: IndexSettings,
) -> pd.DataFrame:
    """Matched-part intersection with log relatives, shares, and flags."""
    if period_a.empty or period_b.empty:
        return pd.DataFrame()

    a = period_a.rename(
        columns={
            "price": "price_a",
            "spend": "spend_a",
            "qty": "qty_a",
            "line_count": "line_count_a",
            "unique_price_count": "unique_price_count_a",
            "min_po_date": "min_po_date_a",
            "max_po_date": "max_po_date_a",
            "price_min": "price_min_a",
            "price_max": "price_max_a",
            "price_dispersion": "price_dispersion_a",
        }
    )
    b = period_b.rename(
        columns={
            "price": "price_b",
            "spend": "spend_b",
            "qty": "qty_b",
            "line_count": "line_count_b",
            "unique_price_count": "unique_price_count_b",
            "min_po_date": "min_po_date_b",
            "max_po_date": "max_po_date_b",
            "price_min": "price_min_b",
            "price_max": "price_max_b",
            "price_dispersion": "price_dispersion_b",
        }
    )
    keep_a = [
        "PartKey",
        "price_a",
        "spend_a",
        "qty_a",
        "line_count_a",
        "unique_price_count_a",
        "min_po_date_a",
        "max_po_date_a",
        "price_min_a",
        "price_max_a",
        "price_dispersion_a",
        "hist_category",
        "description_norm",
        "Description 1",
        "raw_part_number",
    ]
    keep_b = [
        "PartKey",
        "price_b",
        "spend_b",
        "qty_b",
        "line_count_b",
        "unique_price_count_b",
        "min_po_date_b",
        "max_po_date_b",
        "price_min_b",
        "price_max_b",
        "price_dispersion_b",
    ]
    m = a[keep_a].merge(b[keep_b], on="PartKey", how="inner")
    m = m[(m["price_a"] > 0) & (m["price_b"] > 0)].copy()
    if m.empty:
        return m

    m["comparison"] = comparison_label
    m["period_a"] = window_a.label
    m["period_b"] = window_b.label
    m["r"] = np.log(m["price_b"] / m["price_a"])
    m["g"] = np.expm1(m["r"])
    m["no_price_change"] = np.abs(m["r"]) < settings.no_change_abs_log

    # Quantity comparable
    q_ratio = m["qty_b"] / m["qty_a"]
    m["qty_ratio"] = q_ratio
    m["quantity_comparable"] = (
        m["qty_a"].notna()
        & m["qty_b"].notna()
        & (m["qty_a"] > 0)
        & (m["qty_b"] > 0)
        & (q_ratio >= settings.qty_comparable_low)
        & (q_ratio <= settings.qty_comparable_high)
    )

    # Winsorize within this matched basket
    r = m["r"].to_numpy(dtype=float)
    L = float(np.quantile(r, settings.winsor_lower))
    U = float(np.quantile(r, settings.winsor_upper))
    m["winsor_L"] = L
    m["winsor_U"] = U
    m["r_raw"] = m["r"]
    m["r_W"] = np.clip(m["r"], L, U)
    m["is_price_outlier"] = (m["r"] < L) | (m["r"] > U)

    # Spend shares within matched basket
    va = m["spend_a"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    vb = m["spend_b"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    sum_a = va.sum()
    sum_b = vb.sum()
    m["s_a"] = va / sum_a if sum_a > 0 else np.nan
    m["s_b"] = vb / sum_b if sum_b > 0 else np.nan
    m["s_bar"] = 0.5 * (m["s_a"] + m["s_b"])

    # Cap averaged share
    sbar = m["s_bar"].to_numpy(dtype=float)
    if np.isfinite(sbar).any():
        cap = float(np.nanquantile(sbar, settings.weight_cap_quantile))
        c = np.minimum(sbar, cap)
        c_sum = c.sum()
        m["s_bar_capped_raw"] = c
        m["s_tilde"] = c / c_sum if c_sum > 0 else np.nan
        m["weight_cap"] = cap
    else:
        m["s_bar_capped_raw"] = np.nan
        m["s_tilde"] = np.nan
        m["weight_cap"] = np.nan

    # Geometric-spend weights
    h = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    if len(h) and h.sum() > 0:
        h_cap_q = float(np.quantile(h, settings.weight_cap_quantile))
        h_cap = np.minimum(h, h_cap_q)
        m["h"] = h
        m["h_cap"] = h_cap
        m["h_tilde"] = h_cap / h_cap.sum()
    else:
        m["h"] = h
        m["h_cap"] = h
        m["h_tilde"] = np.nan

    m["contrib_headline_log"] = m["s_tilde"] * m["r_W"]

    # Coverage denominators = full scope spend in each period (not renormalized to matched)
    scope_spend_a = _scope_total_spend(scope_df, window_a)
    scope_spend_b = _scope_total_spend(scope_df, window_b)
    m["scope_spend_a"] = scope_spend_a
    m["scope_spend_b"] = scope_spend_b
    m["coverage_a"] = float(va.sum() / scope_spend_a) if scope_spend_a > 0 else np.nan
    m["coverage_b"] = float(vb.sum() / scope_spend_b) if scope_spend_b > 0 else np.nan

    return m.reset_index(drop=True)


def summarize_matched(matched: pd.DataFrame, settings: IndexSettings) -> dict[str, Any]:
    """Compute all required index variants for one matched basket."""
    if matched is None or matched.empty:
        return {
            "matched_parts": 0,
            "headline_robust_capped_tornqvist": np.nan,
            "status": "empty",
        }

    r = matched["r"].to_numpy(dtype=float)
    r_w = matched["r_W"].to_numpy(dtype=float)
    s_bar = matched["s_bar"].to_numpy(dtype=float)
    s_tilde = matched["s_tilde"].to_numpy(dtype=float)
    h_tilde = matched["h_tilde"].to_numpy(dtype=float)

    pi_t_raw = float(np.expm1(np.nansum(s_bar * r)))
    pi_t_w = float(np.expm1(np.nansum(s_bar * r_w)))
    pi_t_cap = float(np.expm1(np.nansum(s_tilde * r_w)))
    pi_equal_w = float(np.expm1(np.nanmean(r_w)))
    pi_equal_raw = float(np.expm1(np.nanmean(r)))
    pi_geo_spend = float(np.expm1(np.nansum(h_tilde * r_w)))

    # Quantity-comparable subset: re-winsorize within subset
    qc = matched.loc[matched["quantity_comparable"]].copy()
    if len(qc) >= 2:
        rq = qc["r"].to_numpy(dtype=float)
        Lq = float(np.quantile(rq, settings.winsor_lower))
        Uq = float(np.quantile(rq, settings.winsor_upper))
        rq_w = np.clip(rq, Lq, Uq)
        pi_qc = float(np.expm1(np.nanmean(rq_w)))
        qc_n = int(len(qc))
    else:
        pi_qc = float("nan")
        qc_n = int(len(qc))

    return {
        "comparison": matched["comparison"].iloc[0],
        "period_a": matched["period_a"].iloc[0],
        "period_b": matched["period_b"].iloc[0],
        "matched_parts": int(len(matched)),
        "median_pct_change": float(np.expm1(np.median(r))),
        "p25_pct_change": float(np.expm1(np.quantile(r, 0.25))),
        "p75_pct_change": float(np.expm1(np.quantile(r, 0.75))),
        "pct_no_price_change": float(matched["no_price_change"].mean()),
        "matched_spend_a": float(matched["spend_a"].fillna(0).clip(lower=0).sum()),
        "matched_spend_b": float(matched["spend_b"].fillna(0).clip(lower=0).sum()),
        "coverage_a": float(matched["coverage_a"].iloc[0]),
        "coverage_b": float(matched["coverage_b"].iloc[0]),
        "winsor_L": float(matched["winsor_L"].iloc[0]),
        "winsor_U": float(matched["winsor_U"].iloc[0]),
        "weight_cap": float(matched["weight_cap"].iloc[0]),
        "raw_tornqvist": pi_t_raw,
        "winsorized_tornqvist": pi_t_w,
        "headline_robust_capped_tornqvist": pi_t_cap,
        "capped_geometric_spend": pi_geo_spend,
        "equal_part_winsorized": pi_equal_w,
        "equal_part_raw": pi_equal_raw,
        "similar_quantity_equal_winsorized": pi_qc,
        "quantity_comparable_parts": qc_n,
        "status": "ok",
    }


def compute_scope_comparisons(
    scoped: pd.DataFrame,
    scope: str,
    settings: IndexSettings,
    ytd_end=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run all default period comparisons for one scope.

    Returns (summary_rows, matched_detail, chain_info).
    """
    col = scope_column(scope)
    use = scoped.loc[scoped[col]].copy()
    pairs = default_comparison_pairs(ytd_end) if ytd_end is not None else default_comparison_pairs()

    summaries = []
    details = []
    headline_by_label: dict[str, float] = {}

    for wa, wb, label in pairs:
        pa = part_period_values(use, wa)
        pb = part_period_values(use, wb)
        matched = build_matched_pair(pa, pb, use, wa, wb, label, settings)
        if matched.empty:
            summaries.append(
                {
                    "scope": scope,
                    "comparison": label,
                    "matched_parts": 0,
                    "headline_robust_capped_tornqvist": np.nan,
                    "status": "empty",
                }
            )
            continue
        matched = matched.copy()
        matched["scope"] = scope
        summary = summarize_matched(matched, settings)
        summary["scope"] = scope
        summaries.append(summary)
        details.append(matched)
        headline_by_label[label] = summary["headline_robust_capped_tornqvist"]

    # Chain complete-year links only
    r1 = headline_by_label.get("FY2023 → FY2024", np.nan)
    r2 = headline_by_label.get("FY2024 → FY2025", np.nan)
    if np.isfinite(r1) and np.isfinite(r2):
        cum, ann = chain_and_annualize([r1, r2])
    else:
        cum, ann = float("nan"), float("nan")

    chain = {
        "scope": scope,
        "fy2023_to_fy2024": r1,
        "fy2024_to_fy2025": r2,
        "cumulative_fy2023_to_fy2025": cum,
        "annualized_fy2023_to_fy2025": ann,
        "fy2026_ytd_yoy": headline_by_label.get("FY2025 YTD → FY2026 YTD", np.nan),
    }

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_df, detail_df, chain


def category_comparisons(
    scoped: pd.DataFrame,
    settings: IndexSettings,
    ytd_end=None,
) -> pd.DataFrame:
    """Physical-input category sensitivity using exact description_norm categories."""
    use = scoped.loc[scoped["in_physical_inputs"]].copy()
    if use.empty:
        return pd.DataFrame()

    cats = sorted(use["description_norm"].dropna().unique())
    pairs = default_comparison_pairs(ytd_end) if ytd_end is not None else default_comparison_pairs()
    rows = []
    for cat in cats:
        sub = use.loc[use["description_norm"] == cat].copy()
        sub["in_physical_inputs"] = True  # already filtered
        for wa, wb, label in pairs:
            # Temporarily treat as its own scope frame
            pa = part_period_values(sub, wa)
            pb = part_period_values(sub, wb)
            matched = build_matched_pair(pa, pb, sub, wa, wb, label, settings)
            if matched.empty:
                rows.append(
                    {
                        "category": cat,
                        "comparison": label,
                        "matched_parts": 0,
                        "headline_robust_capped_tornqvist": np.nan,
                        "coverage_a": np.nan,
                        "coverage_b": np.nan,
                        "low_reliability": True,
                        "reliability_flag": "Low reliability",
                    }
                )
                continue
            summary = summarize_matched(matched, settings)
            low = (
                summary["matched_parts"] < settings.low_reliability_min_matches
                or (summary["coverage_a"] or 0) < settings.low_reliability_min_coverage
                or (summary["coverage_b"] or 0) < settings.low_reliability_min_coverage
            )
            rows.append(
                {
                    "category": cat,
                    "comparison": label,
                    "matched_parts": summary["matched_parts"],
                    "coverage_a": summary["coverage_a"],
                    "coverage_b": summary["coverage_b"],
                    "headline_robust_capped_tornqvist": summary["headline_robust_capped_tornqvist"],
                    "equal_part_winsorized": summary["equal_part_winsorized"],
                    "capped_geometric_spend": summary["capped_geometric_spend"],
                    "low_reliability": low,
                    "reliability_flag": "Low reliability" if low else "OK",
                }
            )
    return pd.DataFrame(rows)


def spend_reconciliation(scoped: pd.DataFrame) -> pd.DataFrame:
    """Compare sum(PO Value) vs sum(Cost × Qty Ordered) by fiscal year for physical inputs."""
    use = scoped.loc[scoped["in_physical_inputs"]].copy()
    if use.empty:
        return pd.DataFrame()
    use = use.copy()
    use["fy"] = use["fiscal_year_label"]
    use["po_value_num"] = pd.to_numeric(use["po_value"], errors="coerce")
    use["cost_x_qty"] = pd.to_numeric(use["price"], errors="coerce") * pd.to_numeric(
        use["qty_ordered"], errors="coerce"
    )
    # Ratio diagnostics
    with np.errstate(divide="ignore", invalid="ignore"):
        use["ratio"] = use["cost_x_qty"] / use["po_value_num"]
    rows = []
    for fy, grp in use.groupby("fy"):
        if fy is None or (isinstance(fy, float) and np.isnan(fy)):
            continue
        po_sum = float(grp["po_value_num"].fillna(0).sum())
        cxq = float(grp["cost_x_qty"].fillna(0).sum())
        # Large-ratio drivers
        large = grp["ratio"].notna() & ((grp["ratio"] > 10) | (grp["ratio"] < 0.1))
        rows.append(
            {
                "period": fy,
                "sum_po_value": po_sum,
                "sum_cost_x_qty_ordered": cxq,
                "ratio_aggregate": cxq / po_sum if po_sum else np.nan,
                "rows": int(len(grp)),
                "rows_with_large_ratio": int(large.sum()),
                "pct_rows_large_ratio": float(large.mean()) if len(grp) else np.nan,
                "note": "Use PO Value for weights; Cost×Qty Ordered is diagnostic only",
            }
        )
    return pd.DataFrame(rows)
