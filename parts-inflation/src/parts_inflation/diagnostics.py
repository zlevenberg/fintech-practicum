"""Data-quality diagnostics and profiling summaries."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation.ingest import SourceFileInfo


def build_data_quality_table(
    cleaned: pd.DataFrame,
    classified: pd.DataFrame,
    pairs: pd.DataFrame,
    infos: list[SourceFileInfo],
    warnings: list[str],
    base_date: Optional[pd.Timestamp] = None,
    last_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    def spend(mask) -> float:
        s = classified.loc[mask, "po_value"]
        return float(s.fillna(0).sum())

    rows = []

    def add(issue: str, mask, note: str = "") -> None:
        m = mask if hasattr(mask, "sum") else pd.Series(False, index=classified.index)
        rows.append(
            {
                "issue": issue,
                "row_count": int(m.sum()),
                "po_value_impact": spend(m),
                "share_of_rows": float(m.mean()) if len(m) else 0.0,
                "notes": note,
            }
        )

    add("invalid_or_missing_date", classified["po_date"].isna())
    add("invalid_price", ~classified["valid_price"])
    add("missing_part_key", classified["PartKey"].isna())
    add("numeric_origin_part_numbers", classified["part_numeric_origin"].fillna(False))
    add("open_orders", classified["is_open_order"].fillna(False))
    add("exact_duplicate_rows", classified["is_exact_duplicate"].fillna(False))
    add("incompatible_categories", classified["incompatible_categories"].fillna(False))
    add("uncertain_scope_needs_review", classified["scope_uncertain"].fillna(False))
    add("service_like_description", classified["service_like_description"].fillna(False))
    add(
        "po_value_not_equal_cost_x_qty_ordered",
        ~classified["po_value_matches_ordered"].fillna(False),
        "Within rtol=1e-4 atol=0.01",
    )
    if not pairs.empty and "extreme_flag" in pairs.columns:
        rows.append(
            {
                "issue": "extreme_price_change_pairs",
                "row_count": int(pairs["extreme_flag"].sum()),
                "po_value_impact": float(
                    pairs.loc[pairs["extreme_flag"], "spend_b"].fillna(0).sum()
                ),
                "share_of_rows": float(pairs["extreme_flag"].mean()),
                "notes": "Adjacent pairs flagged by ratio or MAD rules",
            }
        )

    # Stale prices: latest observation more than 180 days before base date
    if last_prices is not None and not last_prices.empty and "price_staleness_days" in last_prices.columns:
        stale = last_prices["price_staleness_days"].fillna(0) > 180
        spend_impact = float(
            (
                last_prices.loc[stale, "estimated_base_price"].fillna(0)
                * last_prices.loc[stale, "q_star"].fillna(0)
            ).sum()
        ) if "estimated_base_price" in last_prices.columns and "q_star" in last_prices.columns else 0.0
        rows.append(
            {
                "issue": "stale_prices_gt_180d",
                "row_count": int(stale.sum()),
                "po_value_impact": spend_impact,
                "share_of_rows": float(stale.mean()) if len(stale) else 0.0,
                "notes": f"Relative to base_date={base_date}",
            }
        )

    rows.append(
        {
            "issue": "source_file_warnings",
            "row_count": len(warnings),
            "po_value_impact": 0.0,
            "share_of_rows": np.nan,
            "notes": "; ".join(warnings[:10]),
        }
    )
    return pd.DataFrame(rows)


def profile_summary(classified: pd.DataFrame, infos: list[SourceFileInfo]) -> dict[str, Any]:
    dates = classified["po_date"].dropna()
    parts = classified.loc[classified["PartKey"].notna(), "PartKey"]
    # Fiscal period presence approx by source file
    by_file = classified.groupby("source_file")["PartKey"].nunique()
    # Multi-year parts: appear in >=2 source files
    part_files = classified.dropna(subset=["PartKey"]).groupby("PartKey")["source_file"].nunique()
    return {
        "total_rows": len(classified),
        "distinct_parts": int(parts.nunique()),
        "parts_in_ge_2_files": int((part_files >= 2).sum()),
        "parts_in_ge_3_files": int((part_files >= 3).sum()),
        "parts_in_all_4_files": int((part_files >= 4).sum()) if len(infos) >= 4 else int((part_files >= len(infos)).sum()),
        "min_date": dates.min(),
        "max_date": dates.max(),
        "open_orders": int(classified["is_open_order"].sum()),
        "source_files": len(infos),
    }


def compare_to_expected_profile(summary: dict[str, Any]) -> pd.DataFrame:
    expected = [
        ("total_rows", 277529, summary.get("total_rows")),
        ("distinct_parts", 134848, summary.get("distinct_parts")),
        ("parts_in_ge_2_files", 27729, summary.get("parts_in_ge_2_files")),
        ("parts_in_ge_3_files", 9698, summary.get("parts_in_ge_3_files")),
        ("parts_in_all_4_files", 2998, summary.get("parts_in_all_4_files")),
    ]
    rows = []
    for name, exp, act in expected:
        rows.append(
            {
                "metric": name,
                "expected_approx": exp,
                "actual": act,
                "abs_diff": None if act is None else abs(act - exp),
                "rel_diff": None if act in (None, 0) else (act - exp) / exp,
            }
        )
    return pd.DataFrame(rows)
