"""Fiscal-year assignment and aligned YTD window helpers for historical actuals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

# Client fiscal year: October 1 – September 30
FISCAL_YEAR_START_MONTH = 10
DEFAULT_YTD_END = date(2026, 7, 9)


@dataclass(frozen=True)
class PeriodWindow:
    """Inclusive calendar window for a comparison period."""

    label: str
    start: date
    end: date
    is_complete_fy: bool = True
    is_aligned_ytd: bool = False

    def contains(self, ts: pd.Timestamp) -> bool:
        if pd.isna(ts):
            return False
        d = pd.Timestamp(ts).date()
        return self.start <= d <= self.end


def fiscal_year(ts: pd.Timestamp | date | None) -> Optional[int]:
    """Return fiscal year number (year of September 30 ending the FY)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    return int(t.year + 1) if t.month >= FISCAL_YEAR_START_MONTH else int(t.year)


def fiscal_year_label(ts: pd.Timestamp | date | None) -> Optional[str]:
    fy = fiscal_year(ts)
    return f"FY{fy}" if fy is not None else None


def fiscal_year_bounds(fy: int) -> tuple[date, date]:
    """Return (start, end) inclusive dates for a complete fiscal year."""
    return date(fy - 1, 10, 1), date(fy, 9, 30)


def assign_fiscal_year(df: pd.DataFrame, date_col: str = "po_date") -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    out["fiscal_year"] = [fiscal_year(d) for d in dates]
    out["fiscal_year_label"] = [fiscal_year_label(d) for d in dates]
    return out


def ytd_window(
    fy: int,
    ytd_end: date = DEFAULT_YTD_END,
) -> PeriodWindow:
    """
    Aligned year-to-date window for fiscal year ``fy`` ending on ``ytd_end``'s
    month/day (or earlier if the FY ends sooner).

    Example with ytd_end=2026-07-09:
      FY2026 YTD → 2025-10-01 .. 2026-07-09
      FY2025 YTD → 2024-10-01 .. 2025-07-09
    """
    fy_start, fy_end = fiscal_year_bounds(fy)
    if ytd_end.month >= FISCAL_YEAR_START_MONTH:
        aligned_end = date(fy - 1, ytd_end.month, ytd_end.day)
    else:
        aligned_end = date(fy, ytd_end.month, ytd_end.day)
    end = min(aligned_end, fy_end)
    return PeriodWindow(
        label=f"FY{fy} YTD",
        start=fy_start,
        end=end,
        is_complete_fy=False,
        is_aligned_ytd=True,
    )


def complete_fy_window(fy: int) -> PeriodWindow:
    start, end = fiscal_year_bounds(fy)
    return PeriodWindow(label=f"FY{fy}", start=start, end=end, is_complete_fy=True)


def default_comparison_pairs(ytd_end: date = DEFAULT_YTD_END) -> list[tuple[PeriodWindow, PeriodWindow, str]]:
    """
    Headline comparison pairs.

    Complete-year chain links: FY2023→FY2024, FY2024→FY2025.
    Separate aligned YTD: FY2025 YTD → FY2026 YTD.
    """
    return [
        (complete_fy_window(2023), complete_fy_window(2024), "FY2023 → FY2024"),
        (complete_fy_window(2024), complete_fy_window(2025), "FY2024 → FY2025"),
        (
            ytd_window(2025, ytd_end),
            ytd_window(2026, ytd_end),
            "FY2025 YTD → FY2026 YTD",
        ),
    ]


def mask_window(dates: pd.Series, window: PeriodWindow) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    return d.notna() & (d >= start) & (d <= end)


def chain_and_annualize(rates: list[float]) -> tuple[float, float]:
    """
    Geometric chain of period rates → (cumulative, annualized).
    ``rates`` are fractional period inflation rates (e.g. 0.0661).
    Annualization uses n = len(rates) complete-year links.
    """
    if not rates:
        return float("nan"), float("nan")
    m = 1.0
    for r in rates:
        m *= 1.0 + float(r)
    cumulative = m - 1.0
    n = len(rates)
    annualized = m ** (1.0 / n) - 1.0 if n > 0 else float("nan")
    return float(cumulative), float(annualized)
