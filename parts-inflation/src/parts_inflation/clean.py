"""Cleaning, normalization, and open-order / duplicate handling."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation.config import DedupMode, ResolvedConfig

logger = logging.getLogger(__name__)

CLEANING_VERSION = "clean_v2_realized_committed"


def normalize_part_key(value: Any) -> tuple[Optional[str], bool, str]:
    """
    Return (PartKey, numeric_origin_flag, status).
    status: ok | missing | empty_after_norm
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, False, "missing"
    numeric_origin = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    if isinstance(value, bool):
        numeric_origin = False

    # Avoid scientific notation for large numeric Excel cells
    if numeric_origin:
        if isinstance(value, float) and value.is_integer():
            raw = str(int(value))
        elif isinstance(value, (int, np.integer)):
            raw = str(int(value))
        else:
            # Keep decimal representation without sci notation where feasible
            raw = format(float(value), "f").rstrip("0").rstrip(".")
    else:
        raw = str(value)

    s = unicodedata.normalize("NFKC", raw)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.upper()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None, numeric_origin, "empty_after_norm"
    return s, numeric_origin, "ok"


def _to_float(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.abs() > 0))


def _nearest_basis_factor(ratio: float, tolerance: float = 0.03) -> Optional[float]:
    """Return a likely power-of-ten price-basis factor, otherwise None."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    candidates = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    nearest = float(candidates[np.argmin(np.abs(np.log(ratio / candidates)))])
    return nearest if abs(ratio / nearest - 1.0) <= tolerance else None


def clean_po_lines(raw: pd.DataFrame, config: ResolvedConfig) -> pd.DataFrame:
    """Normalize fields, flag issues, and assign inclusion eligibility."""
    df = raw.copy()
    ctrls = config.controls

    # Part key
    keys, numeric_flags, statuses = [], [], []
    for v in df["raw_part_number"] if "raw_part_number" in df.columns else df["Part Number"]:
        k, num, st = normalize_part_key(v)
        keys.append(k)
        numeric_flags.append(num)
        statuses.append(st)
    df["PartKey"] = keys
    df["part_numeric_origin"] = numeric_flags
    df["part_key_status"] = statuses

    # Dates
    df["po_date"] = pd.to_datetime(df["P.O. Date"], errors="coerce")
    receipt_col = next(
        (c for c in ["Receipt Date", "Received Date", "P.O. Receipt Date"] if c in df.columns),
        None,
    )
    df["receipt_date"] = (
        pd.to_datetime(df[receipt_col], errors="coerce") if receipt_col else pd.NaT
    )
    df["effective_historical_date"] = df["receipt_date"].fillna(df["po_date"])
    df["historical_date_proxy"] = df["receipt_date"].isna()
    df["invalid_date"] = df["po_date"].isna() & df["P.O. Date"].notna()
    df["missing_date"] = df["P.O. Date"].isna()

    # Numerics
    df["cost_raw"] = _to_float(df[ctrls.price_field if ctrls.price_field in df.columns else "Cost"])
    df["qty_ordered"] = _to_float(df["Qty Ordered"])
    df["qty_received"] = _to_float(df["Qty Received"])
    df["po_value"] = _to_float(df["PO Value"])
    df["extension_received"] = _to_float(df["Extension (Qty Received)"])

    df["ordered_unit_price_implied"] = _safe_ratio(df["po_value"], df["qty_ordered"])
    df["received_unit_price_implied"] = _safe_ratio(
        df["extension_received"], df["qty_received"]
    )

    has_received = df["qty_received"].fillna(0).gt(0)
    valid_received_implied = (
        has_received
        & df["received_unit_price_implied"].notna()
        & np.isfinite(df["received_unit_price_implied"])
        & df["received_unit_price_implied"].gt(0)
    )
    valid_ordered_implied = (
        df["qty_ordered"].fillna(0).gt(0)
        & df["ordered_unit_price_implied"].notna()
        & np.isfinite(df["ordered_unit_price_implied"])
        & df["ordered_unit_price_implied"].gt(0)
    )
    valid_cost = df["cost_raw"].notna() & np.isfinite(df["cost_raw"]) & df["cost_raw"].gt(0)

    df["historical_unit_price"] = np.where(
        valid_received_implied,
        df["received_unit_price_implied"],
        np.where(has_received & valid_cost, df["cost_raw"], np.nan),
    )
    df["historical_price_source"] = np.select(
        [valid_received_implied, has_received & valid_cost],
        ["received_extension_div_qty", "cost_fallback"],
        default="unavailable",
    )
    df["committed_unit_price"] = np.where(
        valid_ordered_implied, df["ordered_unit_price_implied"], np.where(valid_cost, df["cost_raw"], np.nan)
    )
    df["committed_price_source"] = np.select(
        [valid_ordered_implied, valid_cost],
        ["po_value_div_qty", "cost_fallback"],
        default="unavailable",
    )

    df["qty_fallback_used"] = False
    df["qty"] = np.where(has_received, df["qty_received"], df["qty_ordered"])

    df["calculated_ordered_value"] = df["cost_raw"] * df["qty_ordered"]
    df["calculated_received_value"] = df["cost_raw"] * df["qty_received"]
    df["po_reconciliation_ratio"] = _safe_ratio(df["po_value"], df["calculated_ordered_value"])
    df["received_reconciliation_ratio"] = _safe_ratio(
        df["extension_received"], df["calculated_received_value"]
    )
    df["po_likely_basis_factor"] = df["po_reconciliation_ratio"].map(_nearest_basis_factor)
    df["received_likely_basis_factor"] = df["received_reconciliation_ratio"].map(
        _nearest_basis_factor
    )
    df["po_value_matches_ordered"] = (
        df["po_value"].notna()
        & df["calculated_ordered_value"].notna()
        & np.isclose(df["po_value"], df["calculated_ordered_value"], rtol=1e-4, atol=0.01)
    )
    df["po_value_matches_received"] = (
        df["po_value"].notna()
        & df["cost_raw"].notna()
        & df["qty_received"].notna()
        & np.isclose(df["po_value"], df["cost_raw"] * df["qty_received"], rtol=1e-4, atol=0.01)
    )

    # Open orders
    df["is_open_order"] = (
        (df["qty_received"].fillna(0) <= 0)
        & (df["qty_ordered"].fillna(0) > 0)
        & (df["committed_unit_price"].fillna(0) > 0)
    )
    df["is_partially_received"] = (
        df["qty_received"].fillna(0).gt(0)
        & df["qty_ordered"].fillna(0).gt(df["qty_received"].fillna(0))
    )
    df["remaining_open_qty"] = (
        df["qty_ordered"].fillna(0) - df["qty_received"].fillna(0)
    ).clip(lower=0)
    df["has_open_commitment"] = df["remaining_open_qty"].gt(0) & df["committed_unit_price"].fillna(0).gt(0)
    df["realized_spend"] = np.where(
        valid_received_implied,
        df["extension_received"].clip(lower=0),
        np.where(has_received, df["historical_unit_price"] * df["qty_received"], 0.0),
    )
    df["remaining_open_spend"] = df["remaining_open_qty"] * df["committed_unit_price"]

    # Legacy-compatible generic fields now represent realized history by default.
    df["price"] = df["historical_unit_price"]
    open_fallback = df["price"].isna() & df["is_open_order"] & ctrls.include_open_orders_as_prices
    df.loc[open_fallback, "price"] = df.loc[open_fallback, "committed_unit_price"]

    # Valid price observation
    df["valid_price"] = df["price"].notna() & np.isfinite(df["price"]) & (df["price"] > 0)
    df["valid_qty"] = df["qty"].notna() & np.isfinite(df["qty"]) & (df["qty"] > 0)
    df["usable_price_obs"] = (
        df["valid_price"]
        & df["PartKey"].notna()
        & (~df["po_date"].isna())
        & (
            (~df["is_open_order"])
            | ctrls.include_open_orders_as_prices
        )
    )

    # Duplicate fingerprint (exact row content of key fields)
    fingerprint_cols = [
        "po_date",
        "PartKey",
        "cost_raw",
        "qty_ordered",
        "qty_received",
        "po_value",
        "Bucket",
        "Description",
        "Description 1",
        "Description 2",
    ]
    present = [c for c in fingerprint_cols if c in df.columns]
    df["row_fingerprint"] = pd.util.hash_pandas_object(df[present], index=False).astype(str)
    dup_counts = df.groupby("row_fingerprint")["row_fingerprint"].transform("size")
    df["is_exact_duplicate"] = dup_counts > 1

    if ctrls.deduplication_mode == DedupMode.drop:
        before = len(df)
        df = df.loc[~df.duplicated(subset=["row_fingerprint"], keep="first")].copy()
        logger.info("Dropped %s exact duplicate rows", before - len(df))

    # Labor-in-description flag (part-number-like identifiers that are services)
    desc1 = df["Description 1"].astype(str).str.upper().fillna("")
    desc2 = df.get("Description 2", pd.Series("", index=df.index)).astype(str).str.upper().fillna("")
    combined_desc = desc1 + " " + desc2
    df["service_like_description"] = combined_desc.str.contains(
        r"\bLABOR\b|\bSERVICE\b|\bFREIGHT\b|\bSHIPPING\b", regex=True
    )

    df["exclusion_reason"] = ""
    df.loc[df["PartKey"].isna(), "exclusion_reason"] = "missing_part_key"
    df.loc[df["po_date"].isna(), "exclusion_reason"] = df.loc[df["po_date"].isna(), "exclusion_reason"].where(
        df.loc[df["po_date"].isna(), "exclusion_reason"] != "", "invalid_or_missing_date"
    )
    df.loc[~df["valid_price"] & (df["exclusion_reason"] == ""), "exclusion_reason"] = "invalid_price"
    df.loc[
        df["is_open_order"] & (~ctrls.include_open_orders_as_prices) & (df["exclusion_reason"] == ""),
        "exclusion_reason",
    ] = "open_order_excluded"

    df["included_for_pricing"] = df["usable_price_obs"] & (df["exclusion_reason"] == "")
    # Weight eligibility
    df["included_for_weights"] = (
        df["included_for_pricing"]
        & df["realized_spend"].notna()
        & (df["realized_spend"] > 0)
    )
    if not ctrls.include_open_orders_in_weights:
        df.loc[df["is_open_order"], "included_for_weights"] = False

    logger.info(
        "Cleaned %s rows; usable price obs=%s; open orders=%s; exact dups flagged=%s",
        len(df),
        int(df["included_for_pricing"].sum()),
        int(df["is_open_order"].sum()),
        int(df["is_exact_duplicate"].sum()),
    )
    return df
