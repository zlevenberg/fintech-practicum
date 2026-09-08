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

CLEANING_VERSION = "clean_v1"


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
    df["invalid_date"] = df["po_date"].isna() & df["P.O. Date"].notna()
    df["missing_date"] = df["P.O. Date"].isna()

    # Numerics
    df["price"] = _to_float(df[ctrls.price_field if ctrls.price_field in df.columns else "Cost"])
    df["qty_ordered"] = _to_float(df["Qty Ordered"])
    df["qty_received"] = _to_float(df["Qty Received"])
    df["po_value"] = _to_float(df["PO Value"])
    df["extension_received"] = _to_float(df["Extension (Qty Received)"])

    df["qty"] = df["qty_ordered"]
    df["qty_fallback_used"] = False
    if ctrls.quantity_fallback_to_received:
        need = df["qty"].isna() | (df["qty"] <= 0)
        can = need & df["qty_received"].notna() & (df["qty_received"] > 0)
        df.loc[can, "qty"] = df.loc[can, "qty_received"]
        df.loc[can, "qty_fallback_used"] = True

    df["calculated_ordered_value"] = df["price"] * df["qty_ordered"]
    df["po_value_matches_ordered"] = (
        df["po_value"].notna()
        & df["calculated_ordered_value"].notna()
        & np.isclose(df["po_value"], df["calculated_ordered_value"], rtol=1e-4, atol=0.01)
    )
    df["po_value_matches_received"] = (
        df["po_value"].notna()
        & df["price"].notna()
        & df["qty_received"].notna()
        & np.isclose(df["po_value"], df["price"] * df["qty_received"], rtol=1e-4, atol=0.01)
    )

    # Open orders
    df["is_open_order"] = (
        (df["qty_received"].fillna(0) <= 0)
        & (df["qty_ordered"].fillna(0) > 0)
        & (df["price"].fillna(0) > 0)
    )

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
        "price",
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
    df["included_for_weights"] = df["included_for_pricing"] & df["po_value"].notna() & (df["po_value"] > 0)
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
