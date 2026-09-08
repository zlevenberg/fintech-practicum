"""Scope and category classification with override precedence."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from parts_inflation.config import (
    ResolvedConfig,
    ScopeMode,
    invent_initial_scope_mapping,
)

logger = logging.getLogger(__name__)


def resolved_scope_decision(row: pd.Series) -> str:
    override = row.get("Manual Override")
    if override is not None and str(override).strip() and str(override).strip().lower() not in {"nan", "none"}:
        return str(override).strip().title().replace("Needsreview", "Needs Review")
    default = row.get("Default Scope Decision", "Needs Review")
    return str(default).strip() if default is not None else "Needs Review"


def ensure_scope_mapping(config: ResolvedConfig, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Ensure scope mapping covers all Bucket/Description pairs; invent if empty."""
    pairs = (
        cleaned.groupby(["Bucket", "Description"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    mapping = config.scope_mapping.copy() if config.scope_mapping is not None else pd.DataFrame()
    if mapping.empty:
        mapping = invent_initial_scope_mapping(pairs)
    else:
        # Normalize columns
        for col in [
            "Bucket",
            "Description",
            "Default Scope Decision",
            "Physical Input Category",
            "Reason",
            "Client Approved",
            "Manual Override",
        ]:
            if col not in mapping.columns:
                mapping[col] = "" if col != "Client Approved" else False
        # Add any missing pairs
        existing = set(
            zip(
                mapping["Bucket"].map(lambda x: None if pd.isna(x) else x),
                mapping["Description"].map(lambda x: None if pd.isna(x) else x),
            )
        )
        missing_rows = []
        for _, r in pairs.iterrows():
            key = (
                None if pd.isna(r["Bucket"]) else r["Bucket"],
                None if pd.isna(r["Description"]) else r["Description"],
            )
            if key not in existing:
                from parts_inflation.config import classify_scope_row

                decision, cat, reason = classify_scope_row(r["Bucket"], r["Description"])
                missing_rows.append(
                    {
                        "Bucket": r["Bucket"],
                        "Description": r["Description"],
                        "Default Scope Decision": decision,
                        "Physical Input Category": cat,
                        "Reason": reason,
                        "Client Approved": False,
                        "Manual Override": "",
                    }
                )
        if missing_rows:
            mapping = pd.concat([mapping, pd.DataFrame(missing_rows)], ignore_index=True)

    # Category Mapping overrides
    cat_map = config.category_mapping
    if cat_map is not None and not cat_map.empty and "Physical Input Category" in cat_map.columns:
        key_cols = ["Bucket", "Description"]
        merged = mapping.merge(
            cat_map[key_cols + ["Physical Input Category"]].rename(
                columns={"Physical Input Category": "Category Override"}
            ),
            on=key_cols,
            how="left",
        )
        use = merged["Category Override"].notna() & (merged["Category Override"].astype(str).str.strip() != "")
        merged.loc[use, "Physical Input Category"] = merged.loc[use, "Category Override"]
        mapping = merged.drop(columns=["Category Override"], errors="ignore")

    mapping["Resolved Decision"] = mapping.apply(resolved_scope_decision, axis=1)
    return mapping


def apply_scope_and_category(cleaned: pd.DataFrame, config: ResolvedConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = ensure_scope_mapping(config, cleaned)
    df = cleaned.merge(
        mapping[
            [
                "Bucket",
                "Description",
                "Resolved Decision",
                "Physical Input Category",
                "Default Scope Decision",
                "Manual Override",
                "Reason",
            ]
        ],
        on=["Bucket", "Description"],
        how="left",
    )
    df["Resolved Decision"] = df["Resolved Decision"].fillna("Needs Review")
    df["Physical Input Category"] = df["Physical Input Category"].fillna("Uncategorized")
    df["scope_uncertain"] = df["Resolved Decision"].eq("Needs Review")

    # Part overrides
    overrides = config.part_overrides
    if overrides is not None and not overrides.empty and "PartKey" in overrides.columns:
        ov = overrides.copy()
        ov["PartKey"] = ov["PartKey"].astype(str).str.strip().str.upper()
        df = df.merge(ov, on="PartKey", how="left", suffixes=("", "_ov"))
        if "IncludeExclude" in df.columns:
            excl = df["IncludeExclude"].astype(str).str.upper().eq("EXCLUDE")
            incl = df["IncludeExclude"].astype(str).str.upper().eq("INCLUDE")
            df.loc[excl, "Resolved Decision"] = "Exclude"
            df.loc[incl, "Resolved Decision"] = "Include"
        if "Category" in df.columns:
            use_cat = df["Category"].notna() & (df["Category"].astype(str).str.strip() != "") & (
                df["Category"].astype(str).str.lower() != "nan"
            )
            df.loc[use_cat, "Physical Input Category"] = df.loc[use_cat, "Category"]
        if "ReplacementPartKey" in df.columns:
            use_rep = df["ReplacementPartKey"].notna() & (
                df["ReplacementPartKey"].astype(str).str.strip() != ""
            ) & (df["ReplacementPartKey"].astype(str).str.lower() != "nan")
            df.loc[use_rep, "PartKey"] = (
                df.loc[use_rep, "ReplacementPartKey"].astype(str).str.strip().str.upper()
            )
        if "UoMAdjustmentFactor" in df.columns:
            factor = pd.to_numeric(df["UoMAdjustmentFactor"], errors="coerce")
            adj = factor.notna() & (factor > 0)
            df.loc[adj, "price"] = df.loc[adj, "price"] / factor[adj]
            df.loc[adj, "qty"] = df.loc[adj, "qty"] * factor[adj]
        # Part-level price/qty overrides are carried forward; applied at forecast time
        # so historical pair estimation is not rewritten.
        if "ManualCurrentPrice" in df.columns:
            mcp = pd.to_numeric(df["ManualCurrentPrice"], errors="coerce")
            use_mcp = mcp.notna() & (mcp > 0)
            df["manual_current_price"] = np.where(use_mcp, mcp, np.nan)
            df["price_override_reason"] = np.where(use_mcp, "ManualCurrentPrice", "")
        else:
            df["manual_current_price"] = np.nan
            df["price_override_reason"] = ""
        if "ManualFutureQuantity" in df.columns:
            mfq = pd.to_numeric(df["ManualFutureQuantity"], errors="coerce")
            use_mfq = mfq.notna() & (mfq > 0)
            df["manual_future_quantity"] = np.where(use_mfq, mfq, np.nan)
        else:
            df["manual_future_quantity"] = np.nan
    else:
        df["manual_current_price"] = np.nan
        df["manual_future_quantity"] = np.nan
        df["price_override_reason"] = ""

    # Service-like descriptions should not be treated as physical merely due to part key
    labor_mask = df["service_like_description"].fillna(False) & df["Resolved Decision"].eq("Include")
    df.loc[labor_mask, "scope_warning"] = "service_like_description"
    # For physical_inputs, demote labor-like includes to Needs Review unless manually overridden
    manual = df.get("Manual Override", pd.Series("", index=df.index)).astype(str).str.strip()
    demote = labor_mask & manual.eq("")
    df.loc[demote, "Resolved Decision"] = "Needs Review"
    df.loc[demote, "scope_uncertain"] = True

    scope_mode = config.controls.scope_mode

    def in_scope(decision: str, category: str) -> bool:
        if scope_mode == ScopeMode.all_po_lines:
            return decision != "Exclude"  # still flag services but include usable prices
        if decision == "Exclude":
            return False
        if decision == "Needs Review":
            return False  # excluded by default for inventory/physical; quantified separately
        if scope_mode == ScopeMode.inventory_only:
            return str(category).strip().lower() == "inventory"
        # physical_inputs
        return decision == "Include"

    df["in_selected_scope"] = [
        in_scope(d, c) for d, c in zip(df["Resolved Decision"], df["Physical Input Category"])
    ]
    df["model_eligible"] = df["included_for_pricing"] & df["in_selected_scope"]
    # Parts with a manual current price remain eligible even without an observed price row
    has_mcp = pd.to_numeric(df["manual_current_price"], errors="coerce").notna()
    df.loc[has_mcp & df["in_selected_scope"], "model_eligible"] = True
    df.loc[has_mcp & df["in_selected_scope"], "included_for_pricing"] = True
    df.loc[has_mcp & df["in_selected_scope"], "usable_price_obs"] = True
    df.loc[has_mcp & df["in_selected_scope"], "valid_price"] = True


    # Sensitivity scopes
    df["in_inventory_only"] = (
        df["included_for_pricing"]
        & df["Resolved Decision"].eq("Include")
        & df["Physical Input Category"].astype(str).str.strip().str.lower().eq("inventory")
    )
    df["in_physical_inputs"] = df["included_for_pricing"] & df["Resolved Decision"].eq("Include")
    df["in_all_po_lines"] = df["included_for_pricing"]
    df["in_physical_plus_needs_review"] = df["included_for_pricing"] & df["Resolved Decision"].isin(
        ["Include", "Needs Review"]
    )

    # Modal category per part
    eligible = df.loc[df["PartKey"].notna() & df["Physical Input Category"].notna()]
    if not eligible.empty:
        modal = (
            eligible.groupby(["PartKey", "Physical Input Category"])
            .size()
            .reset_index(name="n")
            .sort_values(["PartKey", "n"], ascending=[True, False])
            .drop_duplicates("PartKey")
            .rename(columns={"Physical Input Category": "modal_category"})
        )
        df = df.merge(modal[["PartKey", "modal_category"]], on="PartKey", how="left")
        # Incompatible category flag
        n_cats = eligible.groupby("PartKey")["Physical Input Category"].nunique()
        multi = set(n_cats[n_cats > 1].index)
        df["incompatible_categories"] = df["PartKey"].isin(multi)
        df["approved_category"] = df["modal_category"].fillna(df["Physical Input Category"])
    else:
        df["modal_category"] = df["Physical Input Category"]
        df["incompatible_categories"] = False
        df["approved_category"] = df["Physical Input Category"]

    logger.info(
        "Scope %s: model_eligible=%s / pricing=%s",
        scope_mode.value,
        int(df["model_eligible"].sum()),
        int(df["included_for_pricing"].sum()),
    )
    return df, mapping
