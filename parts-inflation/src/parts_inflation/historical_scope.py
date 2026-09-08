"""Exact historical-actuals scope membership from normalized Description values."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd

from parts_inflation.config import ResolvedConfig, ScopeMode

# Exact membership for the historical prototype (normalized Description)
INVENTORY_ONLY_DESCRIPTIONS = frozenset({"INVENTORY"})

PHYSICAL_INPUT_DESCRIPTIONS = frozenset(
    {
        "INVENTORY",
        "PRODUCTION SUPPLIES",
        "OPERATING SUPPLIES",
        "PRODUCTION AIDS",
        "SMALL TOOLS",
    }
)

# Categories excluded from physical inputs (disclosure / documentation)
EXCLUDED_FROM_PHYSICAL_EXAMPLES = frozenset(
    {
        "MACHINERY & EQUIPMENT",
        "PRODUCT TESTING/R&D",
        "REPAIRS & MAINT.-MACH & EQUIP",
        "COST OF SALES - OUTSIDE SERVICES",
        "SHIPPING RELATED",
        "OFFICE SUPPLIES",
        "COST OF SALES - PLATING OUTSIDE SERVICES",
        "UNSURE - INCLUDE",
        "REPAIRS & MAINT. - BUILDING & OFFICE",
        "FURNISHINGS & FIXTURES",
    }
)


def normalize_description(value: Any) -> Optional[str]:
    """Uppercase, trim, collapse repeated whitespace (Unicode NFKC)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    s = unicodedata.normalize("NFKC", str(value))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.upper()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def apply_historical_scope(
    cleaned: pd.DataFrame,
    config: Optional[ResolvedConfig] = None,
) -> pd.DataFrame:
    """
    Attach normalized Description and boolean scope columns for historical actuals.

    Exact Description membership defines inventory_only / physical_inputs.
    Manual overrides from config Scope Mapping (when present) win for Include/Exclude.
    """
    df = cleaned.copy()
    df["description_norm"] = df["Description"].map(normalize_description)

    # Base exact membership
    df["hist_inventory_only"] = df["description_norm"].isin(INVENTORY_ONLY_DESCRIPTIONS)
    df["hist_physical_inputs"] = df["description_norm"].isin(PHYSICAL_INPUT_DESCRIPTIONS)
    # All PO lines: valid price obs (applied after usable filter below)
    df["hist_all_po_lines"] = True

    # Honor manual overrides from scope mapping when available
    if config is not None and config.scope_mapping is not None and not config.scope_mapping.empty:
        mapping = config.scope_mapping.copy()
        for col in ["Bucket", "Description", "Manual Override", "Default Scope Decision"]:
            if col not in mapping.columns:
                mapping[col] = ""
        from parts_inflation.classify import resolved_scope_decision

        mapping["Resolved Decision"] = mapping.apply(resolved_scope_decision, axis=1)
        merge_cols = ["Bucket", "Description", "Resolved Decision", "Manual Override"]
        if "Physical Input Category" in mapping.columns:
            merge_cols.append("Physical Input Category")
        # Avoid duplicate Manual Override column names
        merge_cols = list(dict.fromkeys(merge_cols))
        df = df.merge(mapping[merge_cols], on=["Bucket", "Description"], how="left", suffixes=("", "_map"))
        manual = df.get("Manual Override", pd.Series("", index=df.index)).astype(str).str.strip()
        # Only explicit Manual Override changes exact Description membership
        excl = manual.str.lower().isin({"exclude"})
        incl = manual.str.lower().isin({"include"})
        df.loc[excl, "hist_inventory_only"] = False
        df.loc[excl, "hist_physical_inputs"] = False
        # Explicit Include adds inventory/physical only when description already matches those sets
        # (does not expand beyond the exact five physical categories)
        df.loc[incl & df["description_norm"].isin(INVENTORY_ONLY_DESCRIPTIONS), "hist_inventory_only"] = True
        df.loc[incl & df["description_norm"].isin(PHYSICAL_INPUT_DESCRIPTIONS), "hist_physical_inputs"] = True
        if "Physical Input Category" in df.columns:
            df["hist_category"] = df["Physical Input Category"].fillna(df["description_norm"])
        else:
            df["hist_category"] = df["description_norm"]
        df["Resolved Decision"] = df["Resolved Decision"].fillna(
            pd.Series(
                np.where(
                    df["hist_physical_inputs"],
                    "Include",
                    np.where(
                        df["description_norm"].isin(EXCLUDED_FROM_PHYSICAL_EXAMPLES),
                        "Exclude",
                        "Needs Review",
                    ),
                ),
                index=df.index,
            )
        )
    else:
        df["hist_category"] = df["description_norm"]
        df["Resolved Decision"] = np.where(
            df["hist_physical_inputs"],
            "Include",
            np.where(df["description_norm"].isin(EXCLUDED_FROM_PHYSICAL_EXAMPLES), "Exclude", "Needs Review"),
        )

    # Restrict to usable price observations for scope flags used in analysis
    usable = df["included_for_pricing"] if "included_for_pricing" in df.columns else True
    df["in_inventory_only"] = usable & df["hist_inventory_only"]
    df["in_physical_inputs"] = usable & df["hist_physical_inputs"]
    df["in_all_po_lines"] = usable & df["hist_all_po_lines"]

    return df


def scope_column(scope: str | ScopeMode) -> str:
    if isinstance(scope, ScopeMode):
        scope = scope.value
    mapping = {
        "inventory_only": "in_inventory_only",
        "physical_inputs": "in_physical_inputs",
        "all_po_lines": "in_all_po_lines",
    }
    if scope not in mapping:
        raise ValueError(f"Unknown scope: {scope}")
    return mapping[scope]


def scope_disclosure(df: pd.DataFrame) -> pd.DataFrame:
    """PO value by normalized Description with include/exclude flags for physical inputs."""
    use = df.loc[df.get("included_for_pricing", True)].copy()
    if use.empty:
        return pd.DataFrame()
    g = (
        use.groupby("description_norm", dropna=False)
        .agg(
            row_count=("PartKey", "size"),
            po_value=("po_value", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0).sum())),
        )
        .reset_index()
    )
    g["in_physical_inputs"] = g["description_norm"].isin(PHYSICAL_INPUT_DESCRIPTIONS)
    g["in_inventory_only"] = g["description_norm"].isin(INVENTORY_ONLY_DESCRIPTIONS)
    g["scope_role"] = np.where(
        g["in_physical_inputs"],
        "included_physical_inputs",
        "excluded_from_physical_inputs",
    )
    return g.sort_values("po_value", ascending=False).reset_index(drop=True)
