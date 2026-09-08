"""Comparable-item identities and conservative part-family candidates."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


STOP_WORDS = {
    "A", "AN", "AND", "FOR", "OF", "THE", "TO", "WITH", "ASSY", "ASSEMBLY",
}


def normalize_description(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def description_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token for token in normalize_description(value).split()
        if len(token) >= 2 and token not in STOP_WORDS
    )


def part_family_prefix(part: object) -> str:
    """Return a conservative alphabetic/alphanumeric family prefix."""
    if part is None or (isinstance(part, float) and pd.isna(part)):
        return ""
    text = str(part).strip().upper()
    first = re.split(r"[\s/_.-]+", text, maxsplit=1)[0]
    match = re.match(r"[A-Z]{2,}[A-Z0-9]*", first)
    return match.group(0)[:12] if match else ""


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def propose_part_family_candidates(
    classified: pd.DataFrame,
    minimum_description_similarity: float = 0.60,
    maximum_log_price_dispersion: float = 0.35,
) -> pd.DataFrame:
    """Propose, but never activate, reviewable part-family mappings.

    Candidates must share the client-approved direct-cost category and a
    meaningful part prefix. Description and contemporaneous price evidence are
    exported so a Glenair reviewer can approve or reject each group.
    """
    required = {"PartKey", "direct_cost_category", "historical_unit_price"}
    if classified.empty or not required.issubset(classified.columns):
        return pd.DataFrame()
    work = classified.loc[
        classified["in_direct_costs"].fillna(False)
        & classified["PartKey"].notna()
        & classified["historical_unit_price"].fillna(0).gt(0)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    d1 = work.get("Description 1", pd.Series("", index=work.index)).fillna("")
    d2 = work.get("Description 2", pd.Series("", index=work.index)).fillna("")
    work["description_normalized"] = (d1.astype(str) + " " + d2.astype(str)).map(
        normalize_description
    )
    work["prefix"] = work["PartKey"].map(part_family_prefix)
    work["fiscal_year"] = np.where(
        pd.to_datetime(work["effective_historical_date"]).dt.month >= 10,
        pd.to_datetime(work["effective_historical_date"]).dt.year + 1,
        pd.to_datetime(work["effective_historical_date"]).dt.year,
    )
    part = (
        work.groupby(["direct_cost_category", "prefix", "PartKey"], as_index=False)
        .agg(
            description_normalized=("description_normalized", "first"),
            median_price=("historical_unit_price", "median"),
            observations=("PartKey", "size"),
            first_fy=("fiscal_year", "min"),
            last_fy=("fiscal_year", "max"),
        )
    )
    rows: list[dict] = []
    candidate_number = 0
    for (category, prefix), group in part.groupby(
        ["direct_cost_category", "prefix"], sort=True
    ):
        if not prefix or group["PartKey"].nunique() < 2 or group["PartKey"].nunique() > 100:
            continue
        descriptions = group["description_normalized"].tolist()
        similarities = []
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                similarities.append(
                    jaccard_similarity(description_tokens(descriptions[i]), description_tokens(descriptions[j]))
                )
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        prices = group["median_price"].to_numpy(float)
        log_dispersion = float(np.std(np.log(prices))) if np.all(prices > 0) else np.inf
        if mean_similarity < minimum_description_similarity or log_dispersion > maximum_log_price_dispersion:
            continue
        candidate_number += 1
        family_id = f"CANDIDATE-{candidate_number:05d}"
        for _, row in group.iterrows():
            rows.append(
                {
                    "candidate_family_id": family_id,
                    "part_key": row["PartKey"],
                    "direct_cost_category": category,
                    "shared_prefix": prefix,
                    "description": row["description_normalized"],
                    "observations": int(row["observations"]),
                    "first_fiscal_year": int(row["first_fy"]),
                    "last_fiscal_year": int(row["last_fy"]),
                    "median_effective_unit_price": float(row["median_price"]),
                    "group_mean_description_similarity": mean_similarity,
                    "group_log_price_dispersion": log_dispersion,
                    "client_approval": "",
                    "review_notes": "",
                    "official_result_included": False,
                }
            )
    return pd.DataFrame(rows)
