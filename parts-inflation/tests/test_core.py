"""Unit and integration tests for parts inflation prototype."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parts_inflation.aggregate import aggregate_same_day
from parts_inflation.benchmarks import fisher_index, period_part_prices, tornqvist_index
from parts_inflation.clean import normalize_part_key
from parts_inflation.config import (
    ControlDefaults,
    classify_scope_row,
    load_config,
    project_root,
    write_config_workbook,
)
from parts_inflation.forecast import build_forecast_candidates
from parts_inflation.hierarchy import compute_part_residuals, multiplier_category, multiplier_part
from parts_inflation.matched_pairs import build_adjacent_pairs, fractional_month_weights
from parts_inflation.repeat_sales import fit_repeat_sales
from parts_inflation.report import write_results_workbook
from parts_inflation.uncertainty import bootstrap_composite_and_parts


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "model_config.xlsx"
    write_config_workbook(path)
    return load_config(path)


def test_part_key_normalization():
    assert normalize_part_key("  ab-12  ")[0] == "AB-12"
    assert normalize_part_key("a/b\"c")[0] == 'A/B"C'
    k, numeric, status = normalize_part_key(12345.0)
    assert k == "12345"
    assert numeric is True
    assert status == "ok"
    assert normalize_part_key(None)[2] == "missing"
    assert normalize_part_key("   ")[2] == "empty_after_norm"
    # punctuation preserved
    assert "-" in normalize_part_key("510-66050-29")[0]


def test_scope_mapping_and_override_precedence(cfg):
    d, cat, reason = classify_scope_row("1210AC", "Inventory")
    assert d == "Include"
    d2, _, _ = classify_scope_row("4100AC", "COST OF SALES - OUTSIDE SERVICES")
    assert d2 == "Exclude"
    d3, _, _ = classify_scope_row("1131GC", "Unsure - Include")
    assert d3 == "Needs Review"

    from parts_inflation.classify import resolved_scope_decision

    row = pd.Series(
        {"Default Scope Decision": "Include", "Manual Override": "Exclude"}
    )
    assert resolved_scope_decision(row) == "Exclude"


def test_open_order_handling(cfg):
    from parts_inflation.clean import clean_po_lines

    raw = pd.DataFrame(
        {
            "P.O. Date": ["2024-01-01", "2024-02-01"],
            "Part Number": ["A1", "A1"],
            "Description 1": ["WIDGET", "WIDGET"],
            "Description 2": ["", ""],
            "Cost": [10.0, 11.0],
            "Qty Ordered": [5, 5],
            "Qty Received": [0, 5],
            "Extension (Qty Received)": [0, 55],
            "PO Value": [50, 55],
            "Bucket": ["1210AC", "1210AC"],
            "Description": ["Inventory", "Inventory"],
            "source_file": ["t.xlsx", "t.xlsx"],
            "source_sheet": ["Sheet2", "Sheet2"],
            "source_row_number": [2, 3],
            "raw_part_number": ["A1", "A1"],
            "raw_part_number_type": ["str", "str"],
        }
    )
    cleaned = clean_po_lines(raw, cfg)
    assert cleaned.loc[0, "is_open_order"]
    assert cleaned.loc[0, "usable_price_obs"]
    cfg.controls.include_open_orders_as_prices = False
    cleaned2 = clean_po_lines(raw, cfg)
    assert not cleaned2.loc[0, "usable_price_obs"]


def test_same_day_aggregation(cfg):
    from parts_inflation.classify import apply_scope_and_category
    from parts_inflation.clean import clean_po_lines

    rows = []
    for price, qty in [(10, 1), (20, 3)]:
        rows.append(
            {
                "P.O. Date": "2024-03-01",
                "Part Number": "P1",
                "Description 1": "X",
                "Description 2": "",
                "Cost": price,
                "Qty Ordered": qty,
                "Qty Received": qty,
                "Extension (Qty Received)": price * qty,
                "PO Value": price * qty,
                "Bucket": "1210AC",
                "Description": "Inventory",
                "source_file": "t.xlsx",
                "source_sheet": "Sheet2",
                "source_row_number": 2,
                "raw_part_number": "P1",
                "raw_part_number_type": "str",
            }
        )
    cleaned = clean_po_lines(pd.DataFrame(rows), cfg)
    classified, _ = apply_scope_and_category(cleaned, cfg)
    daily = aggregate_same_day(classified, cfg)
    assert len(daily) == 1
    # qty-weighted mean: (10*1 + 20*3)/4 = 17.5
    assert daily.iloc[0]["price"] == pytest.approx(17.5)
    assert daily.iloc[0]["qty"] == pytest.approx(4)


def test_adjacent_pairs_only():
    daily = pd.DataFrame(
        {
            "PartKey": ["A", "A", "A", "B", "B"],
            "po_date": pd.to_datetime(
                ["2023-01-01", "2023-06-01", "2024-01-01", "2023-01-01", "2023-07-01"]
            ),
            "price": [10, 11, 12, 5, 6],
            "qty": [1, 1, 1, 2, 2],
            "spend": [10, 11, 12, 10, 12],
            "approved_category": ["Inventory"] * 5,
        }
    )
    pairs = build_adjacent_pairs(daily)
    # A has 2 adjacent pairs, B has 1 → 3 total (not C(3,2)=3 for A alone would be wrong if all-pairs)
    assert len(pairs) == 3
    assert set(pairs["PartKey"]) == {"A", "B"}


def test_fractional_month_weights():
    # Full month January inside (Dec 31, Feb 1] → Jan weight 1
    w = fractional_month_weights(pd.Timestamp("2023-12-31"), pd.Timestamp("2024-02-01"))
    assert w[pd.Period("2024-01", "M")] == pytest.approx(1.0)
    # Partial: (Jan 10, Jan 20] → 10 days / 31
    w2 = fractional_month_weights(pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-20"))
    assert w2[pd.Period("2024-01", "M")] == pytest.approx(10 / 31)


def test_tornqvist_hand_calculation():
    # Two parts, two periods
    pp = pd.DataFrame(
        {
            "PartKey": ["A", "B", "A", "B"],
            "period": ["P1", "P1", "P2", "P2"],
            "period_end": pd.to_datetime(["2023-03-31"] * 2 + ["2023-06-30"] * 2),
            "price": [10.0, 20.0, 11.0, 22.0],
            "qty": [2.0, 1.0, 2.0, 1.0],
            "spend": [20.0, 20.0, 22.0, 22.0],
            "approved_category": ["C"] * 4,
        }
    )
    # s1: A=0.5 B=0.5; s2: A=0.5 B=0.5; sbar=0.5
    # log changes: log(1.1), log(1.1) → g = log(1.1)
    tq = tornqvist_index(pp)
    assert len(tq) == 1
    assert tq.iloc[0]["pct_change"] == pytest.approx(0.1, rel=1e-6)


def test_fisher_hand_calculation():
    pp = pd.DataFrame(
        {
            "PartKey": ["A", "B", "A", "B"],
            "period": ["P1", "P1", "P2", "P2"],
            "period_end": pd.to_datetime(["2023-03-31"] * 2 + ["2023-06-30"] * 2),
            "price": [10.0, 20.0, 12.0, 20.0],
            "qty": [1.0, 1.0, 1.0, 2.0],
            "spend": [10, 20, 12, 40],
            "approved_category": ["C"] * 4,
        }
    )
    # L = (12*1 + 20*1)/(10*1+20*1) = 32/30
    # P = (12*1 + 20*2)/(10*1+20*2) = 52/50
    # F = sqrt(L*P)
    L = 32 / 30
    P = 52 / 50
    F = np.sqrt(L * P)
    fish = fisher_index(pp)
    assert fish.iloc[0]["fisher"] == pytest.approx(F)


def test_quantity_adjustment_recovery(cfg):
    # Synthetic: inflation 0, elasticity -0.2 so y = -0.2 * x_q
    rng = np.random.default_rng(0)
    rows = []
    for part in range(40):
        q = 10.0
        p = 100.0
        d0 = pd.Timestamp("2022-01-15")
        for k in range(8):
            d = d0 + pd.DateOffset(months=3 * k)
            rows.append(
                {
                    "PartKey": f"Q{part}",
                    "po_date": d,
                    "price": p,
                    "qty": q,
                    "spend": p * q,
                    "approved_category": "Inventory",
                }
            )
            # next period qty shock and price via elasticity
            xq = rng.normal(0, 0.3)
            q = q * np.exp(xq)
            p = p * np.exp(-0.2 * xq)
    daily = pd.DataFrame(rows)
    pairs = build_adjacent_pairs(daily)
    cfg.controls.fast_mode = True
    cfg.controls.category_min_pairs = 10
    model = fit_repeat_sales(pairs, cfg)
    assert model.n_pairs_used > 0
    assert model.gamma0 == pytest.approx(-0.2, abs=0.15)


def test_repeat_sales_monthly_inflation_recovery(cfg):
    # Constant monthly log inflation of 0.01, no qty effects
    rows = []
    for part in range(30):
        p = 50.0 + part
        d0 = pd.Timestamp("2022-01-01")
        for m in range(24):
            d = d0 + pd.DateOffset(months=m)
            rows.append(
                {
                    "PartKey": f"P{part}",
                    "po_date": d,
                    "price": p * np.exp(0.01 * m),
                    "qty": 1.0,
                    "spend": p * np.exp(0.01 * m),
                    "approved_category": "Inventory",
                }
            )
    pairs = build_adjacent_pairs(pd.DataFrame(rows))
    cfg.controls.fast_mode = True
    cfg.controls.quantity_adjustment_mode = "zero"
    model = fit_repeat_sales(pairs, cfg)
    assert np.mean(model.delta0) == pytest.approx(0.01, abs=0.005)


def test_hierarchical_fallback_and_shrinkage(cfg):
    rows = []
    for part, n in [("RICH", 10), ("SPARSE", 1)]:
        p = 10.0
        for i in range(n + 1):
            rows.append(
                {
                    "PartKey": part,
                    "po_date": pd.Timestamp("2022-01-01") + pd.DateOffset(months=6 * i),
                    "price": p * (1.05 ** i),
                    "qty": 1.0,
                    "spend": p,
                    "approved_category": "Inventory",
                }
            )
    pairs = build_adjacent_pairs(pd.DataFrame(rows))
    cfg.controls.part_min_intervals = 3
    cfg.controls.part_min_span_days = 365
    model = fit_repeat_sales(pairs, cfg)
    hier = compute_part_residuals(pairs, model, cfg)
    rich = hier.loc[hier["PartKey"] == "RICH"].iloc[0]
    sparse = hier.loc[hier["PartKey"] == "SPARSE"].iloc[0]
    assert rich["source"] == "part"
    assert sparse["lambda_shrink"] == 0.0


def test_partial_month_compounding(cfg):
    # Fake model with one month rate
    from parts_inflation.repeat_sales import RepeatSalesResult

    months = [pd.Period("2024-01", "M"), pd.Period("2024-02", "M")]
    model = RepeatSalesResult(
        months=months,
        categories=["Inventory"],
        delta0=np.array([0.0, np.log(1.1)]),  # Feb +10%
        u=np.zeros((1, 2)),
        gamma0=0.0,
        kappa=np.array([0.0]),
        pair_weights=np.array([]),
        sigma=0.1,
        converged=True,
        n_pairs=10,
        n_pairs_used=10,
    )
    # Half of February
    m = multiplier_category(
        model, "Inventory", pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-15")
    )
    # days Feb 1..15 = 15/29 (2024 leap) of Feb
    expected = np.exp((15 / 29) * np.log(1.1))
    assert m == pytest.approx(expected, rel=1e-6)


def test_fixed_basket_composite_identity():
    weights = np.array([0.2, 0.5, 0.3])
    muls = np.array([1.1, 1.0, 1.2])
    M = float(np.sum(weights * muls))
    assert M == pytest.approx(0.2 * 1.1 + 0.5 * 1.0 + 0.3 * 1.2)


def test_no_lookahead_in_backtest_dates(cfg):
    """Ensure last_price_forecast never returns dates after cutoff."""
    from parts_inflation.benchmarks import last_price_forecast

    daily = pd.DataFrame(
        {
            "PartKey": ["A", "A"],
            "po_date": pd.to_datetime(["2023-01-01", "2024-06-01"]),
            "price": [10.0, 12.0],
            "qty": [1, 1],
            "approved_category": ["Inventory", "Inventory"],
        }
    )
    cutoff = pd.Timestamp("2023-12-31")
    last = last_price_forecast(daily, cutoff)
    assert last.iloc[0]["latest_date"] <= cutoff
    assert last.iloc[0]["latest_price"] == 10.0


def test_reproducible_bootstrap(cfg):
    daily = pd.DataFrame(
        {
            "PartKey": [f"P{i%10}" for i in range(60)],
            "po_date": pd.date_range("2022-01-01", periods=60, freq="MS"),
            "price": np.linspace(10, 15, 60),
            "qty": np.ones(60),
            "spend": np.linspace(10, 15, 60),
            "approved_category": ["Inventory"] * 60,
        }
    )
    pairs = build_adjacent_pairs(daily)
    cfg.controls.bootstrap_iterations = 5
    cfg.controls.fast_mode = True
    cfg.controls.random_seed = 42
    weights = pd.Series({f"P{i}": 0.1 for i in range(10)})
    a = bootstrap_composite_and_parts(
        daily, pairs, daily, weights, pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), cfg, "trailing_12m_mean"
    )
    b = bootstrap_composite_and_parts(
        daily, pairs, daily, weights, pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), cfg, "trailing_12m_mean"
    )
    assert np.allclose(a["composite_samples"], b["composite_samples"])


def test_config_cli_default_precedence(tmp_path):
    path = tmp_path / "cfg.xlsx"
    write_config_workbook(path)
    # Config value
    cfg = load_config(path, cli_overrides=None)
    assert cfg.controls.random_seed == 42
    # CLI wins
    cfg2 = load_config(path, cli_overrides={"random_seed": 7})
    assert cfg2.controls.random_seed == 7
    assert cfg2.sources["random_seed"].source == "CLI"


def test_cache_invalidation(tmp_path):
    from parts_inflation.ingest import SourceFileInfo, combined_fingerprint

    p = tmp_path / "a.xlsx"
    p.write_bytes(b"abc")
    info = SourceFileInfo(path=p, size=3, mtime=p.stat().st_mtime, sheet="Sheet2", fingerprint="x")
    fp1 = combined_fingerprint([info], "clean_v1")
    fp2 = combined_fingerprint([info], "clean_v2")
    assert fp1 != fp2


def test_path_handling_posix_and_windows_style():
    from pathlib import PurePosixPath, PureWindowsPath

    # pathlib handles both; ensure project_root is absolute Path
    root = project_root()
    assert isinstance(root, Path)
    mixed = Path("data") / "raw" / "file.xlsx"
    assert "raw" in mixed.as_posix()
    assert PurePosixPath("data/raw/file.xlsx").parts[-1] == "file.xlsx"
    assert PureWindowsPath(r"data\raw\file.xlsx").parts[-1] == "file.xlsx"
    # Runtime code builds paths with pathlib operators (OS-agnostic)
    assert (root / "data" / "raw").is_absolute()


def test_excel_report_has_required_sheets(tmp_path):
    path = tmp_path / "out.xlsx"
    payload = {
        "dashboard": {
            "scope": "physical_inputs",
            "data_period": "2022-2026",
            "base_date": "2026-07-09",
            "target_date": "2027-07-09",
            "composite_p10": 1.01,
            "composite_p50": 1.02,
            "composite_p90": 1.03,
            "annualized_p50": 0.02,
            "cost_change_p50": 0.02,
            "matched_spend_coverage": 0.5,
            "pct_weight_part": 0.2,
            "pct_weight_category": 0.5,
            "pct_weight_overall": 0.3,
            "selected_model": "hierarchical",
            "forecast_method": "trailing_12m_mean",
            "long_horizon_warning": "None",
        },
        "controls_used": pd.DataFrame([{"key": "a", "value": "1", "source": "Default", "description": "d"}]),
        "scope_sensitivity": pd.DataFrame([{"scope": "physical_inputs", "status": "ok"}]),
        "historical_index": pd.DataFrame([{"period": "FY2024", "index": 1.0}]),
        "historical_index_chart": pd.DataFrame(
            [{"period": "FY2024", "index": 1.0, "p10": 1.0, "p50": 1.0, "p90": 1.0}]
        ),
        "category_comparison_chart": pd.DataFrame([{"category": "Inventory", "multiplier_p50": 1.02}]),
        "category_results": pd.DataFrame([{"category": "Inventory", "multiplier_p50": 1.02}]),
        "part_forecasts": pd.DataFrame([{"PartKey": "A", "multiplier_p50": 1.02}]),
        "backtests": pd.DataFrame([{"model": "last_price", "WAPE": 0.1}]),
        "scope_mapping": pd.DataFrame([{"Bucket": "1210AC", "Resolved Decision": "Include"}]),
        "data_quality": pd.DataFrame([{"issue": "open_orders", "row_count": 1}]),
        "run_information": pd.DataFrame([{"item": "x", "value": "y"}]),
        "profile_comparison": pd.DataFrame([{"metric": "total_rows", "actual": 1}]),
    }
    write_results_workbook(path, payload)
    xl = pd.ExcelFile(path)
    required = {
        "Dashboard",
        "Controls Used",
        "Scope Sensitivity",
        "Historical Index",
        "Category Results",
        "Part Forecasts",
        "Backtests",
        "Scope Mapping",
        "Data Quality",
        "Methodology",
        "Run Information",
    }
    assert required.issubset(set(xl.sheet_names))


def test_sheet_discovery_required_columns(tmp_path):
    from openpyxl import Workbook

    from parts_inflation.ingest import select_sheet

    path = tmp_path / "toy.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Other"
    ws.append(["foo", "bar"])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(
        [
            "P.O. Date",
            "Part Number",
            "Description 1",
            "Description 2",
            "Cost",
            "Qty Ordered",
            "Qty Received",
            "Extension (Qty Received)",
            "PO Value",
            "Bucket",
            "Description",
        ]
    )
    ws2.append([datetime(2024, 1, 1), "A", "d", "", 1, 1, 1, 1, 1, "1210AC", "Inventory"])
    wb.save(path)
    name, headers = select_sheet(path)
    assert name == "Sheet2"


@pytest.mark.integration
def test_smoke_real_data_if_present():
    raw_dir = project_root() / "data" / "raw"
    files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xlsm"))
    if len(files) < 1:
        pytest.skip("No raw workbooks present")
    from parts_inflation.ingest import load_all_po_lines

    raw, infos, warnings = load_all_po_lines(raw_dir)
    assert len(raw) > 1000
    # Approximate expected totals when all 4 files present
    if len(infos) >= 4:
        assert abs(len(raw) - 277529) < 5000
        mins = [i.min_date for i in infos if i.min_date is not None]
        maxs = [i.max_date for i in infos if i.max_date is not None]
        assert min(mins) <= pd.Timestamp("2022-12-31")
        assert max(maxs) >= pd.Timestamp("2026-01-01")
    # Usable repeated parts exist after basic clean
    from parts_inflation.clean import clean_po_lines
    from parts_inflation.config import load_config

    cfg_path = project_root() / "config" / "model_config.xlsx"
    cfg = load_config(cfg_path if cfg_path.exists() else None, create_if_missing=True)
    cleaned = clean_po_lines(raw, cfg)
    assert cleaned["PartKey"].nunique() > 1000
    vc = cleaned.dropna(subset=["PartKey"]).groupby("PartKey").size()
    assert (vc >= 2).sum() > 100
