# Glenair Direct-Cost Inflation

Version 2 calculates three deliberately separate measures from purchase-order history:

1. **Historical realized inflation** from received purchases.
2. **Current committed-cost pressure** from open PO quantities compared with prior realized prices.
3. **Projected 1-, 2-, and 3-year escalation** for a fixed direct-cost basket.

The official scope is the six client-approved buckets: COS, Inventory, Operating Supplies, Production Supplies, Production Aids, and Small Tooling. Machinery, office supplies, R&D/testing, repairs, freight, and other indirect categories are excluded from the headline.

The four supplied workbooks remain immutable inputs in `data/raw/`. Each run creates a timestamped directory in `outputs/` containing an Excel report, CSV audit tables, JSON configuration/summary files, SHA-256 hashes, and a run manifest.

## Requirements

- Python 3.11 or newer
- Windows 10/11 or macOS
- The four `.xlsx`/`.xlsm` PO workbooks in `data/raw/`
- No local Microsoft Excel installation is required

## Run on macOS

```bash
cd parts-inflation
chmod +x setup_mac.sh run_mac.command
./setup_mac.sh
source .venv/bin/activate
python -m parts_inflation.cli validate
python -m parts_inflation.cli run
```

After setup, `run_mac.command` can also be opened from Finder.

## Run on Windows

```bat
cd parts-inflation
setup_windows.bat
.venv\Scripts\activate
python -m parts_inflation.cli validate
python -m parts_inflation.cli run
```

After setup, `run_windows.bat` can also be double-clicked. The launchers resolve paths relative to the repository, and the package uses `pathlib` for platform-independent paths.

## Main commands

```bash
# Verify inputs and scope without estimating forecasts
python -m parts_inflation.cli validate

# Official v2 run
python -m parts_inflation.cli run

# Faster development run; same estimator and convergence standard
python -m parts_inflation.cli run --fast-mode

# Small deterministic smoke run
python -m parts_inflation.cli run --fast-mode --skip-backtest --bootstrap-iterations 2

# Earlier forecast origin; all learning is truncated at this date
python -m parts_inflation.cli run --base-date 2025-09-30

# Unofficial comparable-part candidates for client review
python -m parts_inflation.cli propose-mappings
```

Explicit CLI flags take precedence over workbook controls, then code defaults. V2 invariants—direct-cost headline scope, realized-only history, open orders excluded from realized weights, and 12/24/36-month horizons—cannot be weakened by legacy settings in an older workbook.

## Method in brief

For consecutive observed purchases of the same exact or client-approved comparison entity:

```text
y_j = ln(p_j,2 / p_j,1)
y_j = sum_m D_jm * delta_m + gamma * ln(q_j,2 / q_j,1) + epsilon_j
```

`D_jm` is the fraction of month `m` crossed by pair `j`. This interval design prevents a multi-month change from being assigned wholly to its endpoint. Estimation uses Huber IRLS, capped spend weights, smoothness/ridge penalties, and downweights extreme pairs without deleting an entire part history.

Overall and bucket paths are fitted separately, which identifies the model. Sparse buckets shrink toward overall. Part-specific perpetual trends are disabled by default.

Annual bucket forecasts also have configurable publication guardrails (default −25% to +50%); every application is flagged in the forecast audit.

The projected composite is a fixed-basket index:

```text
M(T0,T) = sum_i q_i* p_i(T) / sum_i q_i* p_i(T0)
```

The `Planned Basket` sheet can define `q_i*`; otherwise the latest complete fiscal year's realized spend is the documented proxy. Annual composite multipliers are weighted arithmetic means of bucket price multipliers, not averages of rates.

Forecast methods are selected by rolling-origin fixed-basket composite error. Each cutoff contributes one error per method, future records cannot affect its training/mapping/weights, and `last_price = 0% inflation` is not eligible. Uncertainty uses comparison-entity cluster bootstrap with full refitting.

See [Methodology](docs/METHODOLOGY.md) for full equations, controls, acceptance rules, and limitations.

## Price fields and timing

| Measure | Primary price | Weight/value | Timing |
|---|---|---|---|
| Realized history | `Extension (Qty Received) / Qty Received` | Received extension | Receipt date, else PO-date proxy |
| Committed signal | `PO Value / Qty Ordered` | Remaining open quantity × committed unit price | PO date |
| Forecast | Modeled bucket multiplier | Fixed planned or latest-complete-FY basket | 1/2/3 years from base date |

`Cost` is only a documented fallback because the source may contain per-100 or per-1,000 price bases. Reconciliation ratios and likely power-of-ten factors are reported. Client-approved `UoMAdjustmentFactor` values update price and quantity inversely while preserving spend.

## Output package

Every successful run writes `outputs/v2_<UTC timestamp>/` with:

- `inflation_report.xlsx`: executive summary, historical FY/TTM/monthly series, forecasts, committed costs, coverage, backtests, data quality, exclusions, mapping candidates, benchmarks, methodology, and run information.
- Historical, forecast, backtest, coverage, data-quality, exclusion, pair-audit, committed-cost, and mapping-candidate CSVs.
- `summary.json`, `resolved_config.json`, and `run_manifest.json`.

The run fails without an official workbook if the main estimator does not converge, fewer than 50 valid pairs exist, a source file changes mid-run, or required report sheets cannot be reopened after writing.

## Configuration

`config/model_config.xlsx` contains Controls, Scope Mapping, Category Mapping, Part Overrides, and Planned Basket sheets. Only exact normalized part keys and explicit `ReplacementPartKey` mappings affect official matches. `propose-mappings` creates candidates for review but never activates them automatically.

## Tests

```bash
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
python -m pytest -q
```

Tests cover scope, price separation, benchmark chronology, interval exposure, UOM overrides, cutoff leakage, fixed-basket arithmetic, forecasting, report reopening, and retained legacy utilities.

## Known data limitations

- Receipt date is absent, so PO date is a proxy for realized timing.
- Vendor ID is absent, so vendor-level inflation, supplier switching, and concentration cannot yet be measured despite the client's thousands of vendors.
- Currency, UOM/price basis, PO number/line, facility, order status, and contract flags are absent.
- Duplicate-looking rows are flagged and preserved because no PO-line key proves duplication.
- Sparse or low-coverage buckets are disclosed and shrink toward overall.
- A planned BOM/MRP basket is preferable to the latest-complete-FY spend proxy.
- Forecasts beyond 24 months are scenarios, not precise point predictions.

These limitations are surfaced in every report rather than silently filled with assumptions.

## Add another annual workbook

1. Copy it into `data/raw/`.
2. Run `python -m parts_inflation.cli validate`.
3. Review new category, part, and UOM mappings in `model_config.xlsx`.
4. Run `python -m parts_inflation.cli run`.

Raw workbooks and macros are never modified or executed.
