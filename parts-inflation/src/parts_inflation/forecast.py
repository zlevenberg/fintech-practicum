"""Candidate future monthly inflation forecasts and selection helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastCandidate:
    name: str
    monthly_rates: np.ndarray  # future month log rates
    months: list[pd.Period]
    params: dict


def trailing_mean(history: np.ndarray, horizon: int, window: int = 12) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    mu = float(np.mean(history[-window:])) if len(history) else 0.0
    return np.full(horizon, mu)


def ewma_mean(history: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    level = float(history[0])
    for x in history[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return np.full(horizon, level)


def damped_holt(history: np.ndarray, horizon: int, alpha: float = 0.4, beta: float = 0.2, phi: float = 0.8) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    level = float(history[0])
    trend = float(history[1] - history[0]) if len(history) > 1 else 0.0
    for x in history[1:]:
        prev_level = level
        level = alpha * float(x) + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
    out = []
    for h in range(1, horizon + 1):
        damp = sum(phi ** j for j in range(1, h + 1))
        out.append(level + damp * trend)
    return np.array(out, dtype=float)


def mean_reversion(
    history: np.ndarray, horizon: int, window: int = 12, phi: float = 0.85
) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    mu = float(np.mean(history))
    recent = float(np.mean(history[-window:])) if len(history) else mu
    out = []
    for h in range(1, horizon + 1):
        out.append(mu + (recent - mu) * (phi ** h))
    return np.array(out, dtype=float)


def generate_future_months(last_month: pd.Period, n: int) -> list[pd.Period]:
    months = []
    cur = last_month
    for _ in range(n):
        cur = (cur + 1)
        months.append(cur)
    return months


def build_forecast_candidates(
    delta0: np.ndarray,
    months: list[pd.Period],
    target_date: pd.Timestamp,
    base_date: pd.Timestamp,
) -> list[ForecastCandidate]:
    if len(months) == 0:
        # still produce zero-inflation candidates covering needed horizon
        start = pd.Timestamp(base_date).to_period("M")
        end = pd.Timestamp(target_date).to_period("M")
        n = max(1, (end - start).n + 2)
        fut_months = generate_future_months(start, n)
        return [
            ForecastCandidate("trailing_12m_mean", np.zeros(n), fut_months, {}),
            ForecastCandidate("ewma", np.zeros(n), fut_months, {}),
            ForecastCandidate("damped_holt", np.zeros(n), fut_months, {}),
            ForecastCandidate("mean_reversion", np.zeros(n), fut_months, {}),
        ]

    last = months[-1]
    end = pd.Timestamp(target_date).to_period("M")
    n = max(1, (end - last).n + 2)
    fut_months = generate_future_months(last, n)
    hist = np.asarray(delta0, dtype=float)
    return [
        ForecastCandidate("trailing_12m_mean", trailing_mean(hist, n, 12), fut_months, {"window": 12}),
        ForecastCandidate("ewma", ewma_mean(hist, n, 0.3), fut_months, {"alpha": 0.3}),
        ForecastCandidate("damped_holt", damped_holt(hist, n), fut_months, {"alpha": 0.4, "beta": 0.2, "phi": 0.8}),
        ForecastCandidate("mean_reversion", mean_reversion(hist, n), fut_months, {"phi": 0.85}),
    ]


def select_forecast_by_backtest(
    delta0: np.ndarray,
    months: list[pd.Period],
    horizons: list[int],
) -> tuple[str, pd.DataFrame]:
    """
    Rolling-origin selection on historical monthly rates.
    Score = mean absolute error of cumulative log change over horizons.
    """
    if len(delta0) < 18:
        return "trailing_12m_mean", pd.DataFrame(
            [{"method": "trailing_12m_mean", "mae": np.nan, "note": "insufficient history; default"}]
        )

    methods = ["trailing_12m_mean", "ewma", "damped_holt", "mean_reversion"]
    scores = {m: [] for m in methods}
    hist = np.asarray(delta0, dtype=float)
    # cutoffs: leave at least max horizon months
    max_h = max(horizons)
    for t in range(12, len(hist) - max_h):
        train = hist[: t + 1]
        for h in horizons:
            if t + h >= len(hist):
                continue
            actual = float(np.sum(hist[t + 1 : t + 1 + h]))
            preds = {
                "trailing_12m_mean": float(np.sum(trailing_mean(train, h))),
                "ewma": float(np.sum(ewma_mean(train, h))),
                "damped_holt": float(np.sum(damped_holt(train, h))),
                "mean_reversion": float(np.sum(mean_reversion(train, h))),
            }
            for m, p in preds.items():
                scores[m].append(abs(p - actual))

    rows = []
    for m in methods:
        mae = float(np.mean(scores[m])) if scores[m] else np.inf
        rows.append({"method": m, "mae": mae, "n_evals": len(scores[m])})
    table = pd.DataFrame(rows).sort_values("mae")
    best = str(table.iloc[0]["method"]) if not table.empty else "trailing_12m_mean"
    logger.info("Selected forecast method %s", best)
    return best, table
