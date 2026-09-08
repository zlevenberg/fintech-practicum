"""Regularized repeat-purchase hierarchical model with quantity adjustment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from parts_inflation.config import QuantityAdjustmentMode, ResolvedConfig
from parts_inflation.matched_pairs import flag_extreme_pairs, month_coverage_matrix

logger = logging.getLogger(__name__)


@dataclass
class RepeatSalesResult:
    months: list[pd.Period]
    categories: list[str]
    delta0: np.ndarray  # overall monthly log inflation
    u: np.ndarray  # (n_cat, n_months) category deviations
    gamma0: float
    kappa: np.ndarray  # category quantity elasticity deviations
    pair_weights: np.ndarray
    sigma: float
    converged: bool
    n_pairs: int
    n_pairs_used: int
    warnings: list[str] = field(default_factory=list)
    train_pairs: Optional[pd.DataFrame] = None

    def category_monthly(self, category: str) -> np.ndarray:
        if category in self.categories:
            i = self.categories.index(category)
            return self.delta0 + self.u[i]
        return self.delta0.copy()

    def overall_series(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": [str(m) for m in self.months],
                "period": self.months,
                "delta0": self.delta0,
                "pct_change": np.expm1(self.delta0),
            }
        )


def _pair_weights(pairs: pd.DataFrame, method: str) -> np.ndarray:
    n = pairs["n_pairs_part"].to_numpy(dtype=float)
    va = pairs["spend_a"].fillna(pairs["price_a"] * pairs["qty_a"].fillna(0)).to_numpy(dtype=float)
    vb = pairs["spend_b"].fillna(pairs["price_b"] * pairs["qty_b"].fillna(0)).to_numpy(dtype=float)
    geom = np.sqrt(np.clip(va, 0, None) * np.clip(vb, 0, None))
    if method == "equal_part":
        w = np.ones(len(pairs), dtype=float)
    else:
        cap = np.quantile(geom[geom > 0], 0.95) if np.any(geom > 0) else 1.0
        w = np.minimum(geom, cap) / np.sqrt(np.clip(n, 1, None))
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    w = w / w.mean()
    return w


def _build_design(
    pairs: pd.DataFrame,
    D: np.ndarray,
    categories: list[str],
    quantity_mode: QuantityAdjustmentMode,
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """
    Design columns: [delta0_m (M) | u_c,m (C*M) | gamma0 (1) | kappa_c (C)]
    y = D @ delta0 + sum_c 1_c * D @ u_c + (gamma0 + kappa_c) * x_q
    """
    n, M = D.shape
    C = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}
    cat_ids = pairs["approved_category"].map(lambda x: cat_index.get(x, -1)).to_numpy()

    # Build sparse blocks
    blocks = [sparse.csr_matrix(D)]  # delta0
    # category deviations: for each cat, rows belonging to cat get D
    u_blocks = []
    for c_i in range(C):
        mask = cat_ids == c_i
        Dc = D.copy()
        Dc[~mask] = 0.0
        u_blocks.append(sparse.csr_matrix(Dc))
    if u_blocks:
        blocks.append(sparse.hstack(u_blocks))
    else:
        blocks.append(sparse.csr_matrix((n, 0)))

    x_q = pairs["x_q"].to_numpy(dtype=float)
    if quantity_mode == QuantityAdjustmentMode.ignore:
        x_q = np.zeros_like(x_q)
    # gamma0 column
    blocks.append(sparse.csr_matrix(x_q.reshape(-1, 1)))
    # kappa_c columns
    K = np.zeros((n, C))
    for i, c_i in enumerate(cat_ids):
        if c_i >= 0:
            K[i, c_i] = x_q[i]
    blocks.append(sparse.csr_matrix(K))

    X = sparse.hstack(blocks).tocsr()
    y = pairs["y"].to_numpy(dtype=float)
    meta = {"M": M, "C": C, "n_params": X.shape[1]}
    return X, y, meta


def _penalty_matrix(M: int, C: int, lambdas: dict[str, float]) -> sparse.csr_matrix:
    """Return R such that ||R theta||^2 implements the quadratic penalties."""
    rows = []
    # Smoothness on delta0: diff
    for i in range(M - 1):
        r = np.zeros(M + C * M + 1 + C)
        r[i] = -np.sqrt(lambdas["smooth"])
        r[i + 1] = np.sqrt(lambdas["smooth"])
        rows.append(r)
    # Ridge on delta0
    for i in range(M):
        r = np.zeros(M + C * M + 1 + C)
        r[i] = np.sqrt(lambdas["ridge"])
        rows.append(r)
    # Ridge on u and smoothness on u
    base_u = M
    for c in range(C):
        offset = base_u + c * M
        for i in range(M):
            r = np.zeros(M + C * M + 1 + C)
            r[offset + i] = np.sqrt(lambdas["u"])
            rows.append(r)
        for i in range(M - 1):
            r = np.zeros(M + C * M + 1 + C)
            r[offset + i] = -np.sqrt(lambdas["us"])
            r[offset + i + 1] = np.sqrt(lambdas["us"])
            rows.append(r)
    # Ridge on kappa (not gamma0)
    base_k = M + C * M + 1
    for c in range(C):
        r = np.zeros(M + C * M + 1 + C)
        r[base_k + c] = np.sqrt(lambdas["gamma"])
        rows.append(r)
    if not rows:
        return sparse.csr_matrix((0, M + C * M + 1 + C))
    return sparse.csr_matrix(np.vstack(rows))


def fit_repeat_sales(
    pairs: pd.DataFrame,
    config: ResolvedConfig,
    downweight_extremes: bool = True,
    lambdas_override: Optional[dict[str, float]] = None,
    max_iter_override: Optional[int] = None,
) -> RepeatSalesResult:
    """Fit an identified robust interval model.

    The old implementation estimated an overall monthly effect plus a complete
    set of category-month effects in one rank-deficient design. V2 fits the
    overall direct-cost index once, then fits each category on its own rows and
    shrinks the category path toward the overall path. The public result shape
    remains compatible: ``u[c]`` is the identified category-minus-overall path.
    """
    warnings: list[str] = []
    if pairs is None or pairs.empty:
        return RepeatSalesResult(
            months=[], categories=[], delta0=np.array([]), u=np.zeros((0, 0)),
            gamma0=0.0, kappa=np.array([]), pair_weights=np.array([]), sigma=np.nan,
            converged=False, n_pairs=0, n_pairs_used=0,
            warnings=["No consecutive repeat-purchase pairs available"],
        )

    ctrls = config.controls
    work = flag_extreme_pairs(
        pairs,
        ratio_low=ctrls.extreme_ratio_low,
        ratio_high=ctrls.extreme_ratio_high,
        max_days=ctrls.extreme_ratio_max_days,
    ).reset_index(drop=True)
    D, months = month_coverage_matrix(work)
    if D.shape[1] == 0:
        return RepeatSalesResult(
            months=[], categories=[], delta0=np.array([]), u=np.zeros((0, 0)),
            gamma0=0.0, kappa=np.array([]), pair_weights=np.array([]), sigma=np.nan,
            converged=False, n_pairs=len(work), n_pairs_used=0,
            warnings=["No calendar-month exposure could be constructed"],
        )

    lambdas = {
        "smooth": float(ctrls.lambda_smooth),
        "ridge": float(ctrls.lambda_ridge),
        "gamma": float(ctrls.lambda_gamma),
    }
    if lambdas_override:
        for key in lambdas:
            if key in lambdas_override:
                lambdas[key] = float(lambdas_override[key])

    base_weights = _pair_weights(work, ctrls.pair_weight_method)
    if downweight_extremes:
        base_weights *= np.where(work["extreme_flag"].to_numpy(), 0.25, 1.0)
        base_weights /= base_weights.mean()

    quantity_enabled = ctrls.quantity_adjustment_mode == QuantityAdjustmentMode.estimate
    # Fast mode reduces resampling/backtest breadth, not the numerical standard
    # required for an official fit. The real workbooks need about 20 iterations.
    max_iter = int(max_iter_override or (30 if ctrls.fast_mode else 60))

    def fit_one(
        subset: pd.DataFrame,
        d_sub: np.ndarray,
        w_sub: np.ndarray,
    ) -> tuple[np.ndarray, float, float, bool]:
        m = d_sub.shape[1]
        blocks = [d_sub]
        if quantity_enabled:
            blocks.append(subset["x_q"].to_numpy(float).reshape(-1, 1))
        X = np.column_stack(blocks)
        y = subset["y"].to_numpy(float)

        n_params = X.shape[1]
        penalty_gram = np.zeros((n_params, n_params), dtype=float)
        if m >= 3 and lambdas["smooth"] > 0:
            d2 = np.diff(np.eye(m), n=2, axis=0)
            penalty_gram[:m, :m] += lambdas["smooth"] * (d2.T @ d2)
        if lambdas["ridge"] > 0:
            penalty_gram[np.arange(m), np.arange(m)] += lambdas["ridge"]
        if quantity_enabled and lambdas["gamma"] > 0:
            penalty_gram[-1, -1] += lambdas["gamma"]

        theta = np.zeros(n_params, dtype=float)
        sigma = 1.0
        converged = False
        safe_base = np.clip(w_sub, 1e-12, None)
        for _ in range(max_iter):
            residual = y - X @ theta
            sigma = max(
                1e-6,
                float(np.median(np.abs(residual - np.median(residual))) / 0.6745),
            )
            standardized = residual / sigma
            huber = np.ones_like(standardized)
            outlier = np.abs(standardized) > ctrls.huber_delta
            huber[outlier] = ctrls.huber_delta / np.abs(standardized[outlier])
            obs_weight = safe_base * huber
            # There are only roughly 47 coefficients in the current data.
            # Solving the small penalized normal equations is exactly the same
            # weighted least-squares step as the former augmented sparse LSQR,
            # but is orders of magnitude faster for repeated bootstrap fits.
            lhs = X.T @ (X * obs_weight[:, None]) + penalty_gram
            rhs = X.T @ (obs_weight * y)
            try:
                theta_new = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                theta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            relative_change = np.linalg.norm(theta_new - theta) / (1.0 + np.linalg.norm(theta))
            theta = theta_new
            if relative_change < 1e-6:
                converged = True
                break
        rates = theta[:m]
        gamma = float(theta[-1]) if quantity_enabled else 0.0
        return rates, gamma, sigma, converged

    delta0, gamma0, sigma, converged = fit_one(work, D, base_weights)
    if not converged:
        warnings.append("Overall repeat-sales IRLS did not converge; official outputs must be blocked")
    if abs(gamma0) > 1.0:
        warnings.append(f"Quantity elasticity {gamma0:.3f} is economically extreme")

    category_series = work["approved_category"].fillna("Uncategorized").astype(str)
    categories = sorted(category_series.unique().tolist())
    u = np.zeros((len(categories), len(months)), dtype=float)
    kappa = np.zeros(len(categories), dtype=float)
    for ci, category in enumerate(categories):
        mask = category_series.eq(category).to_numpy()
        n_cat = int(mask.sum())
        if n_cat < max(8, min(25, ctrls.category_min_pairs)):
            warnings.append(f"{category}: only {n_cat} pairs; using overall fallback")
            continue
        rates_c, gamma_c, _, converged_c = fit_one(
            work.loc[mask].reset_index(drop=True), D[mask], base_weights[mask]
        )
        if not converged_c:
            warnings.append(f"{category}: category fit did not converge; using overall fallback")
            continue
        shrink = n_cat / (n_cat + max(float(ctrls.category_min_pairs), 1.0))
        u[ci] = shrink * (rates_c - delta0)
        kappa[ci] = shrink * (gamma_c - gamma0)

    result = RepeatSalesResult(
        months=months,
        categories=categories,
        delta0=delta0,
        u=u,
        gamma0=gamma0,
        kappa=kappa,
        pair_weights=base_weights,
        sigma=float(sigma),
        converged=converged,
        n_pairs=len(work),
        n_pairs_used=len(work),
        warnings=warnings,
        train_pairs=work,
    )
    logger.info(
        "Fitted identified repeat-sales model: months=%s categories=%s pairs=%s gamma=%.4f converged=%s",
        len(months), len(categories), len(work), gamma0, converged,
    )
    return result


def lambda_candidate_grid(fast_mode: bool = False) -> list[dict[str, float]]:
    """Documented small grid centered on ControlDefaults; fast_mode uses 3 points."""
    if fast_mode:
        return [
            {"smooth": 10.0, "u": 5.0, "gamma": 2.0},
            {"smooth": 5.0, "u": 2.0, "gamma": 1.0},
            {"smooth": 20.0, "u": 10.0, "gamma": 4.0},
        ]
    return [
        {"smooth": 10.0, "u": 5.0, "gamma": 2.0},
        {"smooth": 5.0, "u": 5.0, "gamma": 2.0},
        {"smooth": 20.0, "u": 5.0, "gamma": 2.0},
        {"smooth": 10.0, "u": 2.0, "gamma": 2.0},
        {"smooth": 10.0, "u": 10.0, "gamma": 2.0},
        {"smooth": 10.0, "u": 5.0, "gamma": 1.0},
        {"smooth": 10.0, "u": 5.0, "gamma": 4.0},
        {"smooth": 5.0, "u": 2.0, "gamma": 1.0},
        {"smooth": 20.0, "u": 10.0, "gamma": 4.0},
    ]


def select_lambdas_by_inner_backtest(
    pairs: pd.DataFrame,
    config: ResolvedConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Choose lambda_smooth / lambda_u / lambda_gamma via inner rolling-origin WAPE
    on training history only (never the final target window).
    """
    from parts_inflation.forecast import build_forecast_candidates, select_forecast_by_backtest
    from parts_inflation.hierarchy import compute_part_residuals, multiplier_category

    ctrls = config.controls
    grid = lambda_candidate_grid(ctrls.fast_mode)
    default = {"smooth": ctrls.lambda_smooth, "u": ctrls.lambda_u, "gamma": ctrls.lambda_gamma}
    if pairs is None or pairs.empty or len(pairs) < 50:
        return default, pd.DataFrame([{"status": "insufficient_pairs", **default}])

    work = pairs.copy()
    work["date_b"] = pd.to_datetime(work["date_b"])
    min_d = work["date_b"].min()
    max_d = work["date_b"].max()
    start = min_d + pd.DateOffset(months=12)
    cutoffs = pd.date_range(start=start, end=max_d - pd.DateOffset(months=3), freq="QE")
    if len(cutoffs) == 0:
        cutoffs = pd.DatetimeIndex([start])
    if ctrls.fast_mode:
        cutoffs = cutoffs[-1:] if len(cutoffs) > 0 else cutoffs
    else:
        cutoffs = cutoffs[-4:] if len(cutoffs) > 4 else cutoffs

    scores: dict[str, list[float]] = {str(i): [] for i in range(len(grid))}
    for cutoff in cutoffs:
        if progress:
            progress(f"Lambda grid cutoff {cutoff.date()}")
        train = work.loc[work["date_b"] <= cutoff].reset_index(drop=True)
        future = work.loc[
            (work["date_b"] > cutoff) & (work["date_b"] <= cutoff + pd.DateOffset(months=6))
        ].reset_index(drop=True)
        if len(train) < 30 or future.empty:
            continue
        # Hyperparameter search may use a spend-weighted train sample; final fit still uses all pairs.
        train_fit = train
        max_train = 15000 if ctrls.fast_mode else 40000
        if len(train_fit) > max_train:
            ww = train_fit["spend_b"].fillna(train_fit["price_b"] * train_fit["qty_b"].fillna(1)).clip(lower=0)
            if ww.sum() > 0:
                rng = np.random.default_rng(ctrls.random_seed)
                p = (ww / ww.sum()).to_numpy()
                idx = rng.choice(train_fit.index.to_numpy(), size=max_train, replace=False, p=p)
                train_fit = train_fit.loc[idx].reset_index(drop=True)
        # Evaluate on a spend-weighted sample of future pairs for speed
        fut = future
        if len(fut) > 400:
            ww = fut["spend_b"].fillna(fut["price_b"] * fut["qty_b"].fillna(1)).clip(lower=0)
            if ww.sum() > 0:
                rng = np.random.default_rng(ctrls.random_seed)
                p = (ww / ww.sum()).to_numpy()
                idx = rng.choice(fut.index.to_numpy(), size=400, replace=False, p=p)
                fut = fut.loc[idx]
        for gi, cand_l in enumerate(grid):
            model = fit_repeat_sales(train_fit, config, lambdas_override=cand_l)
            if model.n_pairs_used == 0:
                continue
            part_hier = compute_part_residuals(train_fit, model, config)
            ph = part_hier.set_index("PartKey") if not part_hier.empty else None
            best_fc, _ = select_forecast_by_backtest(model.delta0, model.months, [3, 6])
            cands = build_forecast_candidates(
                model.delta0, model.months, cutoff + pd.DateOffset(months=6), cutoff
            )
            cand = next((c for c in cands if c.name == best_fc), cands[0])
            preds = []
            acts = []
            qtys = []
            cat_cache: dict = {}
            for _, r in fut.iterrows():
                part = r["PartKey"]
                bd = pd.Timestamp(r["date_a"])
                ad = pd.Timestamp(r["date_b"])
                base_p = float(r["price_a"])
                act_p = float(r["price_b"])
                if part in (ph.index if ph is not None else []):
                    prow = ph.loc[part]
                    cat = str(prow.get("category", "overall"))
                    resid = float(prow.get("shrunk_residual", 0.0) or 0.0)
                else:
                    cat = str(r.get("approved_category", "overall"))
                    resid = 0.0
                key = (cat, bd.normalize(), ad.normalize())
                if key not in cat_cache:
                    cat_cache[key] = multiplier_category(
                        model, cat, bd, ad, cand.monthly_rates, cand.months
                    )
                years = max((ad - bd).days / 365.25, 0.0)
                pred = base_p * cat_cache[key] * float(np.exp(resid * years))
                preds.append(pred)
                acts.append(act_p)
                qtys.append(float(r.get("qty_b") or 1.0))
            if not acts:
                continue
            yt = np.asarray(acts, dtype=float)
            yp = np.asarray(preds, dtype=float)
            q = np.asarray(qtys, dtype=float)
            denom = np.sum(q * yt)
            if denom <= 0:
                continue
            scores[str(gi)].append(float(np.sum(q * np.abs(yp - yt)) / denom))

    rows = []
    best_i = 0
    best_score = float("inf")
    for gi, cand_l in enumerate(grid):
        vals = scores[str(gi)]
        mean_wape = float(np.mean(vals)) if vals else float("inf")
        rows.append({**cand_l, "mean_wape": mean_wape, "n_cutoffs": len(vals)})
        if mean_wape < best_score:
            best_score = mean_wape
            best_i = gi
    selected = dict(grid[best_i])
    # Always include ridge/us from controls
    selected.setdefault("ridge", ctrls.lambda_ridge)
    selected.setdefault("us", ctrls.lambda_us)
    table = pd.DataFrame(rows)
    logger.info("Selected lambdas via inner backtest: %s (WAPE=%.4f)", selected, best_score)
    return selected, table


def predict_pair_log_change(result: RepeatSalesResult, pairs: pd.DataFrame) -> np.ndarray:
    if result.n_pairs_used == 0 or not result.months:
        return np.zeros(len(pairs))
    D, _ = month_coverage_matrix(pairs, result.months)
    cat_index = {c: i for i, c in enumerate(result.categories)}
    yhat = D @ result.delta0
    for i, cat in enumerate(pairs["approved_category"].astype(str)):
        if cat in cat_index:
            yhat[i] += D[i] @ result.u[cat_index[cat]]
            yhat[i] += (result.gamma0 + result.kappa[cat_index[cat]]) * float(pairs.iloc[i]["x_q"])
        else:
            yhat[i] += result.gamma0 * float(pairs.iloc[i]["x_q"])
    return yhat
