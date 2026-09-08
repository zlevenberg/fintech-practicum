"""Regularized repeat-purchase hierarchical model with quantity adjustment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

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
) -> RepeatSalesResult:
    warnings: list[str] = []
    if pairs is None or pairs.empty:
        warnings.append("No adjacent pairs available; hierarchical model unavailable")
        return RepeatSalesResult(
            months=[],
            categories=[],
            delta0=np.array([]),
            u=np.zeros((0, 0)),
            gamma0=0.0,
            kappa=np.array([]),
            pair_weights=np.array([]),
            sigma=np.nan,
            converged=False,
            n_pairs=0,
            n_pairs_used=0,
            warnings=warnings,
        )

    ctrls = config.controls
    pairs = flag_extreme_pairs(
        pairs,
        ratio_low=ctrls.extreme_ratio_low,
        ratio_high=ctrls.extreme_ratio_high,
        max_days=ctrls.extreme_ratio_max_days,
    )
    # Development speed: fit on a spend-weighted subsample without changing definitions
    if ctrls.fast_mode and len(pairs) > 25000:
        wtmp = _pair_weights(pairs, ctrls.pair_weight_method)
        rng = np.random.default_rng(ctrls.random_seed)
        p = wtmp / wtmp.sum()
        idx = rng.choice(len(pairs), size=25000, replace=False, p=p)
        pairs = pairs.iloc[np.sort(idx)].reset_index(drop=True)
        warnings.append("fast_mode subsampled pairs to 25,000 for hierarchical fit")

    D, months = month_coverage_matrix(pairs)
    if len(months) == 0:
        warnings.append("No month coverage; hierarchical model unavailable")
        return RepeatSalesResult(
            months=[],
            categories=[],
            delta0=np.array([]),
            u=np.zeros((0, 0)),
            gamma0=0.0,
            kappa=np.array([]),
            pair_weights=np.array([]),
            sigma=np.nan,
            converged=False,
            n_pairs=len(pairs),
            n_pairs_used=0,
            warnings=warnings,
        )

    # Category list with minimum pairs (sparse cats still present but shrink hard)
    cat_counts = pairs["approved_category"].value_counts()
    categories = sorted(cat_counts.index.astype(str).tolist())
    w = _pair_weights(pairs, ctrls.pair_weight_method)
    if downweight_extremes:
        w = w * np.where(pairs["extreme_flag"].to_numpy(), 0.25, 1.0)
        w = w / w.mean()

    X, y, meta = _build_design(pairs, D, categories, ctrls.quantity_adjustment_mode)
    M, C = meta["M"], meta["C"]
    lambdas = {
        "smooth": ctrls.lambda_smooth,
        "ridge": ctrls.lambda_ridge,
        "u": ctrls.lambda_u,
        "us": ctrls.lambda_us,
        "gamma": ctrls.lambda_gamma,
    }
    # Shrink sparse categories more: inflate lambda_u for cats below min pairs
    # (implemented by pre-scaling u columns — approximate via higher global lambda if many sparse)
    R = _penalty_matrix(M, C, lambdas)

    # IRLS with Huber weights
    n_params = X.shape[1]
    theta = np.zeros(n_params)
    sigma = 1.0
    converged = False
    delta = ctrls.huber_delta
    max_iter = 30 if not ctrls.fast_mode else 12

    sqrt_w = np.sqrt(w)
    for it in range(max_iter):
        pred = X @ theta
        resid = y - pred
        sigma = max(1e-6, float(np.median(np.abs(resid - np.median(resid))) / 0.6745))
        r_std = resid / sigma
        # Huber weight
        huber_w = np.ones_like(r_std)
        mask = np.abs(r_std) > delta
        huber_w[mask] = delta / np.abs(r_std[mask])
        ww = sqrt_w * np.sqrt(huber_w)

        Xw = sparse.diags(ww) @ X
        yw = ww * y
        # Augment with penalty
        X_aug = sparse.vstack([Xw, R]).tocsr()
        y_aug = np.concatenate([yw, np.zeros(R.shape[0])])
        sol = lsqr(X_aug, y_aug, atol=1e-6, btol=1e-6, iter_lim=2000)
        theta_new = sol[0]
        if np.linalg.norm(theta_new - theta) < 1e-6 * (1 + np.linalg.norm(theta)):
            theta = theta_new
            converged = True
            break
        theta = theta_new

    delta0 = theta[:M]
    u = theta[M : M + C * M].reshape(C, M) if C else np.zeros((0, M))
    gamma0 = float(theta[M + C * M]) if n_params > M + C * M else 0.0
    kappa = theta[M + C * M + 1 :] if C else np.array([])

    if abs(gamma0) > 1.0:
        warnings.append(
            f"Quantity elasticity gamma0={gamma0:.3f} is large in magnitude; treat with caution"
        )
    if not converged:
        warnings.append("Repeat-sales IRLS did not fully converge; using last iterate")

    # Zero out category deviations for very sparse categories (shrink to overall)
    for i, cat in enumerate(categories):
        if cat_counts.get(cat, 0) < ctrls.category_min_pairs:
            u[i] = u[i] * (cat_counts.get(cat, 0) / max(ctrls.category_min_pairs, 1))

    if ctrls.quantity_adjustment_mode == QuantityAdjustmentMode.zero:
        gamma0 = 0.0
        kappa = np.zeros_like(kappa)

    result = RepeatSalesResult(
        months=months,
        categories=categories,
        delta0=delta0,
        u=u,
        gamma0=gamma0,
        kappa=kappa,
        pair_weights=w,
        sigma=float(sigma),
        converged=converged,
        n_pairs=len(pairs),
        n_pairs_used=len(pairs),
        warnings=warnings,
        train_pairs=pairs,
    )
    logger.info(
        "Fitted repeat-sales model: months=%s cats=%s pairs=%s gamma0=%.4f sigma=%.4f converged=%s",
        M,
        C,
        len(pairs),
        gamma0,
        sigma,
        converged,
    )
    return result


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
