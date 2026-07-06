"""Parametric Monte Carlo VaR: multivariate normal and Student-t. Positive-loss convention.

Student-t multivariée (SPEC.md B2.2, McNeil-Frey-Embrechts §6.2) : Cholesky
sur la matrice de CORRÉLATION, variable de mélange w ~ chi²_nu/nu PARTAGÉE
par scénario (un tirage commun aux d actifs) — c'est ce mélange commun qui
crée la dépendance de queue (les chocs extrêmes frappent ensemble), ce que
des t indépendantes par actif ne produisent pas. Rendements ramenés à la
variance cible par le facteur sqrt((nu-2)/nu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform._validation import QUANTILE_METHOD, validate_alpha, validate_returns_frame
from riskplatform.distributions import fit_student_df
from riskplatform.portfolio import covariance_matrix


def _cholesky_with_jitter(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        jitter = np.eye(matrix.shape[0]) * 1e-12
        return np.linalg.cholesky(matrix + jitter)


def _simulate_normal_returns(
    mu_vec: np.ndarray,
    cov_array: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scénarios ~ N(mu, cov) via Cholesky de la covariance (moteur B0)."""
    chol = _cholesky_with_jitter(cov_array)
    shocks = rng.standard_normal(size=(n_sims, len(mu_vec)))
    return mu_vec + shocks @ chol.T


def _simulate_student_returns(
    mu_vec: np.ndarray,
    sigma_vec: np.ndarray,
    corr_array: np.ndarray,
    df: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scénarios t multivariée : eps = L·z / sqrt(w), w ~ chi²_df/df partagé."""
    if df <= 2.0:
        raise ValueError("df must be > 2 (finite variance required)")
    chol = _cholesky_with_jitter(corr_array)
    z = rng.standard_normal(size=(n_sims, len(mu_vec)))
    mixing = rng.chisquare(df, size=n_sims) / df
    eps = (z @ chol.T) / np.sqrt(mixing)[:, None]
    return mu_vec + sigma_vec * np.sqrt((df - 2.0) / df) * eps


def var_monte_carlo(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> float:
    """Parametric Monte Carlo VaR from multivariate normal simulations."""
    validate_alpha(alpha)
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    validate_returns_frame(returns)

    cov = covariance_matrix(returns, weights)
    aligned_returns = returns.loc[:, cov.index].astype(float)
    aligned_weights = weights.loc[cov.index].astype(float)
    mu_vec = aligned_returns.mean().to_numpy()

    rng = np.random.default_rng(seed)
    simulated_returns = _simulate_normal_returns(mu_vec, cov.to_numpy(), n_sims, rng)
    portfolio_simulated_returns = simulated_returns @ aligned_weights.to_numpy()
    simulated_losses = -portfolio_simulated_returns * notional
    value = np.quantile(simulated_losses, alpha, method=QUANTILE_METHOD)
    return max(0.0, float(value))


def var_monte_carlo_student(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    df: float | None = None,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> float:
    """Monte Carlo VaR sous t multivariée (mélange partagé, Cholesky sur R).

    df=None : nu estimé par MLE univarié sur la série de portefeuille
    standardisée r_p/std(r_p) (simplification documentée SPEC.md B2.9 #3).
    """
    validate_alpha(alpha)
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    validate_returns_frame(returns)

    cov = covariance_matrix(returns, weights)
    aligned_returns = returns.loc[:, cov.index].astype(float)
    aligned_weights = weights.loc[cov.index].astype(float)
    mu_vec = aligned_returns.mean().to_numpy()

    cov_array = cov.to_numpy()
    sigma_vec = np.sqrt(np.diag(cov_array))
    # Seuil (pas ==0) : la variance d'une colonne constante ressort en ~1e-36
    # de bruit flottant via .cov(), jamais exactement nulle.
    if np.any(sigma_vec < 1e-12):
        raise ValueError("returns contain a constant column (zero variance)")
    corr_array = cov_array / np.outer(sigma_vec, sigma_vec)

    if df is None:
        portfolio_series = aligned_returns @ aligned_weights
        df = fit_student_df(portfolio_series / portfolio_series.std(ddof=1))

    rng = np.random.default_rng(seed)
    simulated_returns = _simulate_student_returns(
        mu_vec, sigma_vec, corr_array, float(df), n_sims, rng
    )
    portfolio_simulated_returns = simulated_returns @ aligned_weights.to_numpy()
    simulated_losses = -portfolio_simulated_returns * notional
    value = np.quantile(simulated_losses, alpha, method=QUANTILE_METHOD)
    return max(0.0, float(value))
