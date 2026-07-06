"""Expected Shortfall (CVaR). Positive-loss convention (see var/__init__.py).

ES_alpha = E[L | L > VaR_alpha] — perte moyenne au-delà du seuil de VaR.
Quatre estimateurs (SPEC.md §5 et B2.4) :
- historique : moyenne des pertes pires que le quantile empirique (B0) ;
- paramétrique fermé, normale :        ES = sigma · phi(z_alpha) / (1-alpha) ;
- paramétrique fermé, Student-t std. : ES = sigma · sqrt((nu-2)/nu) ·
      [f_nu(t⁻¹_nu(alpha)) / (1-alpha)] · [(nu + t⁻¹_nu(alpha)²) / (nu-1)] ;
- Monte Carlo : moyenne des pertes simulées au-delà du quantile (mêmes
  moteurs de simulation que la VaR MC, normal ou t multivariée).

Les fermés sont validés contre intégration numérique scipy dans les tests.
Source : McNeil-Frey-Embrechts, Quantitative Risk Management (éq. 2.23-2.24).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from riskplatform._validation import (
    QUANTILE_METHOD,
    validate_alpha,
    validate_returns_frame,
    validate_series,
)
from riskplatform.distributions import fit_student_df
from riskplatform.portfolio import covariance_matrix
from riskplatform.var.monte_carlo import _simulate_normal_returns, _simulate_student_returns


def _es_factor_std(alpha: float, df: float | None) -> float:
    """ES de la loi STANDARDISÉE (variance 1) au niveau alpha, mu=0."""
    if df is None:
        z = norm.ppf(alpha)
        return float(norm.pdf(z) / (1.0 - alpha))
    if df <= 2.0:
        raise ValueError("df must be > 2 (finite variance required)")
    quantile_raw = student_t.ppf(alpha, df)
    es_raw = (
        student_t.pdf(quantile_raw, df) / (1.0 - alpha) * (df + quantile_raw**2) / (df - 1.0)
    )
    return float(es_raw * np.sqrt((df - 2.0) / df))


def _tail_mean(losses: np.ndarray, alpha: float) -> float:
    """Moyenne des pertes strictement au-delà du quantile alpha (fallback >=)."""
    var_threshold = np.quantile(losses, alpha, method=QUANTILE_METHOD)
    tail_losses = losses[losses > var_threshold]
    if tail_losses.size == 0:
        tail_losses = losses[losses >= var_threshold]
    return max(0.0, float(tail_losses.mean()))


def expected_shortfall(
    pnl_returns: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
) -> float:
    """Historical Expected Shortfall: average losses beyond VaR_alpha."""
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    losses = -clean.to_numpy() * notional
    return _tail_mean(losses, alpha)


def es_parametric(
    pnl_returns: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    df: float | None = None,
) -> float:
    """ES paramétrique fermé : normale si df=None, Student-t standardisée sinon.

    sigma = écart-type d'échantillon (ddof=1), mu = 0 (convention horizon court).
    """
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    sigma = float(clean.std(ddof=1))
    if np.isnan(sigma):
        raise ValueError("pnl_returns must contain at least two observations")
    return _es_factor_std(alpha, df) * sigma * notional


def es_conditional(
    sigma: float | pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    df: float | None = None,
) -> float | pd.Series:
    """ES conditionnel : ES_t = ES_alpha(loi standardisée) · sigma_t · notional.

    Scalaire ou série (même index) — la forme série alimente le backtest d'ES.
    """
    validate_alpha(alpha)
    factor = _es_factor_std(alpha, df)
    if isinstance(sigma, pd.Series):
        clean = validate_series(sigma, "sigma")
        if (clean < 0).any():
            raise ValueError("sigma must be >= 0")
        return (factor * clean * notional).rename("es_conditional")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return float(factor * sigma * notional)


def es_monte_carlo(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    dist: str = "normal",
    df: float | None = None,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> float:
    """ES Monte Carlo : moyenne des pertes simulées au-delà du quantile alpha.

    Mêmes moteurs de simulation que la VaR MC (normal, ou t multivariée à
    mélange partagé). dist="student" avec df=None : nu estimé par MLE sur la
    série de portefeuille standardisée (SPEC.md B2.9 #3).
    """
    validate_alpha(alpha)
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if dist not in {"normal", "student"}:
        raise ValueError("dist must be 'normal' or 'student'")
    validate_returns_frame(returns)

    cov = covariance_matrix(returns, weights)
    aligned_returns = returns.loc[:, cov.index].astype(float)
    aligned_weights = weights.loc[cov.index].astype(float)
    mu_vec = aligned_returns.mean().to_numpy()
    rng = np.random.default_rng(seed)

    if dist == "normal":
        simulated = _simulate_normal_returns(mu_vec, cov.to_numpy(), n_sims, rng)
    else:
        cov_array = cov.to_numpy()
        sigma_vec = np.sqrt(np.diag(cov_array))
        if np.any(sigma_vec < 1e-12):
            raise ValueError("returns contain a constant column (zero variance)")
        corr_array = cov_array / np.outer(sigma_vec, sigma_vec)
        if df is None:
            portfolio_series = aligned_returns @ aligned_weights
            df = fit_student_df(portfolio_series / portfolio_series.std(ddof=1))
        simulated = _simulate_student_returns(
            mu_vec, sigma_vec, corr_array, float(df), n_sims, rng
        )

    losses = -(simulated @ aligned_weights.to_numpy()) * notional
    return _tail_mean(losses, alpha)
