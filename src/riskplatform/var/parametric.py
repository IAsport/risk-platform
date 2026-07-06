"""Gaussian parametric (variance-covariance) VaR. Positive-loss convention."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskplatform._validation import validate_alpha, validate_returns_frame, validate_series
from riskplatform.portfolio import covariance_matrix


def var_parametric(
    pnl_returns: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    mean_zero: bool = True,
) -> float:
    """Gaussian parametric VaR: -(mu + z_{1-alpha} * sigma) * notional.

    mean_zero=True forces mu=0, the usual short-horizon assumption.
    """
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    sigma = float(clean.std(ddof=1))
    if np.isnan(sigma):
        raise ValueError("pnl_returns must contain at least two observations")

    mu = 0.0 if mean_zero else float(clean.mean())
    z = norm.ppf(1.0 - alpha)
    return max(0.0, float(-(mu + z * sigma) * notional))


def var_parametric_portfolio(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
) -> float:
    """Multivariate variance-covariance VaR: sigma_p = sqrt(w' Sigma w)."""
    validate_alpha(alpha)
    validate_returns_frame(returns)

    cov = covariance_matrix(returns, weights)
    aligned_weights = weights.loc[cov.index].astype(float)
    variance = float(aligned_weights.to_numpy() @ cov.to_numpy() @ aligned_weights.to_numpy())
    sigma_p = float(np.sqrt(max(variance, 0.0)))
    z = norm.ppf(1.0 - alpha)
    return max(0.0, float(-z * sigma_p * notional))
