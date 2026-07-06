"""Time scaling and out-of-sample rolling VaR (backtest input)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform._validation import validate_alpha, validate_series
from riskplatform.var.historical import var_historical
from riskplatform.var.parametric import var_parametric


def scale_var(var_1d: float, horizon_days: int) -> float:
    """Square-root-of-time scaling: VaR_h = VaR_1d * sqrt(h)."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return float(var_1d * np.sqrt(horizon_days))


def rolling_var(
    pnl_returns: pd.Series,
    method: str,
    alpha: float = 0.99,
    window: int = 250,
    notional: float = 1.0,
) -> pd.Series:
    """Out-of-sample rolling VaR.

    For each t >= window, estimate VaR on [t-window, t-1].
    method must be "historical" or "parametric".
    """
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(clean):
        raise ValueError("window must be <= len(pnl_returns)")
    if method not in {"historical", "parametric"}:
        raise ValueError("method must be 'historical' or 'parametric'")

    values: list[float] = []
    index = clean.index[window:]
    for end in range(window, len(clean)):
        sample = clean.iloc[end - window : end]
        if method == "historical":
            value = var_historical(sample, alpha=alpha, notional=notional)
        else:
            value = var_parametric(sample, alpha=alpha, notional=notional)
        values.append(value)
    return pd.Series(values, index=index)
