"""Daily VaR exception indicator (input of the Kupiec/Christoffersen tests)."""

from __future__ import annotations

import pandas as pd

from riskplatform.backtest._common import validate_series


def count_exceptions(
    realized_returns: pd.Series,
    var_series: pd.Series,
    notional: float = 1.0,
) -> pd.Series:
    """Indicator of daily VaR exceptions on common dates only."""
    realized = validate_series(realized_returns, "realized_returns").astype(float)
    var = validate_series(var_series, "var_series").astype(float)

    common_index = realized.index.intersection(var.index)
    if common_index.empty:
        raise ValueError("realized_returns and var_series have an empty intersection")

    realized_aligned = realized.loc[common_index]
    var_aligned = var.loc[common_index]
    losses = -realized_aligned * notional
    exceptions = (losses > var_aligned).astype(int)
    exceptions.name = "exception"
    return exceptions
