"""Historical (non-parametric) VaR. Positive-loss convention (see var/__init__.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform._validation import QUANTILE_METHOD, validate_alpha, validate_series


def var_historical(
    pnl_returns: pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
) -> float:
    """Historical VaR: empirical return quantile converted to positive loss.

    VaR = -Quantile_{1-alpha}(r_p) * notional.

    Quantile convention: this uses ``np.quantile(..., method="linear")``, the
    NumPy default. It linearly interpolates between adjacent order statistics,
    which is continuous and easy to defend on small samples.

    Returns:
        VaR >= 0 (positive loss).
    """
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    return_quantile = np.quantile(clean.to_numpy(), 1.0 - alpha, method=QUANTILE_METHOD)
    return max(0.0, float(-return_quantile * notional))
