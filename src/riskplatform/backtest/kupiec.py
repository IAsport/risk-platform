"""Kupiec Proportion Of Failures (POF) test. See SPEC.md §6.1."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2

from riskplatform.backtest._common import log_term, validate_alpha, validate_exceptions


def kupiec_pof(exceptions: pd.Series, alpha: float = 0.99) -> dict:
    """Kupiec Proportion Of Failures test.

    H0: observed exception rate equals p = 1 - alpha. LR_POF ~ chi2(1).
    """
    validate_alpha(alpha)
    clean = validate_exceptions(exceptions)

    t_obs = int(len(clean))
    x = int(clean.sum())
    p = 1.0 - alpha
    pi_hat = x / t_obs

    ll_h0 = log_term(t_obs - x, 1.0 - p) + log_term(x, p)
    ll_unrestricted = log_term(t_obs - x, 1.0 - pi_hat) + log_term(x, pi_hat)
    lr_stat = max(0.0, float(-2.0 * (ll_h0 - ll_unrestricted)))
    p_value = float(chi2.sf(lr_stat, df=1))

    return {
        "n_obs": t_obs,
        "n_exceptions": x,
        "expected": p * t_obs,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject": bool(lr_stat > 3.841),
    }
