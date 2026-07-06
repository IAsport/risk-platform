"""Christoffersen independence and conditional coverage tests. See SPEC.md §6.2."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2

from riskplatform.backtest._common import log_term, validate_exceptions
from riskplatform.backtest.kupiec import kupiec_pof


def christoffersen_independence(exceptions: pd.Series) -> dict:
    """Christoffersen independence test for exception clustering."""
    clean = validate_exceptions(exceptions)
    if len(clean) < 2:
        raise ValueError("exceptions must contain at least two observations")

    previous = clean.iloc[:-1].to_numpy()
    current = clean.iloc[1:].to_numpy()

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    ll_restricted = log_term(n00 + n10, 1.0 - pi) + log_term(n01 + n11, pi)
    ll_unrestricted = (
        log_term(n00, 1.0 - pi01)
        + log_term(n01, pi01)
        + log_term(n10, 1.0 - pi11)
        + log_term(n11, pi11)
    )
    lr_stat = max(0.0, float(-2.0 * (ll_restricted - ll_unrestricted)))
    p_value = float(chi2.sf(lr_stat, df=1))

    return {
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject": bool(lr_stat > 3.841),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def christoffersen_cc(exceptions: pd.Series, alpha: float = 0.99) -> dict:
    """Conditional coverage test: Kupiec POF + independence."""
    pof = kupiec_pof(exceptions, alpha=alpha)
    independence = christoffersen_independence(exceptions)

    lr_stat = float(pof["lr_stat"] + independence["lr_stat"])
    p_value = float(chi2.sf(lr_stat, df=2))

    return {
        "n_obs": pof["n_obs"],
        "n_exceptions": pof["n_exceptions"],
        "expected": pof["expected"],
        "lr_pof": pof["lr_stat"],
        "lr_ind": independence["lr_stat"],
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject": bool(lr_stat > 5.991),
        "p_value_pof": pof["p_value"],
        "p_value_ind": independence["p_value"],
        "reject_pof": pof["reject"],
        "reject_ind": independence["reject"],
        "n00": independence["n00"],
        "n01": independence["n01"],
        "n10": independence["n10"],
        "n11": independence["n11"],
    }
