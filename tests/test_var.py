from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.es import expected_shortfall
from riskplatform.portfolio import portfolio_returns
from riskplatform.var import (
    rolling_var,
    scale_var,
    var_historical,
    var_monte_carlo,
    var_parametric,
    var_parametric_portfolio,
)


def _zero_mean_multiticker_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "AAA": [-0.02, -0.01, 0.00, 0.01, 0.02],
            "BBB": [0.01, -0.02, 0.00, 0.02, -0.01],
        },
        index=dates,
    )


def test_var_is_positive_loss():
    returns = pd.Series([-0.05, -0.02, 0.01, 0.03])

    assert var_historical(returns, alpha=0.95) >= 0.0
    assert var_parametric(returns, alpha=0.95) >= 0.0


def test_var_increases_with_confidence_level():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    assert var_historical(returns, alpha=0.99) >= var_historical(returns, alpha=0.95)


def test_var_parametric_matches_normal_quantile():
    sigma = 0.02
    notional = 1_000_000.0
    returns = pd.Series([-sigma, 0.0, sigma])

    result = var_parametric(returns, alpha=0.99, notional=notional)

    assert result == pytest.approx(2.3263478740408408 * sigma * notional, rel=1e-3)


def test_var_historical_matches_empirical_quantile():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    result = var_historical(returns, alpha=0.80)

    assert result == pytest.approx(0.06)


def test_monte_carlo_converges_to_parametric_under_normality():
    returns = _zero_mean_multiticker_returns()
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})

    parametric = var_parametric_portfolio(returns, weights, alpha=0.99)
    monte_carlo = var_monte_carlo(
        returns,
        weights,
        alpha=0.99,
        n_sims=200_000,
        seed=123,
    )

    assert monte_carlo == pytest.approx(parametric, rel=0.04)


def test_monte_carlo_cholesky_orientation_converges_on_asymmetric_three_assets():
    base_returns = pd.DataFrame(
        {
            "AAA": [-0.04, -0.01, 0.02, 0.03, 0.00, 0.05, -0.03],
            "BBB": [0.03, -0.02, -0.01, 0.04, -0.04, 0.01, 0.02],
            "CCC": [-0.01, 0.05, -0.03, 0.00, 0.02, -0.02, 0.04],
        }
    )
    returns = base_returns - base_returns.mean()
    weights = pd.Series({"AAA": 0.20, "BBB": 0.50, "CCC": 0.30})

    parametric = var_parametric_portfolio(returns, weights, alpha=0.99)
    monte_carlo = var_monte_carlo(
        returns,
        weights,
        alpha=0.99,
        n_sims=500_000,
        seed=321,
    )

    assert monte_carlo == pytest.approx(parametric, rel=0.03)


def test_monte_carlo_includes_nonzero_drift():
    returns = _zero_mean_multiticker_returns() + pd.Series({"AAA": 0.004, "BBB": 0.002})
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})
    mu_p = float(returns.mean().loc[weights.index] @ weights)
    parametric_mean_zero = var_parametric_portfolio(returns, weights, alpha=0.99)

    monte_carlo = var_monte_carlo(
        returns,
        weights,
        alpha=0.99,
        n_sims=500_000,
        seed=456,
    )

    assert monte_carlo == pytest.approx(
        parametric_mean_zero - mu_p,
        abs=0.02 * parametric_mean_zero,
    )


def test_var_parametric_portfolio_matches_parametric_on_aggregated_returns():
    returns = _zero_mean_multiticker_returns()
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})
    aggregated = portfolio_returns(returns, weights)

    portfolio_var = var_parametric_portfolio(returns, weights, alpha=0.99)
    aggregated_var = var_parametric(aggregated, alpha=0.99)

    assert portfolio_var == pytest.approx(aggregated_var, rel=1e-12)


def test_monte_carlo_is_reproducible_with_seed():
    returns = _zero_mean_multiticker_returns()
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})

    first = var_monte_carlo(returns, weights, n_sims=10_000, seed=42)
    second = var_monte_carlo(returns, weights, n_sims=10_000, seed=42)

    assert first == second


def test_expected_shortfall_geq_var():
    returns = pd.Series([-0.10, -0.08, -0.06, 0.01, 0.02])

    var_value = var_historical(returns, alpha=0.80)
    es_value = expected_shortfall(returns, alpha=0.80)

    assert es_value > var_value


def test_monte_carlo_and_expected_shortfall_have_exact_positive_loss_values():
    returns = pd.DataFrame({"AAA": [-0.01, 0.01]})
    weights = pd.Series({"AAA": 1.0})

    monte_carlo = var_monte_carlo(
        returns,
        weights,
        alpha=0.80,
        n_sims=5,
        seed=7,
    )
    es_value = expected_shortfall(pd.Series([-0.10, -0.03, 0.02]), alpha=0.50)

    assert monte_carlo == pytest.approx(0.007662986840256052)
    assert es_value == pytest.approx(0.10)


def test_expected_shortfall_uses_strict_tail_when_losses_tie_at_threshold():
    # Current convention: ES averages losses strictly greater than VaR_alpha.
    # For losses [0, 3, 3, 3, 10] and alpha=0.50, the threshold is 3, so ES=10.
    returns = pd.Series([0.0, -3.0, -3.0, -3.0, -10.0])

    assert var_historical(returns, alpha=0.50) == pytest.approx(3.0)
    assert expected_shortfall(returns, alpha=0.50) == pytest.approx(10.0)


def test_scale_var_sqrt_time_rule():
    assert scale_var(0.02, 10) == pytest.approx(0.02 * np.sqrt(10))


def test_rolling_var_is_out_of_sample():
    dates = pd.date_range("2024-01-01", periods=5)
    returns = pd.Series([-0.03, -0.02, -0.01, 0.01, 0.02], index=dates)

    result = rolling_var(returns, method="historical", alpha=0.80, window=3)

    assert len(result) == len(returns) - 3
    assert list(result.index) == [dates[3], dates[4]]
    assert result.loc[dates[3]] == pytest.approx(
        var_historical(returns.iloc[0:3], alpha=0.80)
    )


def test_invalid_alpha_raises_value_error():
    returns = pd.Series([-0.01, 0.01])

    with pytest.raises(ValueError, match="alpha"):
        var_historical(returns, alpha=1.0)


def test_rolling_var_raises_when_window_too_large():
    returns = pd.Series([-0.01, 0.01])

    with pytest.raises(ValueError, match="window"):
        rolling_var(returns, method="historical", window=3)


def test_monte_carlo_raises_when_n_sims_is_not_positive():
    returns = _zero_mean_multiticker_returns()
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})

    with pytest.raises(ValueError, match="n_sims"):
        var_monte_carlo(returns, weights, n_sims=0)


def test_var_rejects_empty_or_nan_series():
    with pytest.raises(ValueError, match="must not be empty"):
        var_historical(pd.Series(dtype=float))

    with pytest.raises(ValueError, match="missing values"):
        var_parametric(pd.Series([-0.01, np.nan, 0.01]))
