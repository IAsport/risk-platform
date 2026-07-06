from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.volatility import ewma_variance, ewma_volatility


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_recursion_matches_hand_computed_values():
    # init_window=2 : sigma2_init = var([0.01, -0.02], ddof=1) = 0.00045
    returns = _series([0.01, -0.02, 0.015, 0.005])
    lam = 0.9

    sigma2 = ewma_variance(returns, lam=lam, init_window=2)

    assert len(sigma2) == 2
    assert sigma2.iloc[0] == pytest.approx(0.00045)
    # sigma2_1 = 0.9 * 0.00045 + 0.1 * 0.015² = 0.0004275
    assert sigma2.iloc[1] == pytest.approx(0.9 * 0.00045 + 0.1 * 0.015**2)


def test_last_shock_weight_is_one_minus_lambda():
    returns = _series([0.0] * 40 + [0.05, 0.0])
    lam = 0.94

    sigma2 = ewma_variance(returns, lam=lam, init_window=30)

    # Avant le choc, variance nulle ; le lendemain du choc : (1-lam) * 0.05².
    assert sigma2.iloc[-2] == pytest.approx(0.0)
    assert sigma2.iloc[-1] == pytest.approx((1 - lam) * 0.05**2)


def test_no_look_ahead_sigma_t_does_not_depend_on_r_t():
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.01, size=60)
    modified = base.copy()
    modified[-1] = 0.25  # choc énorme le dernier jour

    sigma2_base = ewma_variance(_series(list(base)), init_window=30)
    sigma2_modified = ewma_variance(_series(list(modified)), init_window=30)

    # La prévision pour la dernière date n'utilise que r jusqu'à t-1.
    assert sigma2_modified.iloc[-1] == pytest.approx(sigma2_base.iloc[-1])
    assert sigma2_modified.index.equals(sigma2_base.index)


def test_index_starts_after_init_window():
    returns = _series(list(np.full(40, 0.01)))

    sigma2 = ewma_variance(returns, init_window=30)

    assert sigma2.index.equals(returns.index[30:])


def test_constant_series_gives_zero_variance():
    returns = _series([0.01] * 40)

    sigma2 = ewma_variance(returns, init_window=30)

    assert (sigma2 >= 0).all()
    assert sigma2.iloc[0] == pytest.approx(0.0)


def test_volatility_is_sqrt_and_annualization_scales_by_sqrt_252():
    returns = _series(list(np.random.default_rng(1).normal(0, 0.01, 50)))

    sigma2 = ewma_variance(returns, init_window=30)
    vol_daily = ewma_volatility(returns, init_window=30)
    vol_annual = ewma_volatility(returns, init_window=30, annualize=True)

    np.testing.assert_allclose(vol_daily.to_numpy(), np.sqrt(sigma2.to_numpy()))
    np.testing.assert_allclose(vol_annual.to_numpy(), vol_daily.to_numpy() * np.sqrt(252))


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.1, 1.5])
def test_lambda_out_of_range_rejected(lam):
    returns = _series([0.01] * 40)

    with pytest.raises(ValueError, match="lam"):
        ewma_variance(returns, lam=lam)


def test_too_short_series_rejected():
    returns = _series([0.01] * 31)

    with pytest.raises(ValueError, match="at least"):
        ewma_variance(returns, init_window=30)


def test_nan_rejected():
    returns = _series([0.01] * 39 + [np.nan])

    with pytest.raises(ValueError, match="missing values"):
        ewma_variance(returns)
