from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskplatform.var import (
    rolling_var_conditional,
    var_conditional,
    var_conditional_monte_carlo,
)
from riskplatform.volatility import ewma_variance
from riskplatform.volatility.garch import GarchParams


def _returns(n: int, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0, 0.01, n), index=pd.date_range("2015-01-01", periods=n, freq="B")
    )


def test_var_conditional_scalar_is_z_times_sigma():
    sigma = 0.02

    value = var_conditional(sigma, alpha=0.99)

    assert value == pytest.approx(-norm.ppf(0.01) * sigma)
    assert value == pytest.approx(2.326 * sigma, rel=1e-3)


def test_var_conditional_series_is_elementwise_with_same_index():
    sigma = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))

    values = var_conditional(sigma, alpha=0.95, notional=100.0)

    assert isinstance(values, pd.Series)
    assert values.index.equals(sigma.index)
    expected = -norm.ppf(0.05) * sigma * 100.0
    np.testing.assert_allclose(values.to_numpy(), expected.to_numpy())


def test_var_conditional_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        var_conditional(-0.01)
    with pytest.raises(ValueError, match="sigma"):
        var_conditional(pd.Series([0.01, -0.02]))


def test_monte_carlo_converges_to_closed_form():
    sigma = 0.015

    mc = var_conditional_monte_carlo(sigma, alpha=0.99, n_sims=200_000, seed=42)
    closed = var_conditional(sigma, alpha=0.99)

    assert mc == pytest.approx(closed, rel=0.02)


def test_monte_carlo_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="sigma_t"):
        var_conditional_monte_carlo(-0.01)
    with pytest.raises(ValueError, match="n_sims"):
        var_conditional_monte_carlo(0.01, n_sims=0)


def test_rolling_ewma_matches_direct_composition():
    returns = _returns(120)

    rolled = rolling_var_conditional(returns, "ewma", alpha=0.99)
    expected = var_conditional(np.sqrt(ewma_variance(returns)), alpha=0.99)

    assert rolled.index.equals(returns.index[30:])
    np.testing.assert_allclose(rolled.to_numpy(), expected.to_numpy())


def test_rolling_garch_refits_every_refit_every_days(monkeypatch):
    returns = _returns(400)
    fit_calls: list[int] = []
    stub_params = GarchParams(omega=5e-6, alpha=0.08, beta=0.90, loglik=0.0, n_obs=300)

    def fake_fit(sample, min_obs=250):
        fit_calls.append(len(sample))
        return stub_params

    monkeypatch.setattr("riskplatform.var.conditional.fit_garch", fake_fit)

    rolled = rolling_var_conditional(returns, "garch", alpha=0.99, window=300, refit_every=25)

    # 100 dates prévues (300..399), refits aux positions 300, 325, 350, 375.
    assert len(fit_calls) == 4
    assert all(size == 300 for size in fit_calls)
    assert rolled.index.equals(returns.index[300:])
    assert (rolled > 0).all()


def test_rolling_garch_follows_filter_between_refits(monkeypatch):
    returns = _returns(320)
    stub_params = GarchParams(omega=5e-6, alpha=0.08, beta=0.90, loglik=0.0, n_obs=300)
    monkeypatch.setattr(
        "riskplatform.var.conditional.fit_garch", lambda sample, min_obs=250: stub_params
    )

    rolled = rolling_var_conditional(returns, "garch", alpha=0.99, window=300, refit_every=1000)

    # Reconstruit la récursion à la main entre deux refits.
    z_abs = -norm.ppf(0.01)
    values = returns.to_numpy()
    sigma2 = (rolled.iloc[0] / z_abs) ** 2
    for k in range(1, len(rolled)):
        sigma2 = stub_params.omega + stub_params.alpha * values[300 + k - 1] ** 2 + (
            stub_params.beta * sigma2
        )
        assert rolled.iloc[k] == pytest.approx(z_abs * np.sqrt(sigma2))


def test_rolling_invalid_inputs_rejected():
    returns = _returns(120)

    with pytest.raises(ValueError, match="vol_method"):
        rolling_var_conditional(returns, "arch")
    with pytest.raises(ValueError, match="window"):
        rolling_var_conditional(returns, "garch", window=120)
    with pytest.raises(ValueError, match="refit_every"):
        rolling_var_conditional(returns, "garch", window=100, refit_every=0)
    with pytest.raises(ValueError, match="dist"):
        rolling_var_conditional(returns, "ewma", dist="cauchy")
    with pytest.raises(ValueError, match="df is only valid"):
        rolling_var_conditional(returns, "ewma", dist="normal", df=5.0)


def test_var_conditional_student_quantile():
    from riskplatform.distributions import student_quantile_std

    sigma = 0.02
    value = var_conditional(sigma, alpha=0.99, df=4.0)

    assert value == pytest.approx(-student_quantile_std(0.01, 4.0) * sigma)
    assert value > var_conditional(sigma, alpha=0.99)  # queue plus épaisse que la normale


def test_monte_carlo_student_converges_to_closed_form():
    sigma = 0.015

    mc = var_conditional_monte_carlo(sigma, alpha=0.99, n_sims=400_000, seed=42, df=5.0)
    closed = var_conditional(sigma, alpha=0.99, df=5.0)

    assert mc == pytest.approx(closed, rel=0.02)


def test_rolling_ewma_student_fixed_df_matches_composition():
    returns = _returns(160)
    window, df = 60, 6.0

    rolled = rolling_var_conditional(
        returns, "ewma", alpha=0.99, window=window, refit_every=10, dist="student", df=df
    )
    sigma = np.sqrt(ewma_variance(returns))
    expected = var_conditional(sigma, alpha=0.99, df=df)

    assert rolled.index.equals(returns.index[30 + window :])
    np.testing.assert_allclose(rolled.to_numpy(), expected.iloc[window:].to_numpy())


def test_rolling_ewma_student_refits_df_on_schedule(monkeypatch):
    returns = _returns(160)
    fit_calls: list[int] = []

    def fake_fit_df(residuals, bounds=(2.05, 100.0)):
        fit_calls.append(len(residuals))
        return 8.0

    monkeypatch.setattr("riskplatform.var.conditional.fit_student_df", fake_fit_df)

    rolled = rolling_var_conditional(
        returns, "ewma", alpha=0.99, window=60, refit_every=25, dist="student"
    )

    # 70 dates prévues (positions résidus 60..129), refits à 60, 85, 110 -> 3.
    assert len(fit_calls) == 3
    assert all(size == 60 for size in fit_calls)
    assert (rolled > 0).all()


def test_rolling_garch_student_uses_fatter_quantile(monkeypatch):
    returns = _returns(400)
    stub_params = GarchParams(omega=5e-6, alpha=0.08, beta=0.90, loglik=0.0, n_obs=300)
    monkeypatch.setattr(
        "riskplatform.var.conditional.fit_garch", lambda sample, min_obs=250: stub_params
    )
    monkeypatch.setattr(
        "riskplatform.var.conditional.fit_student_df", lambda residuals, bounds=(2.05, 100.0): 4.0
    )

    normal = rolling_var_conditional(returns, "garch", alpha=0.99, window=300, refit_every=25)
    student = rolling_var_conditional(
        returns, "garch", alpha=0.99, window=300, refit_every=25, dist="student"
    )

    assert student.index.equals(normal.index)
    ratio = (student / normal).to_numpy()
    from scipy.stats import norm as normal_dist

    from riskplatform.distributions import student_quantile_std

    expected_ratio = student_quantile_std(0.01, 4.0) / normal_dist.ppf(0.01)
    np.testing.assert_allclose(ratio, expected_ratio)
