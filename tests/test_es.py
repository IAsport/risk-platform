from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import t as student_t

from riskplatform.distributions import student_quantile_std
from riskplatform.es import (
    es_conditional,
    es_monte_carlo,
    es_parametric,
    expected_shortfall,
)
from riskplatform.var import (
    var_conditional,
    var_historical,
    var_monte_carlo,
    var_monte_carlo_student,
    var_parametric,
)


def _pnl(n: int = 1000, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_t(5, n) * 0.01, index=pd.date_range("2020-01-01", periods=n, freq="B")
    )


def _returns_frame(n: int = 750, seed: int = 9) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = rng.normal(0.0, 0.01, n)
    frame = pd.DataFrame(
        {
            "AAA": base + rng.normal(0.0, 0.006, n),
            "BBB": 0.5 * base + rng.normal(0.0, 0.008, n),
        },
        index=dates,
    )
    return frame, pd.Series(0.5, index=["AAA", "BBB"])


# ---------- Formules fermées vs intégration numérique (SPEC B2.7) ----------


def test_es_normal_closed_form_matches_numerical_integration():
    alpha = 0.99
    pnl = _pnl()
    observed_sigma = float(pnl.std(ddof=1))

    closed = es_parametric(pnl, alpha=alpha)

    # E[L | L > VaR] avec L = -r, r ~ N(0, sigma²) : intégrale de queue.
    z = norm.ppf(alpha)
    tail, _ = quad(lambda x: x * norm.pdf(x), z, np.inf)
    expected = observed_sigma * tail / (1.0 - alpha)
    assert closed == pytest.approx(expected, rel=1e-8)


def test_es_student_closed_form_matches_numerical_integration():
    alpha, df = 0.99, 5.0
    pnl = _pnl()
    observed_sigma = float(pnl.std(ddof=1))

    closed = es_parametric(pnl, alpha=alpha, df=df)

    # Intégrale de queue de la t STANDARDISÉE : densité f_eps(x) = f_df(x/s)/s.
    scale = np.sqrt((df - 2.0) / df)
    quantile = -student_quantile_std(1.0 - alpha, df)
    tail, _ = quad(
        lambda x: x * student_t.pdf(x / scale, df) / scale, quantile, np.inf
    )
    expected = observed_sigma * tail / (1.0 - alpha)
    assert closed == pytest.approx(expected, rel=1e-8)


def test_es_student_converges_to_normal_for_large_df():
    pnl = _pnl()

    assert es_parametric(pnl, df=1e6) == pytest.approx(es_parametric(pnl), rel=1e-4)


def test_es_increases_as_df_decreases():
    pnl = _pnl()

    assert es_parametric(pnl, df=4.0) > es_parametric(pnl, df=8.0) > es_parametric(pnl)


# ---------- Propriété ES >= VaR, systématique (SPEC B2.7) ----------


@pytest.mark.parametrize("alpha", [0.95, 0.975, 0.99])
def test_es_historical_dominates_var(alpha):
    pnl = _pnl()

    assert expected_shortfall(pnl, alpha=alpha) >= var_historical(pnl, alpha=alpha)


@pytest.mark.parametrize("alpha", [0.95, 0.975, 0.99])
@pytest.mark.parametrize("df", [4.0, 8.0, None])
def test_es_parametric_dominates_var(alpha, df):
    pnl = _pnl()

    es_value = es_parametric(pnl, alpha=alpha, df=df)
    var_value = (
        var_parametric(pnl, alpha=alpha)
        if df is None
        else -student_quantile_std(1 - alpha, df) * float(pnl.std(ddof=1))
    )
    assert es_value >= var_value


@pytest.mark.parametrize("dist,df", [("normal", None), ("student", 5.0)])
def test_es_monte_carlo_dominates_var(dist, df):
    returns, weights = _returns_frame()

    es_value = es_monte_carlo(returns, weights, alpha=0.99, dist=dist, df=df)
    var_value = (
        var_monte_carlo(returns, weights, alpha=0.99)
        if dist == "normal"
        else var_monte_carlo_student(returns, weights, alpha=0.99, df=df)
    )
    assert es_value >= var_value


def test_es_conditional_dominates_var_conditional():
    sigma = pd.Series([0.01, 0.03], index=pd.date_range("2024-01-01", periods=2))

    es_series = es_conditional(sigma, alpha=0.99, df=5.0)
    var_series = var_conditional(sigma, alpha=0.99, df=5.0)

    assert isinstance(es_series, pd.Series)
    assert (es_series >= var_series).all()


# ---------- Calibrage FRTB (SPEC B2.5) ----------


def test_es_975_normal_approximates_var_99_normal():
    pnl = _pnl()

    es_975 = es_parametric(pnl, alpha=0.975)
    var_99 = var_parametric(pnl, alpha=0.99)

    assert es_975 == pytest.approx(var_99, rel=0.01)


# ---------- MC vs fermé, erreurs ----------


def test_es_monte_carlo_matches_closed_form_on_gaussian_inputs():
    returns, weights = _returns_frame()
    portfolio_series = (returns @ weights).rename("p")

    mc = es_monte_carlo(returns, weights, alpha=0.99, n_sims=200_000)
    closed = es_parametric(portfolio_series, alpha=0.99)

    assert mc == pytest.approx(closed, rel=0.05)


def test_invalid_inputs_rejected():
    returns, weights = _returns_frame(n=100)
    pnl = _pnl(100)

    with pytest.raises(ValueError, match="df"):
        es_parametric(pnl, df=2.0)
    with pytest.raises(ValueError, match="dist"):
        es_monte_carlo(returns, weights, dist="cauchy")
    with pytest.raises(ValueError, match="sigma"):
        es_conditional(-0.01)
