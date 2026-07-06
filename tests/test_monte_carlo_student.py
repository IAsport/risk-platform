from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.var import var_monte_carlo, var_monte_carlo_student
from riskplatform.var.monte_carlo import (
    _simulate_normal_returns,
    _simulate_student_returns,
)


def _returns_frame(n: int = 750, seed: int = 9) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = rng.normal(0.0, 0.01, n)
    frame = pd.DataFrame(
        {
            "AAA": base + rng.normal(0.0, 0.006, n),
            "BBB": 0.5 * base + rng.normal(0.0, 0.008, n),
            "CCC": rng.normal(0.0, 0.012, n),
        },
        index=dates,
    )
    weights = pd.Series(1 / 3, index=["AAA", "BBB", "CCC"])
    return frame, weights


def test_large_df_converges_to_normal_mc():
    returns, weights = _returns_frame()

    student = var_monte_carlo_student(returns, weights, alpha=0.99, df=80.0, n_sims=100_000)
    normal = var_monte_carlo(returns, weights, alpha=0.99, n_sims=100_000)

    assert student == pytest.approx(normal, rel=0.03)


def test_small_df_gives_fatter_tail_var():
    returns, weights = _returns_frame()

    student = var_monte_carlo_student(returns, weights, alpha=0.99, df=4.0)
    normal = var_monte_carlo(returns, weights, alpha=0.99)

    assert student > normal


def test_df_none_uses_mle_and_matches_explicit_df():
    returns, weights = _returns_frame()
    from riskplatform.distributions import fit_student_df

    portfolio_series = returns @ weights
    fitted = fit_student_df(portfolio_series / portfolio_series.std(ddof=1))

    implicit = var_monte_carlo_student(returns, weights, alpha=0.99, df=None)
    explicit = var_monte_carlo_student(returns, weights, alpha=0.99, df=fitted)

    assert implicit == pytest.approx(explicit)


def test_seed_reproducibility():
    returns, weights = _returns_frame()

    first = var_monte_carlo_student(returns, weights, df=5.0, seed=7)
    second = var_monte_carlo_student(returns, weights, df=5.0, seed=7)
    other = var_monte_carlo_student(returns, weights, df=5.0, seed=8)

    assert first == second
    assert first != other


def test_shared_mixing_creates_tail_dependence():
    """P(les deux actifs simultanément dans leur queue 1 %) : t partagée >> normale."""
    rng_t = np.random.default_rng(21)
    rng_n = np.random.default_rng(21)
    mu = np.zeros(2)
    sigma = np.array([0.01, 0.01])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = corr * np.outer(sigma, sigma)
    n = 200_000

    sims_t = _simulate_student_returns(mu, sigma, corr, df=4.0, n_sims=n, rng=rng_t)
    sims_n = _simulate_normal_returns(mu, cov, n_sims=n, rng=rng_n)

    def joint_tail_probability(sims: np.ndarray) -> float:
        q0 = np.quantile(sims[:, 0], 0.01)
        q1 = np.quantile(sims[:, 1], 0.01)
        return float(np.mean((sims[:, 0] < q0) & (sims[:, 1] < q1)))

    assert joint_tail_probability(sims_t) > 2.0 * joint_tail_probability(sims_n)


def test_invalid_inputs_rejected():
    returns, weights = _returns_frame(n=100)

    with pytest.raises(ValueError, match="df"):
        var_monte_carlo_student(returns, weights, df=2.0)
    constant = returns.copy()
    constant["AAA"] = 0.01
    with pytest.raises(ValueError, match="constant column"):
        var_monte_carlo_student(constant, weights, df=5.0)
