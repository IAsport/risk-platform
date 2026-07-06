from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.portfolio import covariance_matrix, make_equal_weight, portfolio_returns


def test_make_equal_weight_sums_to_one():
    portfolio = make_equal_weight(
        ["AAA", "BBB", "CCC"],
        {"AAA": "EUR", "BBB": "USD", "CCC": "EUR"},
    )

    assert portfolio.weights.sum() == pytest.approx(1.0)
    assert portfolio.weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_make_equal_weight_raises_when_tickers_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        make_equal_weight([], {})


def test_make_equal_weight_raises_when_tickers_are_duplicated():
    with pytest.raises(ValueError, match="must be unique"):
        make_equal_weight(["AAA", "AAA"], {"AAA": "EUR"})


def test_portfolio_returns_is_weighted_sum():
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame(
        {"AAA": [0.01, 0.03], "BBB": [0.03, 0.05]},
        index=dates,
    )
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})

    result = portfolio_returns(returns, weights)

    expected = pd.Series([0.02, 0.04], index=dates)
    pd.testing.assert_series_equal(result, expected)


def test_portfolio_returns_raises_when_returns_empty():
    returns = pd.DataFrame(columns=["AAA"])
    weights = pd.Series({"AAA": 1.0})

    with pytest.raises(ValueError, match="returns must not be empty"):
        portfolio_returns(returns, weights)


def test_portfolio_returns_raises_when_weights_empty():
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"AAA": [0.01, 0.02]}, index=dates)
    weights = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="weights must not be empty"):
        portfolio_returns(returns, weights)


def test_portfolio_returns_reorders_columns_to_match_weights():
    dates = pd.date_range("2024-01-01", periods=1)
    returns = pd.DataFrame({"BBB": [0.10], "AAA": [0.02]}, index=dates)
    weights = pd.Series({"AAA": 0.25, "BBB": 0.75})

    result = portfolio_returns(returns, weights)

    assert result.iloc[0] == pytest.approx(0.08)


def test_covariance_matrix_self_covariance_equals_variance():
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.04, 0.08]}, index=dates)

    cov = covariance_matrix(returns)

    assert cov.loc["AAA", "AAA"] == pytest.approx(returns["AAA"].var())


def test_covariance_matrix_with_weights_keeps_only_weighted_tickers():
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.04, 0.05, 0.06],
            "CCC": [0.07, 0.08, 0.09],
        },
        index=dates,
    )
    weights = pd.Series({"BBB": 0.6, "AAA": 0.4})

    cov = covariance_matrix(returns, weights)

    assert cov.shape == (2, 2)
    assert list(cov.index) == ["BBB", "AAA"]
    assert list(cov.columns) == ["BBB", "AAA"]
    assert "CCC" not in cov.index
    assert "CCC" not in cov.columns


def test_covariance_matrix_cross_covariance_matches_manual_value():
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [2.0, 4.0, 6.0],
        },
        index=dates,
    )
    expected_cross_covariance = ((1.0 - 2.0) * (2.0 - 4.0) + 0.0 + (3.0 - 2.0) * (6.0 - 4.0)) / 2

    cov = covariance_matrix(returns)

    assert cov.loc["AAA", "BBB"] == pytest.approx(expected_cross_covariance)


def test_covariance_matrix_raises_when_weights_do_not_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"AAA": [0.01, 0.02], "BBB": [0.03, 0.04]}, index=dates)
    weights = pd.Series({"AAA": 0.5, "BBB": 0.4})

    with pytest.raises(ValueError, match="sum to 1.0"):
        covariance_matrix(returns, weights)


def test_covariance_matrix_raises_when_returns_empty():
    returns = pd.DataFrame(columns=["AAA"])

    with pytest.raises(ValueError, match="returns must not be empty"):
        covariance_matrix(returns)


def test_covariance_matrix_raises_when_weights_empty():
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"AAA": [0.01, 0.02]}, index=dates)
    weights = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="weights must not be empty"):
        covariance_matrix(returns, weights)


def test_covariance_matrix_is_symmetric_psd():
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"AAA": [0.01, 0.02, 0.03, 0.04], "BBB": [0.04, 0.01, 0.03, 0.02]},
        index=dates,
    )

    cov = covariance_matrix(returns)

    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() >= -1e-12


def test_portfolio_returns_raises_when_weights_do_not_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"AAA": [0.01, 0.02], "BBB": [0.03, 0.04]}, index=dates)
    weights = pd.Series({"AAA": 0.5, "BBB": 0.4})

    with pytest.raises(ValueError, match="sum to 1.0"):
        portfolio_returns(returns, weights)
