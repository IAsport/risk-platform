from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from riskplatform.backtest import (
    christoffersen_cc,
    christoffersen_independence,
    count_exceptions,
    kupiec_pof,
)


def test_count_exceptions_flags_loss_above_var():
    dates = pd.date_range("2024-01-01", periods=4)
    realized_returns = pd.Series([-0.02, -0.05, 0.01, -0.11], index=dates)
    var_series = pd.Series([0.01, 0.04, 0.02, 0.10], index=dates)

    result = count_exceptions(realized_returns, var_series)

    expected = pd.Series([1, 1, 0, 1], index=dates, name="exception")
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_count_exceptions_aligns_dates():
    realized_dates = pd.date_range("2024-01-01", periods=3)
    var_dates = pd.date_range("2024-01-02", periods=3)
    realized_returns = pd.Series([-0.10, -0.03, -0.07], index=realized_dates)
    var_series = pd.Series([0.02, 0.06, 0.01], index=var_dates)

    result = count_exceptions(realized_returns, var_series)

    expected = pd.Series([1, 1], index=realized_dates[1:], name="exception")
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_kupiec_not_rejected_when_rate_matches_p():
    exceptions = pd.Series([1] * 10 + [0] * 90)

    result = kupiec_pof(exceptions, alpha=0.90)

    assert result["n_obs"] == 100
    assert result["n_exceptions"] == 10
    assert result["expected"] == pytest.approx(10.0)
    assert result["lr_stat"] == pytest.approx(0.0)
    assert result["reject"] is False


def test_kupiec_rejected_when_too_many_exceptions():
    exceptions = pd.Series([1] * 25 + [0] * 75)

    result = kupiec_pof(exceptions, alpha=0.99)

    assert result["lr_stat"] > 3.841
    assert result["p_value"] < 0.05
    assert result["reject"] is True


def test_kupiec_handles_zero_exceptions():
    exceptions = pd.Series([0] * 100)

    result = kupiec_pof(exceptions, alpha=0.99)

    assert result["n_exceptions"] == 0
    assert np.isfinite(result["lr_stat"])
    assert np.isfinite(result["p_value"])


def test_kupiec_statistic_is_chi2_with_known_value():
    exceptions = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    alpha = 0.90
    p = 1.0 - alpha
    t_obs = 10
    x = 3
    pi_hat = x / t_obs
    expected_lr = -2.0 * (
        (t_obs - x) * math.log(1.0 - p)
        + x * math.log(p)
        - ((t_obs - x) * math.log(1.0 - pi_hat) + x * math.log(pi_hat))
    )

    result = kupiec_pof(exceptions, alpha=alpha)

    assert result["lr_stat"] == pytest.approx(expected_lr, abs=1e-6)


def test_christoffersen_independence_detects_clustering():
    exceptions = pd.Series([0] * 50 + [1] * 50)

    result = christoffersen_independence(exceptions)

    assert result["n00"] == 49
    assert result["n01"] == 1
    assert result["n10"] == 0
    assert result["n11"] == 49
    assert result["lr_stat"] > 3.841
    assert result["reject"] is True


def test_christoffersen_independence_does_not_reject_spread_exceptions():
    exceptions = pd.Series(
        [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    result = christoffersen_independence(exceptions)

    assert result["lr_stat"] < 3.841
    assert result["reject"] is False


def test_christoffersen_independence_handles_missing_transition():
    exceptions = pd.Series([0, 0, 0, 0, 0])

    result = christoffersen_independence(exceptions)

    assert result["n00"] == 4
    assert result["n01"] == 0
    assert result["n10"] == 0
    assert result["n11"] == 0
    assert np.isfinite(result["lr_stat"])
    assert np.isfinite(result["p_value"])
    assert result["reject"] is False


def test_christoffersen_cc_is_sum_of_pof_and_ind():
    exceptions = pd.Series([0] * 50 + [1] * 50)

    pof = kupiec_pof(exceptions, alpha=0.50)
    independence = christoffersen_independence(exceptions)
    cc = christoffersen_cc(exceptions, alpha=0.50)

    assert cc["lr_stat"] == pytest.approx(pof["lr_stat"] + independence["lr_stat"])
    assert cc["lr_pof"] == pytest.approx(pof["lr_stat"])
    assert cc["lr_ind"] == pytest.approx(independence["lr_stat"])
    assert cc["p_value"] == pytest.approx(chi2.sf(cc["lr_stat"], df=2))
    assert cc["reject"] == (cc["lr_stat"] > 5.991)


def test_christoffersen_cc_uses_chi2_df_two_for_p_value():
    exceptions = pd.Series(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    )

    cc = christoffersen_cc(exceptions, alpha=0.90)

    assert 2.0 < cc["lr_stat"] < 4.0
    assert cc["p_value"] == pytest.approx(chi2.sf(cc["lr_stat"], df=2), rel=1e-9)
    assert cc["p_value"] != pytest.approx(chi2.sf(cc["lr_stat"], df=1), rel=1e-3)


def test_christoffersen_independence_statistic_matches_manual_value():
    exceptions = pd.Series([0, 1, 0, 0, 1, 1, 0])
    n00, n01, n10, n11 = 1, 2, 2, 1
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    expected_lr = -2.0 * (
        (n00 + n10) * math.log(1.0 - pi)
        + (n01 + n11) * math.log(pi)
        - (
            n00 * math.log(1.0 - pi01)
            + n01 * math.log(pi01)
            + n10 * math.log(1.0 - pi11)
            + n11 * math.log(pi11)
        )
    )

    result = christoffersen_independence(exceptions)

    assert result["n00"] == n00
    assert result["n01"] == n01
    assert result["n10"] == n10
    assert result["n11"] == n11
    assert result["lr_stat"] == pytest.approx(expected_lr, abs=1e-6)
    assert result["lr_stat"] == pytest.approx(0.6795961471815897, abs=1e-6)


def test_kupiec_handles_all_exceptions():
    exceptions = pd.Series([1] * 100)

    result = kupiec_pof(exceptions, alpha=0.99)

    assert result["n_exceptions"] == 100
    assert np.isfinite(result["lr_stat"])
    assert np.isfinite(result["p_value"])


def test_empty_series_raises_value_error():
    empty = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="must not be empty"):
        kupiec_pof(empty)

    with pytest.raises(ValueError, match="must not be empty"):
        christoffersen_independence(empty)


def test_count_exceptions_raises_when_intersection_is_empty():
    realized_returns = pd.Series([-0.01], index=[pd.Timestamp("2024-01-01")])
    var_series = pd.Series([0.02], index=[pd.Timestamp("2024-01-02")])

    with pytest.raises(ValueError, match="empty intersection"):
        count_exceptions(realized_returns, var_series)


def test_invalid_alpha_raises_value_error():
    exceptions = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="alpha"):
        kupiec_pof(exceptions, alpha=1.0)

    with pytest.raises(ValueError, match="alpha"):
        christoffersen_cc(exceptions, alpha=0.0)
