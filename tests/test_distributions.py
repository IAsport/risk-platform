from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import t as student_t

from riskplatform.distributions import fit_student_df, student_quantile_std


def _standardized_t_sample(df: float, n: int, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    raw = student_t.rvs(df, size=n, random_state=rng)
    return pd.Series(raw * np.sqrt((df - 2.0) / df))


def test_quantile_std_has_unit_variance_density():
    df = 6.0
    scale = np.sqrt((df - 2.0) / df)
    variance, _ = quad(
        lambda x: x**2 * student_t.pdf(x / scale, df) / scale, -np.inf, np.inf
    )
    assert variance == pytest.approx(1.0, rel=1e-8)


def test_quantile_std_converges_to_normal_for_large_df():
    assert student_quantile_std(0.01, 1e6) == pytest.approx(norm.ppf(0.01), abs=1e-3)


def test_quantile_std_fatter_tail_than_normal_for_small_df():
    # À variance égale (1), la t_4 standardisée a un quantile 1 % plus extrême.
    assert student_quantile_std(0.01, 4.0) < norm.ppf(0.01)


def test_quantile_std_hand_value():
    # t_5 : t⁻¹(0.01) = -3.3649, écart std : sqrt(3/5).
    expected = student_t.ppf(0.01, 5) * np.sqrt(3 / 5)
    assert student_quantile_std(0.01, 5.0) == pytest.approx(expected)


def test_fit_recovers_df_on_simulated_t():
    sample = _standardized_t_sample(df=5.0, n=5000)

    fitted = fit_student_df(sample)

    assert fitted == pytest.approx(5.0, abs=1.0)


def test_fit_hits_upper_bound_on_gaussian_data():
    rng = np.random.default_rng(3)
    sample = pd.Series(rng.standard_normal(5000))

    fitted = fit_student_df(sample)

    assert fitted > 50.0  # données ≈ gaussiennes -> nu très grand


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="df"):
        student_quantile_std(0.01, 2.0)
    with pytest.raises(ValueError, match="p"):
        student_quantile_std(1.5, 5.0)
    with pytest.raises(ValueError, match="at least"):
        fit_student_df(pd.Series([0.1] * 10))
    with pytest.raises(ValueError, match="missing"):
        fit_student_df(pd.Series([0.1] * 60 + [np.nan]))
    with pytest.raises(ValueError, match="bounds"):
        fit_student_df(_standardized_t_sample(5.0, 100), bounds=(1.0, 10.0))
