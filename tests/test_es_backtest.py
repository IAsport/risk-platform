from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskplatform.backtest import acerbi_szekely_z2
from riskplatform.es import es_conditional
from riskplatform.var import var_conditional


def _forecasts(n: int = 500, sigma: float = 0.01, alpha: float = 0.99):
    """Prévisions cohérentes d'un modèle normal à sigma constant."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    sigma_series = pd.Series(sigma, index=dates)
    var_series = var_conditional(sigma_series, alpha=alpha)
    es_series = es_conditional(sigma_series, alpha=alpha)
    assert isinstance(var_series, pd.Series) and isinstance(es_series, pd.Series)
    return dates, sigma_series, var_series, es_series


def test_correct_model_is_not_rejected():
    dates, sigma_series, var_series, es_series = _forecasts()
    rng = np.random.default_rng(15)
    realized = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)

    result = acerbi_szekely_z2(realized, var_series, es_series, sigma_series, alpha=0.99)

    assert result["reject"] is False
    assert result["p_value"] > 0.05
    # Z2 est bruité à ~5 exceptions attendues : on borne large, le verdict prime.
    assert abs(result["z_stat"]) < 1.0


def test_underestimated_es_is_rejected():
    dates, sigma_series, var_series, es_series = _forecasts(sigma=0.01)
    rng = np.random.default_rng(15)
    # Pertes réelles deux fois plus volatiles que ce que le modèle annonce.
    realized = pd.Series(rng.normal(0.0, 0.02, len(dates)), index=dates)

    result = acerbi_szekely_z2(realized, var_series, es_series, sigma_series, alpha=0.99)

    assert result["z_stat"] > 0
    assert result["reject"] is True


def test_zero_exception_gives_z_minus_one_without_nan():
    dates, sigma_series, var_series, es_series = _forecasts()
    realized = pd.Series(0.0001, index=dates)  # aucune perte notable

    result = acerbi_szekely_z2(realized, var_series, es_series, sigma_series, alpha=0.99)

    assert result["n_exceptions"] == 0
    assert result["z_stat"] == pytest.approx(-1.0)
    assert np.isfinite(result["p_value"])
    assert result["reject"] is False  # VaR trop prudente = pas de sous-estimation


def test_student_h0_innovations_supported():
    dates, sigma_series, _, _ = _forecasts()
    df = 5.0
    from riskplatform.distributions import student_quantile_std

    var_series = var_conditional(sigma_series, alpha=0.99, df=df)
    es_series = es_conditional(sigma_series, alpha=0.99, df=df)
    rng = np.random.default_rng(8)
    eps = rng.standard_t(df, len(dates)) * np.sqrt((df - 2) / df)
    realized = pd.Series(0.01 * eps, index=dates)

    result = acerbi_szekely_z2(
        realized, var_series, es_series, sigma_series, alpha=0.99, df=df
    )

    assert result["reject"] is False
    assert student_quantile_std(0.01, df) < norm.ppf(0.01)  # cohérence quantiles


def test_series_aligned_on_intersection():
    dates, sigma_series, var_series, es_series = _forecasts(n=100)
    realized = pd.Series(0.0, index=dates[10:])  # 90 dates communes

    result = acerbi_szekely_z2(realized, var_series, es_series, sigma_series)

    assert result["n_obs"] == 90


def test_invalid_inputs_rejected():
    dates, sigma_series, var_series, es_series = _forecasts(n=100)
    realized = pd.Series(0.0, index=dates)

    with pytest.raises(ValueError, match="strictly positive"):
        acerbi_szekely_z2(realized, var_series, es_series * 0.0, sigma_series)
    with pytest.raises(ValueError, match="empty date intersection"):
        acerbi_szekely_z2(realized.iloc[:0].reindex(pd.date_range("1999", periods=3)).fillna(0),
                          var_series, es_series, sigma_series)
    with pytest.raises(ValueError, match="df"):
        acerbi_szekely_z2(realized, var_series, es_series, sigma_series, df=2.0)
