"""Tests du traffic light bâlois (SPEC.md B3.6 et B3.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from riskplatform.backtest import (
    basel_zone_bounds,
    kupiec_pof,
    rolling_traffic_light,
    traffic_light,
)


def _exceptions(n_obs: int, one_positions: list[int]) -> pd.Series:
    values = np.zeros(n_obs, dtype=int)
    values[one_positions] = 1
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.Series(values, index=index, name="exception")


# ------------------------------------------------------------- zone bounds


def test_canonical_bounds_match_basel_table():
    """À (99 %, 250 j), la CDF binomiale redonne 0-4 / 5-9 / >= 10 (Bâle 1996)."""
    assert basel_zone_bounds(alpha=0.99, window=250) == (4, 9)


def test_bounds_are_derived_from_binomial_cdf():
    """Propriété générique : green_max = dernier k avec P(X<=k) < 0.95."""
    for alpha, window in ((0.99, 250), (0.95, 250), (0.99, 500)):
        green_max, yellow_max = basel_zone_bounds(alpha=alpha, window=window)
        p = 1.0 - alpha
        assert binom.cdf(green_max, window, p) < 0.95 <= binom.cdf(green_max + 1, window, p)
        assert binom.cdf(yellow_max, window, p) < 0.9999 <= binom.cdf(yellow_max + 1, window, p)
        assert green_max < yellow_max


def test_bounds_invalid_inputs_raise():
    with pytest.raises(ValueError, match="alpha"):
        basel_zone_bounds(alpha=1.2)
    with pytest.raises(ValueError, match="window"):
        basel_zone_bounds(window=0)


# ------------------------------------------------------------ traffic_light


@pytest.mark.parametrize(
    ("n_exceptions", "zone", "plus_factor"),
    [
        (0, "green", 0.0),
        (4, "green", 0.0),
        (5, "yellow", 0.40),
        (6, "yellow", 0.50),
        (7, "yellow", 0.65),
        (8, "yellow", 0.75),
        (9, "yellow", 0.85),
        (10, "red", 1.00),
        (15, "red", 1.00),
    ],
)
def test_zones_and_plus_factors_on_canonical_window(n_exceptions, zone, plus_factor):
    """Table de Bâle 1996 complète : zone + plus-factor + multiplicateur."""
    exceptions = _exceptions(250, list(range(n_exceptions)))
    result = traffic_light(exceptions, alpha=0.99, window=250)

    assert result["n_exceptions"] == n_exceptions
    assert result["zone"] == zone
    assert result["plus_factor"] == pytest.approx(plus_factor)
    assert result["multiplier"] == pytest.approx(3.0 + plus_factor)
    assert result["cum_prob"] == pytest.approx(float(binom.cdf(n_exceptions, 250, 0.01)))


def test_counts_last_window_not_first():
    """10 exceptions en tête de série, 0 dans les 250 derniers points -> verte
    (tue la mutation « compter les 250 PREMIERS points »)."""
    early = traffic_light(_exceptions(500, list(range(10))), window=250)
    late = traffic_light(_exceptions(500, list(range(490, 500))), window=250)

    assert early["zone"] == "green"
    assert early["n_exceptions"] == 0
    assert late["zone"] == "red"
    assert late["n_exceptions"] == 10


def test_non_canonical_config_has_no_plus_factor():
    result = traffic_light(_exceptions(250, list(range(5))), alpha=0.95, window=250)
    assert result["plus_factor"] is None
    assert result["multiplier"] is None
    assert result["zone"] == "green"  # 5 exceptions pour 12.5 attendues à 95 %


def test_red_zone_implies_kupiec_rejection():
    """Cohérence croisée : une série en zone rouge est aussi rejetée par Kupiec."""
    exceptions = _exceptions(250, list(range(10)))
    assert traffic_light(exceptions)["zone"] == "red"
    assert kupiec_pof(exceptions, alpha=0.99)["reject"] is True


def test_too_short_series_raises():
    with pytest.raises(ValueError, match="at least 250"):
        traffic_light(_exceptions(249, [0]))


def test_non_binary_series_raises():
    series = pd.Series([0, 1, 2] * 100, index=pd.date_range("2020-01-01", periods=300, freq="B"))
    with pytest.raises(ValueError, match="0/1"):
        traffic_light(series)


# ---------------------------------------------------- rolling_traffic_light


def test_rolling_zones_track_the_window():
    """5 exceptions consécutives : jaune tant qu'elles sont dans la fenêtre."""
    exceptions = _exceptions(300, list(range(255, 260)))
    rolling = rolling_traffic_light(exceptions, window=250)

    assert len(rolling) == 51  # 300 - 250 + 1 dates produites
    assert rolling.loc[exceptions.index[254], "zone"] == "green"  # avant les exceptions
    assert rolling.loc[exceptions.index[259], "zone"] == "yellow"  # les 5 dans la fenêtre
    assert rolling.loc[exceptions.index[259], "n_exceptions"] == 5
    assert (rolling["zone"].iloc[-40:] == "yellow").all()  # elles restent 250 j


def test_rolling_first_date_is_the_window_th_observation():
    exceptions = _exceptions(260, [])
    rolling = rolling_traffic_light(exceptions, window=250)
    assert rolling.index[0] == exceptions.index[249]
    assert (rolling["zone"] == "green").all()


def test_rolling_too_short_raises():
    with pytest.raises(ValueError, match="at least 250"):
        rolling_traffic_light(_exceptions(100, [0]))
