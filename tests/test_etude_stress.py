"""Verrouillage des verdicts de l'étude B3 (SPEC.md B3.7) sur le snapshot committé.

PLAN §8 : pas de résultat affiché sans test. Les chiffres du notebook
etude_stress_traffic_light et du README (stress + traffic light) sont
vérifiés ici sur data/cache/ (snapshot 2026-07-06, benchmark ^STOXX50E
inclus) — offline, déterministe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from riskplatform.backtest import count_exceptions, rolling_traffic_light, traffic_light
from riskplatform.config import load_config
from riskplatform.data import load_returns
from riskplatform.portfolio import portfolio_returns
from riskplatform.stress import run_stress_suite
from riskplatform.var import rolling_var, rolling_var_conditional

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.99
NOTIONAL = 1_000_000.0


@pytest.fixture(scope="module")
def market():
    config = load_config(REPO / "config" / "portfolio.yaml")
    cache = REPO / "data" / "cache"
    tickers = list(config.portfolio.weights.index)
    _, returns = load_returns(
        tickers, config.portfolio.currencies, config.start, config.end, cache_dir=cache
    )
    _, bench_frame = load_returns(
        [config.benchmark_ticker],
        {config.benchmark_ticker: config.benchmark_currency},
        config.start,
        config.end,
        cache_dir=cache,
    )
    benchmark = bench_frame[config.benchmark_ticker]
    return config, returns, benchmark


@pytest.fixture(scope="module")
def suite(market):
    config, returns, benchmark = market
    return run_stress_suite(
        returns,
        config.portfolio.weights,
        notional=NOTIONAL,
        benchmark_returns=benchmark,
        alpha=ALPHA,
    )


# ------------------------------------------------------------------- stress


def test_worst_window_extraction_lands_on_march_2020(suite):
    """L'extraction automatique PROUVE que le pire épisode 20 j est mars 2020
    (à un jour près de la fenêtre COVID datée dans la spec)."""
    assert suite.worst == "Pire fenetre 20 j (2020-02-20 -> 2020-03-18)"


def test_covid_replay_exceeds_var_and_capital_proxy(suite):
    """Le replay COVID perd ~37 % du notional : > 11x la VaR 99 % 1 j et
    au-delà du proxy de capital 3·sqrt(10)·VaR — la démonstration que le
    stress test couvre ce que la VaR ne voit pas."""
    covid = suite.pnl_table.loc["COVID-19 (19/02-18/03/2020)"]

    assert 0.35 < covid["pct_notional"] < 0.40  # ~36.9 % du notional
    assert covid["ratio_var"] > 10.0
    assert covid["ratio_capital"] > 1.0


def test_rate_hike_2022_energy_gains_tech_loses(suite):
    """Lecture par position du scénario taux 2022 : TotalEnergies finit POSITIF
    (énergie, année de choc pétrolier) et NVDA est la pire position — la table
    par position raconte la rotation sectorielle."""
    row = suite.pnl_by_position.loc["Hausse des taux 2022 (03/01-12/10)"]

    assert row["TTE.PA"] > 0.0
    assert row.idxmin() == "NVDA"
    assert suite.pnl_table.loc["Hausse des taux 2022 (03/01-12/10)", "ratio_var"] > 4.0


def test_correlation_shock_kills_one_third_of_diversification(suite):
    """rho -> 1 : VaR paramétrique x ~1.57 — la diversification vaut ~36 % de
    la VaR sur ce portefeuille ; le combiné systémique est le produit des deux
    chocs (2 x 1.57)."""
    corr = suite.risk_table.loc["Correlations -> 1"]
    vol = suite.risk_table.loc["Volatilites x2"]
    combined = suite.risk_table.loc["Crise systemique (sigma x2, rho -> 1)"]

    assert 1.5 < corr["ratio"] < 1.65
    assert vol["ratio"] == pytest.approx(2.0)
    assert combined["ratio"] == pytest.approx(2.0 * corr["ratio"], rel=1e-9)


def test_index_shock_average_beta_below_one(suite):
    """Choc indice -15 % propagé par bêtas : perte ~12.8 % du notional, soit un
    bêta moyen pondéré ~0.85 (le bloc USD est peu corrélé au Stoxx)."""
    index_row = suite.pnl_table.loc["Euro Stoxx 50 -15 % (betas)"]
    implied_beta = index_row["pct_notional"] / 0.15

    assert 0.8 < implied_beta < 0.9


# ------------------------------------------------------------ traffic light


@pytest.fixture(scope="module")
def zones(market):
    config, returns, _ = market
    pnl = portfolio_returns(returns, config.portfolio.weights)
    models = {
        "parametric": rolling_var(pnl, "parametric", alpha=ALPHA, window=250),
        "ewma_t": rolling_var_conditional(
            pnl, "ewma", alpha=ALPHA, window=1000, refit_every=20, dist="student"
        ),
        "garch_t": rolling_var_conditional(
            pnl, "garch", alpha=ALPHA, window=1000, refit_every=20, dist="student"
        ),
    }
    result = {}
    for name, var_series in models.items():
        exceptions = count_exceptions(pnl.loc[var_series.index], var_series)
        result[name] = {
            "rolling": rolling_traffic_light(exceptions, alpha=ALPHA, window=250),
            "last": traffic_light(exceptions, alpha=ALPHA, window=250),
        }
    return result


def test_parametric_stays_red_for_years(zones):
    """La paramétrique 250 j passe au rouge dès novembre 2018 (vol du T4 2018,
    max 15 exceptions/250 j) et n'en sort qu'en février 2021 : ~480 jours
    ouvrés en zone rouge, multiplicateur plein 4.0 pendant plus de deux ans."""
    rolling = zones["parametric"]["rolling"]
    red = rolling.index[rolling["zone"] == "red"]

    assert 400 < len(red) < 600  # 483 sur le snapshot
    assert red.min() < pd.Timestamp("2019-01-01")  # rouge avant même le COVID
    assert red.max() > pd.Timestamp("2021-01-01")
    assert int(rolling["n_exceptions"].max()) == 15


def test_conditional_t_models_touch_red_only_around_covid(zones):
    """Nuance honnête vs l'attendu spec (« les conditionnels-t restent
    verts/jaunes ») : EWMA-t et GARCH-t touchent AUSSI le rouge, mais
    seulement ~80 jours (mars-août 2020, max 11 exceptions : les sauts jour-1
    diagnostiqués en B2), puis en sortent — quand la paramétrique y reste
    coincée jusqu'en 2021."""
    parametric_rolling = zones["parametric"]["rolling"]
    n_red_parametric = int((parametric_rolling["zone"] == "red").sum())
    for name in ("ewma_t", "garch_t"):
        rolling = zones[name]["rolling"]
        red = rolling.index[rolling["zone"] == "red"]

        assert 50 < len(red) < 120  # 81 dates sur le snapshot
        assert (red.year == 2020).all()
        assert int(rolling["n_exceptions"].max()) == 11
        assert len(red) < n_red_parametric / 4  # ~6x moins que la paramétrique


def test_all_models_are_green_today_with_floor_multiplier(zones):
    """Sur les 250 dernières dates du snapshot, les trois modèles sont en zone
    verte : multiplicateur plancher 3.0 (config canonique 99 %/250 j)."""
    for name in ("parametric", "ewma_t", "garch_t"):
        last = zones[name]["last"]
        assert last["zone"] == "green"
        assert last["multiplier"] == pytest.approx(3.0)


def test_garch_t_is_greenest_of_the_three(zones):
    """Classement : GARCH-t passe ~62 % de ses dates en vert, EWMA-t ~53 %,
    la paramétrique ~45 % (et 18.5 % en rouge)."""
    shares = {
        name: (data["rolling"]["zone"] == "green").mean() for name, data in zones.items()
    }
    assert shares["garch_t"] > shares["ewma_t"] > shares["parametric"]
    red_share = (zones["parametric"]["rolling"]["zone"] == "red").mean()
    assert red_share > 0.15
