"""Tests du package stress (SPEC.md B3.2-B3.5 et B3.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.stress import (
    DEFAULT_SCENARIOS,
    HistoricalWindow,
    IndexShock,
    PriceShock,
    RiskParamShock,
    apply_index_shock,
    apply_price_shock,
    estimate_betas,
    replay_window,
    run_stress_suite,
    stressed_var_parametric,
    worst_window,
)

NOTIONAL = 1_000_000.0


@pytest.fixture()
def two_asset_returns() -> pd.DataFrame:
    """3 jours x 2 titres, valeurs choisies pour un calcul à la main."""
    dates = pd.date_range("2020-03-02", periods=3, freq="B")
    return pd.DataFrame(
        {"AAA": [0.01, -0.05, 0.02], "BBB": [-0.02, -0.03, 0.01]}, index=dates
    )


@pytest.fixture()
def equal_weights() -> pd.Series:
    return pd.Series({"AAA": 0.5, "BBB": 0.5})


@pytest.fixture()
def market_returns() -> pd.DataFrame:
    """Échantillon simulé corrélé (250 jours x 3 titres) pour les chocs de paramètres."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=250, freq="B")
    common = rng.normal(0.0, 0.01, size=250)
    frame = pd.DataFrame(
        {
            "AAA": common + rng.normal(0.0, 0.006, size=250),
            "BBB": common + rng.normal(0.0, 0.009, size=250),
            "CCC": common + rng.normal(0.0, 0.012, size=250),
        },
        index=dates,
    )
    return frame


@pytest.fixture()
def market_weights() -> pd.Series:
    return pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})


# ---------------------------------------------------------------- replay B3.2


def test_replay_window_hand_computed(two_asset_returns, equal_weights):
    """R_i = exp(sum r_i) - 1 exact, P&L = N·w_i·R_i, perte = -P&L total."""
    scenario = HistoricalWindow("test", "2020-03-02", "2020-03-04")
    result = replay_window(two_asset_returns, equal_weights, scenario, notional=NOTIONAL)

    shock_aaa = np.expm1(0.01 - 0.05 + 0.02)
    shock_bbb = np.expm1(-0.02 - 0.03 + 0.01)
    assert result.kind == "historical"
    assert result.pnl_by_position["AAA"] == pytest.approx(NOTIONAL * 0.5 * shock_aaa)
    assert result.pnl_by_position["BBB"] == pytest.approx(NOTIONAL * 0.5 * shock_bbb)
    assert result.pnl_total == pytest.approx(
        NOTIONAL * 0.5 * (shock_aaa + shock_bbb)
    )
    assert result.loss == pytest.approx(-result.pnl_total)


def test_replay_uses_exact_arithmetic_not_log_approx():
    """Sur un choc cumulé de -50 % en log, l'approximation log donnerait une
    perte de 50 % ; l'exacte donne 1 - exp(-0.5) = 39.35 % (B3.10 #1)."""
    dates = pd.date_range("2020-03-02", periods=5, freq="B")
    returns = pd.DataFrame({"AAA": [-0.1] * 5}, index=dates)
    weights = pd.Series({"AAA": 1.0})
    scenario = HistoricalWindow("crash", "2020-03-02", "2020-03-06")

    result = replay_window(returns, weights, scenario, notional=1.0)

    assert result.loss == pytest.approx(1.0 - np.exp(-0.5))
    assert result.loss != pytest.approx(0.5, abs=1e-3)  # tue la mutation exp(Σr)-1 -> Σr


def test_replay_window_outside_data_raises(two_asset_returns, equal_weights):
    scenario = HistoricalWindow("vide", "2019-01-01", "2019-06-30")
    with pytest.raises(ValueError, match="no return dates"):
        replay_window(two_asset_returns, equal_weights, scenario)


def test_historical_window_inverted_dates_raise():
    with pytest.raises(ValueError, match="must be < end"):
        HistoricalWindow("inversé", "2020-03-18", "2020-02-19")


def test_historical_window_bad_date_raises():
    with pytest.raises(ValueError, match="ISO dates"):
        HistoricalWindow("mauvaise date", "2020/02/19", "2020-03-18")


def test_replay_missing_ticker_raises(two_asset_returns):
    weights = pd.Series({"AAA": 0.5, "ZZZ": 0.5})
    scenario = HistoricalWindow("test", "2020-03-02", "2020-03-04")
    with pytest.raises(ValueError, match="missing returns"):
        replay_window(two_asset_returns, weights, scenario)


# ---------------------------------------------------------- worst_window B3.2


def test_worst_window_finds_planted_trough():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    values = np.full(100, 0.001)
    values[50:55] = -0.05  # creux placé sur 5 jours
    pnl = pd.Series(values, index=dates)

    window = worst_window(pnl, horizon=5)

    assert window.start == dates[50].date().isoformat()
    assert window.end == dates[54].date().isoformat()


def test_worst_window_too_short_raises():
    pnl = pd.Series([0.01, -0.02], index=pd.date_range("2021-01-01", periods=2, freq="B"))
    with pytest.raises(ValueError, match="at least"):
        worst_window(pnl, horizon=5)
    with pytest.raises(ValueError, match="horizon"):
        worst_window(pnl, horizon=0)


# ------------------------------------------------------------ PriceShock B3.3


def test_uniform_price_shock_loses_exactly_x_times_notional(equal_weights):
    """Sanity structurel : chocs identiques + somme des poids = 1 => perte = x·N."""
    result = apply_price_shock(equal_weights, PriceShock("uniforme", -0.20), notional=NOTIONAL)
    assert result.loss == pytest.approx(0.20 * NOTIONAL)
    assert result.kind == "price"


def test_partial_price_shock_leaves_other_tickers_untouched(equal_weights):
    result = apply_price_shock(
        equal_weights, PriceShock("partiel", {"AAA": -0.30}), notional=NOTIONAL
    )
    assert result.pnl_by_position["AAA"] == pytest.approx(-NOTIONAL * 0.5 * 0.30)
    assert result.pnl_by_position["BBB"] == 0.0


def test_price_shock_unknown_ticker_raises(equal_weights):
    with pytest.raises(ValueError, match="not in portfolio"):
        apply_price_shock(equal_weights, PriceShock("inconnu", {"ZZZ": -0.10}))


# ------------------------------------------------------------ IndexShock B3.3


def test_index_shock_propagates_exact_betas():
    """r_i = beta_i · r_b sans bruit => bêtas OLS exacts et P&L = N·w_i·beta_i·choc."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    bench = pd.Series(rng.normal(0.0, 0.01, size=120), index=dates, name="IDX")
    returns = pd.DataFrame({"AAA": 1.2 * bench, "BBB": 0.6 * bench}, index=dates)
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})

    betas = estimate_betas(returns, bench)
    assert betas["AAA"] == pytest.approx(1.2)
    assert betas["BBB"] == pytest.approx(0.6)

    result = apply_index_shock(
        returns, bench, weights, IndexShock("indice", -0.15), notional=NOTIONAL
    )
    assert result.pnl_by_position["AAA"] == pytest.approx(NOTIONAL * 0.5 * 1.2 * -0.15)
    assert result.pnl_by_position["BBB"] == pytest.approx(NOTIONAL * 0.5 * 0.6 * -0.15)
    assert result.kind == "index"


def test_index_shock_without_benchmark_raises(two_asset_returns, equal_weights):
    with pytest.raises(ValueError, match="requires benchmark_returns"):
        apply_index_shock(two_asset_returns, None, equal_weights, IndexShock("indice", -0.15))


def test_estimate_betas_constant_benchmark_raises(two_asset_returns):
    bench = pd.Series(0.0, index=two_asset_returns.index)
    with pytest.raises(ValueError, match="variance is zero"):
        estimate_betas(two_asset_returns, bench)


def test_estimate_betas_disjoint_dates_raise(two_asset_returns):
    bench = pd.Series(
        [0.01, 0.02, 0.03], index=pd.date_range("1999-01-01", periods=3, freq="B")
    )
    with pytest.raises(ValueError, match="common dates"):
        estimate_betas(two_asset_returns, bench)


# -------------------------------------------------------- RiskParamShock B3.4


def test_no_shock_leaves_var_unchanged(market_returns, market_weights):
    result = stressed_var_parametric(
        market_returns, market_weights, RiskParamShock("neutre"), alpha=0.99
    )
    assert result.var_stressed == pytest.approx(result.var_base)
    assert result.ratio == pytest.approx(1.0)


def test_vol_shock_scales_var_linearly(market_returns, market_weights):
    """sigma_i -> k·sigma_i uniforme, s=0 => VaR* = k·VaR (homogénéité)."""
    result = stressed_var_parametric(
        market_returns, market_weights, RiskParamShock("vol x2", vol_multiplier=2.0)
    )
    assert result.var_stressed == pytest.approx(2.0 * result.var_base)


def test_corr_to_one_reaches_comonotonic_bound(market_returns, market_weights):
    """s=1 => sigma_p* = somme w_i·sigma_i (la diversification meurt)."""
    result = stressed_var_parametric(
        market_returns, market_weights, RiskParamShock("rho 1", corr_shift=1.0), alpha=0.99
    )
    sigma_vec = market_returns.loc[:, market_weights.index].std(ddof=1)
    sigma_comonotonic = float((market_weights * sigma_vec).sum())
    from scipy.stats import norm

    expected_var = -norm.ppf(0.01) * sigma_comonotonic
    assert result.var_stressed == pytest.approx(expected_var, rel=1e-9)
    assert result.var_stressed > result.var_base  # la corrélation ne peut qu'aggraver


def test_blended_correlation_stays_positive_semidefinite(market_returns, market_weights):
    cov = market_returns.cov().to_numpy()
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    for s in (0.0, 0.5, 1.0):
        blended = (1.0 - s) * corr + s * np.ones_like(corr)
        eigenvalues = np.linalg.eigvalsh(blended)
        assert eigenvalues.min() >= -1e-10


def test_stressed_es_dominates_stressed_var(market_returns, market_weights):
    """ES >= VaR reste vraie sous stress (propriété systématique, B2)."""
    for scenario in (
        RiskParamShock("vol x2", vol_multiplier=2.0),
        RiskParamShock("rho 1", corr_shift=1.0),
        RiskParamShock("systemique", vol_multiplier=2.0, corr_shift=1.0),
    ):
        result = stressed_var_parametric(market_returns, market_weights, scenario)
        assert result.es_stressed > result.var_stressed
        assert result.es_base > result.var_base


def test_single_asset_correlation_shock_is_noop():
    """Portefeuille à 1 actif : rho n'a pas de sens, sigma_p* = k·sigma (légal)."""
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    rng = np.random.default_rng(11)
    returns = pd.DataFrame({"AAA": rng.normal(0.0, 0.01, size=100)}, index=dates)
    weights = pd.Series({"AAA": 1.0})

    result = stressed_var_parametric(
        returns, weights, RiskParamShock("seul", vol_multiplier=2.0, corr_shift=1.0)
    )
    assert result.var_stressed == pytest.approx(2.0 * result.var_base)


def test_risk_param_shock_invalid_parameters_raise():
    with pytest.raises(ValueError, match="vol_multiplier"):
        RiskParamShock("mauvais k", vol_multiplier=0.0)
    with pytest.raises(ValueError, match="corr_shift"):
        RiskParamShock("mauvais s", corr_shift=1.5)
    with pytest.raises(ValueError, match="corr_shift"):
        RiskParamShock("s negatif", corr_shift=-0.1)


def test_constant_column_raises(market_weights):
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0, 0.01, size=100),
            "BBB": np.zeros(100),
            "CCC": rng.normal(0.0, 0.01, size=100),
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="constant column"):
        stressed_var_parametric(returns, market_weights, RiskParamShock("neutre"))


# --------------------------------------------------------- run_stress_suite


@pytest.fixture()
def suite_inputs(market_returns, market_weights):
    rng = np.random.default_rng(21)
    bench = pd.Series(
        rng.normal(0.0, 0.01, size=len(market_returns)),
        index=market_returns.index,
        name="IDX",
    )
    scenarios = (
        HistoricalWindow("fenetre test", "2021-03-01", "2021-04-30"),
        PriceShock("uniforme -20 %", -0.20),
        PriceShock("choc AAA", {"AAA": -0.30}),
        IndexShock("indice -15 %", -0.15),
        RiskParamShock("systemique", vol_multiplier=2.0, corr_shift=1.0),
    )
    return market_returns, market_weights, bench, scenarios


def test_suite_builds_both_panels(suite_inputs):
    returns, weights, bench, scenarios = suite_inputs
    suite = run_stress_suite(
        returns, weights, notional=NOTIONAL, benchmark_returns=bench,
        scenarios=scenarios, add_worst_window=True, horizon=10,
    )

    assert len(suite.pnl_table) == 5  # 4 scénarios P&L + pire fenêtre extraite
    assert len(suite.risk_table) == 1
    assert list(suite.pnl_by_position.columns) == list(weights.index)
    assert set(suite.pnl_table.columns) == {
        "kind", "loss_eur", "pct_notional", "ratio_var", "ratio_capital"
    }
    # Le choc uniforme -20 % domine les autres scénarios sur cet échantillon calme.
    assert suite.worst == "uniforme -20 %"
    assert suite.pnl_table.loc["uniforme -20 %", "loss_eur"] == pytest.approx(0.20 * NOTIONAL)


def test_suite_ratios_are_consistent(suite_inputs):
    returns, weights, bench, scenarios = suite_inputs
    var_ref = 50_000.0
    suite = run_stress_suite(
        returns, weights, notional=NOTIONAL, benchmark_returns=bench,
        scenarios=scenarios, add_worst_window=False, var_ref=var_ref,
    )
    row = suite.pnl_table.loc["uniforme -20 %"]
    assert row["ratio_var"] == pytest.approx(row["loss_eur"] / var_ref)
    assert row["ratio_capital"] == pytest.approx(
        row["loss_eur"] / (3.0 * np.sqrt(10.0) * var_ref)
    )
    assert row["pct_notional"] == pytest.approx(0.20)


def test_suite_index_shock_without_benchmark_raises(suite_inputs):
    returns, weights, _, scenarios = suite_inputs
    with pytest.raises(ValueError, match="requires benchmark_returns"):
        run_stress_suite(returns, weights, scenarios=scenarios, add_worst_window=False)


def test_suite_rejects_non_positive_var_ref(suite_inputs):
    returns, weights, bench, scenarios = suite_inputs
    with pytest.raises(ValueError, match="var_ref"):
        run_stress_suite(
            returns, weights, benchmark_returns=bench, scenarios=scenarios, var_ref=0.0
        )


def test_suite_rejects_unknown_scenario_type(market_returns, market_weights):
    with pytest.raises(ValueError, match="unsupported scenario type"):
        run_stress_suite(
            market_returns, market_weights, scenarios=("pas un scenario",),
            add_worst_window=False,
        )


def test_default_catalogue_shape():
    """Le catalogue spec B3.5 : 5 scénarios de P&L (2 fenêtres + 2 prix +
    1 indice) et 3 chocs de paramètres."""
    pnl_like = [
        s for s in DEFAULT_SCENARIOS
        if isinstance(s, HistoricalWindow | PriceShock | IndexShock)
    ]
    risk_like = [s for s in DEFAULT_SCENARIOS if isinstance(s, RiskParamShock)]
    assert len(pnl_like) == 5
    assert len(risk_like) == 3
