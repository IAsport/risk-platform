from __future__ import annotations

import sys

import pandas as pd
import pytest

from riskplatform import cli, pipeline
from riskplatform.config import RunConfig
from riskplatform.portfolio import make_equal_weight

REFERENCE_TICKERS = [
    "TTE.PA",
    "MC.PA",
    "SAN.PA",
    "BNP.PA",
    "AIR.PA",
    "AAPL",
    "MSFT",
    "NVDA",
]
REFERENCE_CURRENCIES = {
    "TTE.PA": "EUR",
    "MC.PA": "EUR",
    "SAN.PA": "EUR",
    "BNP.PA": "EUR",
    "AIR.PA": "EUR",
    "AAPL": "USD",
    "MSFT": "USD",
    "NVDA": "USD",
}


def _reference_config(**overrides) -> RunConfig:
    params = {
        "name": "test",
        "portfolio": make_equal_weight(REFERENCE_TICKERS, REFERENCE_CURRENCIES),
        "start": "2024-01-01",
        "end": "2024-01-06",
        "alphas": (0.95,),
        "horizon_days": 2,
    }
    params.update(overrides)
    return RunConfig(**params)


def test_main_loads_yaml_and_applies_cli_overrides(monkeypatch, tmp_path):
    config_file = tmp_path / "portfolio.yaml"
    config_file.write_text(
        """
name: mini
start: "2020-01-01"
end: "2021-01-01"
alphas: [0.95, 0.99]
horizon_days: 1
positions:
  - {ticker: AAA, currency: EUR}
  - {ticker: BBB, currency: USD}
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run(config, cache_dir="data/cache"):
        captured["config"] = config
        captured["cache_dir"] = cache_dir

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["riskplatform", "--config", str(config_file), "--alphas", "0.9", "--horizon-days", "5"],
    )

    cli.main()

    config = captured["config"]
    assert captured["cache_dir"] == "data/cache"  # défaut CLI
    assert config.name == "mini"
    assert list(config.portfolio.weights.index) == ["AAA", "BBB"]
    assert config.alphas == (0.9,)  # override CLI
    assert config.horizon_days == 5  # override CLI
    assert config.start == "2020-01-01"  # YAML conservé (pas d'override)
    assert config.end == "2021-01-01"


def test_run_pipeline_with_monkeypatched_data(monkeypatch):
    # 300 dates : assez pour déclencher le traffic light (fenêtre 250) et la
    # pire fenêtre 20 j de la suite de stress.
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    pattern = [0.001, -0.002, 0.003, -0.004, 0.005] * 60
    tickers_seen: list[str] = []
    rendered: dict[str, object] = {}

    def fake_load_returns(tickers, currencies, start, end, cache_dir=None):
        tickers_seen.extend(tickers)
        returns = pd.DataFrame({ticker: pattern for ticker in tickers}, index=dates)
        prices = pd.DataFrame({ticker: range(100, 400) for ticker in tickers}, index=dates)
        return prices, returns

    def fake_rolling_var(pnl_returns, method, alpha, window=250, notional=1.0):
        return pd.Series(0.001, index=dates[2:])

    def fake_count_exceptions(realized_returns, var_series, notional=1.0):
        return pd.Series(0, index=var_series.index)

    def fake_render_report(var_results, backtest_results, out_dir="outputs"):
        rendered["var_results"] = var_results
        rendered["backtest_results"] = backtest_results
        rendered["out_dir"] = out_dir

    def fake_render_stress_report(suite, out_dir="outputs"):
        rendered["stress_suite"] = suite

    def fake_render_daily_report(analysis, out_path="outputs/daily_report.html"):
        rendered["daily_analysis"] = analysis
        return "<html></html>"

    # Depuis la B4 le calcul vit dans riskplatform.pipeline (SPEC.md B4.1) :
    # les monkeypatches ciblent les modules via `pipeline` (mêmes objets).
    monkeypatch.setattr(pipeline.data, "load_returns", fake_load_returns)
    monkeypatch.setattr(pipeline.var, "var_historical", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(
        pipeline.var, "var_parametric_portfolio", lambda *args, **kwargs: 0.02
    )
    monkeypatch.setattr(pipeline.var, "var_monte_carlo", lambda *args, **kwargs: 0.03)
    monkeypatch.setattr(pipeline.es, "expected_shortfall", lambda *args, **kwargs: 0.04)
    monkeypatch.setattr(
        pipeline.var, "scale_var", lambda value, horizon_days: value * horizon_days
    )
    monkeypatch.setattr(pipeline.var, "rolling_var", fake_rolling_var)
    monkeypatch.setattr(pipeline.backtest, "count_exceptions", fake_count_exceptions)
    monkeypatch.setattr(
        pipeline.backtest,
        "kupiec_pof",
        lambda exceptions, alpha: {
            "n_obs": len(exceptions),
            "n_exceptions": int(exceptions.sum()),
            "expected": 0.1,
            "lr_stat": 0.0,
            "p_value": 1.0,
            "reject": False,
        },
    )
    monkeypatch.setattr(
        pipeline.backtest,
        "christoffersen_cc",
        lambda exceptions, alpha: {"lr_stat": 0.0, "p_value": 1.0, "reject": False},
    )
    monkeypatch.setattr(cli.report, "render_report", fake_render_report)
    monkeypatch.setattr(cli.report, "render_stress_report", fake_render_stress_report)
    monkeypatch.setattr(cli.daily_report, "render_daily_report", fake_render_daily_report)

    cli.run(_reference_config())

    assert tickers_seen == REFERENCE_TICKERS
    assert rendered["out_dir"] == "outputs"
    # Le rapport quotidien reçoit le même RiskAnalysis que le reste du rendu.
    assert rendered["daily_analysis"].backtest_results is rendered["backtest_results"]
    assert len(rendered["var_results"]) == 3
    assert {row["method"] for row in rendered["var_results"]} == {
        "historical",
        "parametric",
        "monte_carlo",
    }
    assert set(rendered["backtest_results"]) == {"historical_95", "parametric_95"}

    # Traffic light : 0 exception sur les 250 dernières dates -> zone verte ;
    # alpha=0.95 est hors config canonique -> pas de plus-factor.
    hist_backtest = rendered["backtest_results"]["historical_95"]
    assert hist_backtest["tl_zone"] == "green"
    assert hist_backtest["tl_exceptions_250d"] == 0
    assert hist_backtest["tl_plus_factor"] is None

    # Stress : pas de benchmark configuré -> scénario indiciel écarté ; les
    # fenêtres 2020/2022 sont hors échantillon (2024) -> écartées aussi.
    suite = rendered["stress_suite"]
    assert list(suite.risk_table.index) == [
        "Volatilites x2",
        "Correlations -> 1",
        "Crise systemique (sigma x2, rho -> 1)",
    ]
    assert len(suite.pnl_table) == 3  # 2 chocs de prix + pire fenêtre 20 j
    assert suite.worst == "Actions uniformes -20 %"
    assert suite.pnl_table.loc["Actions uniformes -20 %", "loss_eur"] == pytest.approx(
        0.20 * 1_000_000.0
    )
