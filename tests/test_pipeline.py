"""Tests du pipeline réutilisable (SPEC.md B4.1 / B4.5).

Données fabriquées, aucun réseau : seul `data.load_returns` (la frontière
réseau) est monkeypatché ; tout le quant tourne en vrai.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform import pipeline
from riskplatform.config import RunConfig
from riskplatform.portfolio import make_equal_weight

# Inclut AAPL/MSFT/NVDA : le scénario sectoriel du catalogue les choque, et le
# moteur est strict sur les tickers inconnus (SPEC.md B3.3).
TICKERS = ["AAA", "AAPL", "MSFT", "NVDA"]
CURRENCIES = {"AAA": "EUR", "AAPL": "USD", "MSFT": "USD", "NVDA": "USD"}
# 550 dates : > 250 (fenêtre rolling) + 250 (traffic light) -> tl_* présents.
DATES = pd.date_range("2024-01-01", periods=550, freq="B")


def _fabricated_returns(tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    values = 0.01 * rng.standard_normal((len(DATES), len(tickers)))
    return pd.DataFrame(values, index=DATES, columns=tickers)


def _config(**overrides) -> RunConfig:
    params = {
        "name": "test",
        "portfolio": make_equal_weight(TICKERS, CURRENCIES),
        "start": "2024-01-01",
        "end": "2026-03-01",
        "alphas": (0.95, 0.99),
        "horizon_days": 1,
    }
    params.update(overrides)
    return RunConfig(**params)


@pytest.fixture
def patched_loader(monkeypatch):
    """load_returns fabriqué : gère l'appel portefeuille ET l'appel benchmark."""

    def fake_load_returns(tickers, currencies, start, end, cache_dir=None):
        returns = _fabricated_returns(list(tickers))
        prices = (1.0 + returns).cumprod() * 100.0
        return prices, returns

    monkeypatch.setattr(pipeline.data, "load_returns", fake_load_returns)


def test_run_analysis_is_silent_and_coherent(patched_loader, capsys):
    analysis = pipeline.run_analysis(_config(), cache_dir=None)

    assert capsys.readouterr().out == ""  # SPEC B4.1 : aucun print

    # VaR/ES : 3 méthodes x 2 alphas, VaR > 0, ES partagée par alpha,
    # ES historique >= VaR historique (propriété B2).
    assert len(analysis.var_results) == 6
    for alpha in (0.95, 0.99):
        rows = [row for row in analysis.var_results if row["alpha"] == alpha]
        assert {row["method"] for row in rows} == {"historical", "parametric", "monte_carlo"}
        assert all(row["var"] > 0 for row in rows)
        assert len({row["es"] for row in rows}) == 1
        hist = next(row for row in rows if row["method"] == "historical")
        assert hist["es"] >= hist["var"]

    # Backtests : 2 méthodes x 2 alphas, séries alignées, traffic light présent
    # (550 - 250 = 300 points de prévision >= 250).
    assert set(analysis.backtest_results) == {
        "historical_95",
        "parametric_95",
        "historical_99",
        "parametric_99",
    }
    for result in analysis.backtest_results.values():
        assert {"n_obs", "n_exceptions", "p_value", "reject", "cc_p_value"} <= set(result)
        assert len(result["var_series"]) == len(result["exceptions"])
        assert result["tl_zone"] in {"green", "yellow", "red"}
        assert 0 <= result["tl_exceptions_250d"] <= 250

    assert analysis.as_of == DATES[-1]
    assert analysis.benchmark_returns is None


def test_run_analysis_skips_inapplicable_scenarios(patched_loader):
    analysis = pipeline.run_analysis(_config(), cache_dir=None)

    # Échantillon 2024-2026, pas de benchmark : les 2 fenêtres historiques et
    # le scénario indiciel sont écartés avec leur raison (SPEC B4.1).
    reasons = dict(analysis.skipped_scenarios)
    assert len(reasons) == 3
    assert reasons["Euro Stoxx 50 -15 % (betas)"] == "no benchmark configured"
    assert set(reasons.values()) == {"no benchmark configured", "window outside sample"}

    suite = analysis.stress
    assert suite is not None
    assert not (suite.pnl_table["kind"] == "index").any()
    # Aucun scénario écarté ne figure dans les tables.
    for name in reasons:
        assert name not in suite.pnl_table.index
        assert name not in suite.risk_table.index
    # La pire fenêtre 20 j extraite des données est bien ajoutée.
    assert (suite.pnl_table["kind"] == "historical").sum() == 1


def test_run_analysis_with_benchmark_runs_index_scenario(patched_loader):
    analysis = pipeline.run_analysis(
        _config(benchmark_ticker="^IDX", benchmark_currency="EUR"), cache_dir=None
    )

    assert analysis.benchmark_returns is not None
    assert (analysis.stress.pnl_table["kind"] == "index").any()
    reasons = dict(analysis.skipped_scenarios)
    assert set(reasons.values()) == {"window outside sample"}


def test_run_analysis_wraps_market_data_errors(monkeypatch):
    def failing_load_returns(tickers, currencies, start, end, cache_dir=None):
        raise ValueError("no data for AAA")

    monkeypatch.setattr(pipeline.data, "load_returns", failing_load_returns)

    with pytest.raises(RuntimeError, match=r"ticker\(s\) \['AAA'\]"):
        pipeline.run_analysis(_config(), cache_dir=None)
