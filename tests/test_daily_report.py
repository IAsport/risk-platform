"""Tests du rapport quotidien HTML (SPEC.md B4.3 / B4.5).

Le HTML doit être autonome (figure base64, aucune référence externe) et
mentionner explicitement les sections indisponibles au lieu de crasher.
"""

from __future__ import annotations

import dataclasses
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from riskplatform import pipeline
from riskplatform.config import RunConfig
from riskplatform.portfolio import make_equal_weight
from riskplatform.reporting import render_daily_report

TICKERS = ["AAA", "AAPL", "MSFT", "NVDA"]
CURRENCIES = {"AAA": "EUR", "AAPL": "USD", "MSFT": "USD", "NVDA": "USD"}


def _config() -> RunConfig:
    return RunConfig(
        name="test-daily",
        portfolio=make_equal_weight(TICKERS, CURRENCIES),
        start="2024-01-01",
        end="2026-03-01",
        alphas=(0.95, 0.99),
        horizon_days=1,
    )


def _analysis(n_dates: int) -> pipeline.RiskAnalysis:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(11)

    def fake_load_returns(tickers, currencies, start, end, cache_dir=None):
        returns = pd.DataFrame(
            0.01 * rng.standard_normal((len(dates), len(tickers))),
            index=dates,
            columns=list(tickers),
        )
        return (1.0 + returns).cumprod() * 100.0, returns

    with mock.patch.object(pipeline.data, "load_returns", fake_load_returns):
        return pipeline.run_analysis(_config(), cache_dir=None)


@pytest.fixture(scope="module")
def analysis() -> pipeline.RiskAnalysis:
    # 550 dates : 300 points de backtest -> traffic light présent.
    return _analysis(550)


def test_daily_report_is_self_contained_and_shows_key_figures(analysis, tmp_path):
    out_path = tmp_path / "daily_report.html"

    html = render_daily_report(analysis, out_path=str(out_path))

    assert out_path.read_text(encoding="utf-8") == html
    assert analysis.as_of.date().isoformat() in html
    assert "Rapport de risque quotidien — test-daily" in html
    # Zone traffic light rendue en badge.
    assert any(zone in html for zone in ("VERTE", "JAUNE", "ROUGE"))
    # ES 97,5 % FRTB, top risques et dernières exceptions datées.
    assert "ES 97,5 %" in html
    assert analysis.stress.worst in html
    assert "Dernières exceptions" in html or "Aucune exception" in html
    # Autonome : figure embarquée, aucune référence externe.
    assert "data:image/png;base64," in html
    assert "http" not in html


def test_daily_report_var_values_scaled_by_notional(analysis):
    html = render_daily_report(analysis, out_path=None)

    notional = analysis.config.portfolio.notional_eur
    hist_99 = next(
        row
        for row in analysis.var_results
        if row["method"] == "historical" and row["alpha"] == 0.99
    )
    assert f"{hist_99['var'] * notional:,.0f} EUR" in html


def test_daily_report_short_sample_flags_missing_traffic_light():
    # 300 dates : backtests présents mais 50 points < 250 -> pas de zone.
    html = render_daily_report(_analysis(300), out_path=None)

    assert "&lt; 250 pts" in html
    assert "data:image/png;base64," in html


def test_daily_report_without_stress_or_backtests_mentions_it(analysis):
    stripped = dataclasses.replace(analysis, stress=None, backtest_results={})

    html = render_daily_report(stripped, out_path=None)

    assert "Suite de stress indisponible" in html
    assert "Aucun backtest disponible" in html
    assert "Pas de série de backtest à tracer" in html
