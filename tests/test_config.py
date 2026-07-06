from __future__ import annotations

from pathlib import Path

import pytest

from riskplatform.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "portfolio.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "portfolio.yaml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = """
start: "2020-01-01"
positions:
  - {{ticker: AAA, currency: EUR}}
  - {{ticker: BBB, currency: USD}}
{extra}
"""


def test_reference_config_loads_expected_portfolio():
    config = load_config(REPO_CONFIG)

    assert config.name == "reference"
    assert list(config.portfolio.weights.index) == [
        "TTE.PA",
        "MC.PA",
        "SAN.PA",
        "BNP.PA",
        "AIR.PA",
        "AAPL",
        "MSFT",
        "NVDA",
    ]
    assert config.portfolio.weights.tolist() == pytest.approx([1 / 8] * 8)
    assert config.portfolio.currencies["TTE.PA"] == "EUR"
    assert config.portfolio.currencies["NVDA"] == "USD"
    assert config.portfolio.notional_eur == pytest.approx(1_000_000.0)
    assert config.start == "2014-01-01"
    assert config.end is None
    assert config.alphas == (0.95, 0.99)
    assert config.horizon_days == 1
    # Benchmark HORS poids : présent dans la config, absent du portefeuille.
    assert config.benchmark_ticker == "^STOXX50E"
    assert config.benchmark_currency == "EUR"
    assert "^STOXX50E" not in config.portfolio.weights.index


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_default_values_and_equal_weights(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL.format(extra="")))

    assert config.name == "reference"
    assert config.end is None
    assert config.alphas == (0.95, 0.99)
    assert config.horizon_days == 1
    assert config.benchmark_ticker is None
    assert config.portfolio.weights.tolist() == pytest.approx([0.5, 0.5])
    assert config.portfolio.notional_eur == pytest.approx(1_000_000.0)


def test_explicit_weights_must_sum_to_one(tmp_path):
    body = """
start: "2020-01-01"
positions:
  - {ticker: AAA, currency: EUR, weight: 0.6}
  - {ticker: BBB, currency: USD, weight: 0.5}
"""
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(_write(tmp_path, body))


def test_partial_weights_rejected(tmp_path):
    body = """
start: "2020-01-01"
positions:
  - {ticker: AAA, currency: EUR, weight: 1.0}
  - {ticker: BBB, currency: USD}
"""
    with pytest.raises(ValueError, match="all positions define 'weight' or none"):
        load_config(_write(tmp_path, body))


def test_explicit_weights_accepted(tmp_path):
    body = """
start: "2020-01-01"
positions:
  - {ticker: AAA, currency: EUR, weight: 0.7}
  - {ticker: BBB, currency: USD, weight: 0.3}
"""
    config = load_config(_write(tmp_path, body))
    assert config.portfolio.weights.tolist() == pytest.approx([0.7, 0.3])


def test_duplicate_tickers_rejected(tmp_path):
    body = """
start: "2020-01-01"
positions:
  - {ticker: AAA, currency: EUR}
  - {ticker: AAA, currency: EUR}
"""
    with pytest.raises(ValueError, match="unique"):
        load_config(_write(tmp_path, body))


def test_unsupported_currency_rejected(tmp_path):
    body = """
start: "2020-01-01"
positions:
  - {ticker: AAA, currency: GBP}
"""
    with pytest.raises(ValueError, match="unsupported currency"):
        load_config(_write(tmp_path, body))


def test_empty_positions_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty 'positions'"):
        load_config(_write(tmp_path, 'start: "2020-01-01"\npositions: []\n'))


def test_invalid_dates_rejected(tmp_path):
    with pytest.raises(ValueError, match="ISO date"):
        load_config(_write(tmp_path, MINIMAL.format(extra="").replace("2020-01-01", "01/01/2020")))

    body = MINIMAL.format(extra='end: "2019-01-01"')
    with pytest.raises(ValueError, match="after start"):
        load_config(_write(tmp_path, body))


def test_alpha_out_of_range_rejected(tmp_path):
    with pytest.raises(ValueError, match="alphas"):
        load_config(_write(tmp_path, MINIMAL.format(extra="alphas: [1.5]")))


def test_horizon_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="horizon_days"):
        load_config(_write(tmp_path, MINIMAL.format(extra="horizon_days: 0")))


def test_negative_notional_rejected(tmp_path):
    with pytest.raises(ValueError, match="notional_eur"):
        load_config(_write(tmp_path, MINIMAL.format(extra="notional_eur: -5")))


def test_benchmark_requires_ticker_and_currency(tmp_path):
    with pytest.raises(ValueError, match="benchmark"):
        load_config(_write(tmp_path, MINIMAL.format(extra="benchmark: {ticker: XXX}")))
