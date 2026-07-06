from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskplatform.data import (
    convert_to_eur,
    download_fx,
    download_prices,
    load_returns,
    to_log_returns,
)


def test_to_log_returns_formula_matches_definition():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"ABC": [100.0, 110.0]}, index=dates)

    returns = to_log_returns(prices)

    assert returns.loc[dates[1], "ABC"] == pytest.approx(np.log(110.0 / 100.0))


def test_to_log_returns_drops_first_nan_row():
    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"ABC": [100.0, 110.0, 121.0]}, index=dates)

    returns = to_log_returns(prices)

    assert list(returns.index) == [dates[1], dates[2]]


def test_to_log_returns_rejects_nan_values():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"ABC": [100.0, np.nan]}, index=dates)

    with pytest.raises(ValueError, match="missing values"):
        to_log_returns(prices)


def test_convert_to_eur_divides_usd_by_eurusd():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame(
        {"US": [108.0, 216.0], "EU": [50.0, 55.0]},
        index=dates,
    )
    eurusd = pd.Series([1.08, 1.20], index=dates)

    converted = convert_to_eur(prices, {"US": "USD", "EU": "EUR"}, eurusd)

    assert converted.loc[dates[0], "US"] == pytest.approx(100.0)
    assert converted.loc[dates[1], "US"] == pytest.approx(180.0)
    assert converted["EU"].equals(prices["EU"])


def test_convert_to_eur_raises_when_currency_is_missing():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"US": [108.0, 120.0]}, index=dates)
    eurusd = pd.Series([1.08, 1.20], index=dates)

    with pytest.raises(ValueError, match="missing currencies"):
        convert_to_eur(prices, {}, eurusd)


def test_convert_to_eur_raises_when_currency_is_unsupported():
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"GB": [90.0, 91.0]}, index=dates)
    eurusd = pd.Series([1.08, 1.20], index=dates)

    with pytest.raises(ValueError, match="unsupported currencies"):
        convert_to_eur(prices, {"GB": "GBP"}, eurusd)


def test_convert_to_eur_aligns_dates_with_fx_and_drops_unmatched_start():
    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"US": [108.0, 120.0, 132.0]}, index=dates)
    eurusd = pd.Series([1.20], index=[dates[1]])

    converted = convert_to_eur(prices, {"US": "USD"}, eurusd)

    assert list(converted.index) == [dates[1], dates[2]]
    assert converted.loc[dates[1], "US"] == pytest.approx(100.0)
    assert converted.loc[dates[2], "US"] == pytest.approx(110.0)


def test_convert_to_eur_mixed_portfolio_ffills_middle_fx_hole_and_drops_start():
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "US": [108.0, 120.0, 132.0, 121.0],
            "EU": [50.0, 51.0, 52.0, 53.0],
        },
        index=dates,
    )
    eurusd = pd.Series([1.20, np.nan, 1.10], index=dates[1:])

    converted = convert_to_eur(prices, {"US": "USD", "EU": "EUR"}, eurusd)

    assert list(converted.index) == [dates[1], dates[2], dates[3]]
    assert converted.loc[dates[1], "US"] == pytest.approx(100.0)
    assert converted.loc[dates[2], "US"] == pytest.approx(110.0)
    assert converted.loc[dates[3], "US"] == pytest.approx(110.0)
    pd.testing.assert_series_equal(converted["EU"], prices.loc[dates[1]:, "EU"])


def test_download_prices_raises_when_tickers_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        download_prices([], "2024-01-01", "2024-01-04")


def test_download_prices_raises_when_tickers_are_duplicated():
    with pytest.raises(ValueError, match="must be unique"):
        download_prices(["AAA", "AAA"], "2024-01-01", "2024-01-04")


def test_download_prices_inner_join_drops_partial_dates(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame(
        {
            ("Close", "AAA"): [10.0, 11.0, np.nan],
            ("Close", "BBB"): [20.0, np.nan, 22.0],
        },
        index=dates,
    )

    def fake_download(tickers, start, end, auto_adjust, progress):
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    prices = download_prices(["AAA", "BBB"], "2024-01-01", "2024-01-04")

    expected = pd.DataFrame({"AAA": [10.0], "BBB": [20.0]}, index=[dates[0]])
    pd.testing.assert_frame_equal(prices, expected, check_freq=False)


def _mock_yfinance(monkeypatch, raw: pd.DataFrame) -> None:
    def fake_download(tickers, start, end, auto_adjust, progress):
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))


def test_download_prices_raises_on_empty_download(monkeypatch):
    _mock_yfinance(monkeypatch, pd.DataFrame())

    with pytest.raises(ValueError, match="no price data downloaded"):
        download_prices(["AAA"], "2024-01-01", "2024-01-04")


def test_download_prices_raises_when_a_ticker_has_no_data(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame(
        {
            ("Close", "AAA"): [10.0, 11.0],
            ("Close", "BBB"): [np.nan, np.nan],
        },
        index=dates,
    )
    _mock_yfinance(monkeypatch, raw)

    with pytest.raises(ValueError, match="missing price data for tickers"):
        download_prices(["AAA", "BBB"], "2024-01-01", "2024-01-03")


def test_download_prices_single_ticker_flat_columns(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame({"Close": [10.0, 11.0, 12.0], "Volume": [1, 1, 1]}, index=dates)
    _mock_yfinance(monkeypatch, raw)

    prices = download_prices(["AAA"], "2024-01-01", "2024-01-04")

    assert list(prices.columns) == ["AAA"]
    assert prices["AAA"].tolist() == [10.0, 11.0, 12.0]


def test_download_fx_multiindex_close(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame({("Close", "EURUSD=X"): [1.08, 1.09, 1.10]}, index=dates)
    _mock_yfinance(monkeypatch, raw)

    fx = download_fx("EURUSD=X", "2024-01-01", "2024-01-04")

    assert isinstance(fx, pd.Series)
    assert fx.name == "EURUSD=X"
    assert fx.tolist() == [1.08, 1.09, 1.10]


def test_download_fx_flat_columns_drops_nan(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame({"Close": [1.08, np.nan, 1.10]}, index=dates)
    _mock_yfinance(monkeypatch, raw)

    fx = download_fx("EURUSD=X", "2024-01-01", "2024-01-04")

    assert fx.tolist() == [1.08, 1.10]


def test_download_fx_raises_on_empty_download(monkeypatch):
    _mock_yfinance(monkeypatch, pd.DataFrame())

    with pytest.raises(ValueError, match="no FX data downloaded"):
        download_fx("EURUSD=X", "2024-01-01", "2024-01-04")


def test_load_returns_raises_when_currency_is_missing():
    with pytest.raises(ValueError, match="missing currencies"):
        load_returns(["US"], {}, "2024-01-01", "2024-01-04")


def test_load_returns_cache_write_through_then_offline(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=4)
    local_prices = pd.DataFrame({"US": [100.0, 110.0, 121.0, 133.1]}, index=dates)
    fx = pd.Series([1.10, 1.10, 1.10, 1.10], index=dates, name="EURUSD=X")
    downloads = {"prices": 0}

    def fake_prices(tickers, start, end):
        downloads["prices"] += 1
        return local_prices

    monkeypatch.setattr("riskplatform.data.loader.download_prices", fake_prices)
    monkeypatch.setattr("riskplatform.data.loader.download_fx", lambda pair, start, end: fx)

    prices_1, returns_1 = load_returns(
        ["US"], {"US": "USD"}, "2024-01-01", "2024-01-06", cache_dir=tmp_path
    )

    assert downloads["prices"] == 1
    assert (tmp_path / "prices.csv").is_file()
    assert (tmp_path / "eurusd.csv").is_file()

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted while cache exists")

    monkeypatch.setattr("riskplatform.data.loader.download_prices", no_network)
    monkeypatch.setattr("riskplatform.data.loader.download_fx", no_network)

    prices_2, returns_2 = load_returns(
        ["US"], {"US": "USD"}, "2024-01-01", "2024-01-06", cache_dir=tmp_path
    )

    pd.testing.assert_frame_equal(prices_1, prices_2, check_freq=False)
    pd.testing.assert_frame_equal(returns_1, returns_2, check_freq=False)


def test_cache_slices_requested_period(tmp_path):
    dates = pd.date_range("2024-01-01", periods=5)
    pd.DataFrame({"US": [100.0, 110.0, 121.0, 133.1, 146.4]}, index=dates).to_csv(
        tmp_path / "prices.csv"
    )
    pd.Series([1.1] * 5, index=dates, name="EURUSD=X").to_csv(tmp_path / "eurusd.csv")

    prices_eur, _ = load_returns(
        ["US"], {"US": "USD"}, "2024-01-02", "2024-01-05", cache_dir=tmp_path
    )

    assert list(prices_eur.index) == list(dates[1:4])  # end exclusif


def test_cache_missing_ticker_raises(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3)
    pd.DataFrame({"US": [100.0, 110.0, 121.0]}, index=dates).to_csv(tmp_path / "prices.csv")
    pd.Series([1.1] * 3, index=dates, name="EURUSD=X").to_csv(tmp_path / "eurusd.csv")

    with pytest.raises(ValueError, match="lack tickers"):
        load_returns(
            ["US", "OTHER"],
            {"US": "USD", "OTHER": "EUR"},
            "2024-01-01",
            "2024-01-04",
            cache_dir=tmp_path,
        )


def test_load_returns_returns_prices_and_returns_aligned(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    local_prices = pd.DataFrame({"US": [108.0, 120.0, 132.0]}, index=dates)
    fx = pd.Series([1.08, 1.20, 1.20], index=dates)

    monkeypatch.setattr(
        "riskplatform.data.loader.download_prices",
        lambda tickers, start, end: local_prices,
    )
    monkeypatch.setattr("riskplatform.data.loader.download_fx", lambda pair, start, end: fx)

    prices_eur, returns = load_returns(
        ["US"],
        {"US": "USD"},
        "2024-01-01",
        "2024-01-04",
    )

    assert list(prices_eur.columns) == ["US"]
    assert list(returns.columns) == ["US"]
    assert list(returns.index) == list(prices_eur.index[1:])
