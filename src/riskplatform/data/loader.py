"""Module data — acquisition des prix, change EUR/USD, log-returns.

Responsabilité : récupérer les clôtures ajustées (yfinance), le taux EURUSD,
convertir en EUR et produire la table de log-returns. AUCUN calcul de risque ici.
Voir ARCHITECTURE.md §1 et SPEC.md §1-2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Télécharge les clôtures ajustées via yfinance.

    Args:
        tickers: symboles (ex. ["TTE.PA", "AAPL"]).
        start, end: bornes "YYYY-MM-DD" (end exclusif côté yfinance).

    Returns:
        DataFrame index=DatetimeIndex, colonnes=tickers, prix en devise LOCALE
        (EUR pour .PA, USD pour les US), NaN éliminés par jointure interne.

    Hypothèse: clôtures journalières ajustées dividendes/splits ('Adj Close').
    """
    if not tickers:
        raise ValueError("tickers must not be empty")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique")

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("no price data downloaded")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        elif "Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Close", axis=1, level=1)
        elif "Adj Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Adj Close", axis=1, level=1)
        else:
            raise ValueError("downloaded data does not contain adjusted close prices")
    else:
        column = "Close" if "Close" in raw.columns else "Adj Close"
        if column not in raw.columns:
            raise ValueError("downloaded data does not contain adjusted close prices")
        prices = raw[[column]].rename(columns={column: tickers[0]})

    prices = prices.reindex(columns=tickers)
    if prices.isna().all(axis=0).any():
        missing = prices.columns[prices.isna().all(axis=0)].tolist()
        raise ValueError(f"missing price data for tickers: {missing}")

    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("empty price index after joining tickers on common dates")
    return prices


def download_fx(pair: str, start: str, end: str) -> pd.Series:
    """Télécharge le taux de change journalier (ex. 'EURUSD=X').

    Returns:
        Series index=dates, valeur = nombre d'USD pour 1 EUR (EURUSD).
    """
    import yfinance as yf

    raw = yf.download(pair, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"no FX data downloaded for {pair}")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            fx = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            fx = raw["Adj Close"]
        else:
            raise ValueError(f"downloaded FX data for {pair} has no close column")
        if isinstance(fx, pd.DataFrame):
            fx = fx.iloc[:, 0]
    else:
        column = "Close" if "Close" in raw.columns else "Adj Close"
        if column not in raw.columns:
            raise ValueError(f"downloaded FX data for {pair} has no close column")
        fx = raw[column]

    fx = fx.dropna()
    if fx.empty:
        raise ValueError(f"empty FX index after dropping missing values for {pair}")
    fx.name = pair
    return fx


def convert_to_eur(
    prices: pd.DataFrame,
    currencies: dict[str, str],
    eurusd: pd.Series,
) -> pd.DataFrame:
    """Convertit les colonnes en devise étrangère vers l'EUR.

    Args:
        prices: prix en devise locale (sortie download_prices).
        currencies: mapping ticker -> "EUR" | "USD".
        eurusd: taux EURUSD aligné sur l'index des prix.

    Returns:
        PricesEUR : toutes colonnes en EUR (price_eur = price_usd / eurusd).

    Hypothèse: le risque de change est ainsi incorporé aux rendements EUR.
    """
    missing = [ticker for ticker in prices.columns if ticker not in currencies]
    if missing:
        raise ValueError(f"missing currencies for tickers: {missing}")
    unsupported = {
        ticker: currencies[ticker]
        for ticker in prices.columns
        if currencies[ticker] not in {"EUR", "USD"}
    }
    if unsupported:
        raise ValueError(f"unsupported currencies: {unsupported}")
    if prices.empty:
        raise ValueError("prices must not be empty")
    if prices.isna().any().any():
        raise ValueError("prices contains missing values")

    fx = eurusd.sort_index().reindex(prices.index).ffill()
    converted = prices.copy()
    usd_tickers = [ticker for ticker in converted.columns if currencies[ticker] == "USD"]

    if usd_tickers:
        valid = fx.notna()
        converted = converted.loc[valid]
        fx = fx.loc[valid]
        if converted.empty:
            raise ValueError("empty price index after aligning prices with FX")
        if not converted.index.equals(fx.index):
            raise ValueError("prices and FX indexes are not aligned")
        converted.loc[:, usd_tickers] = converted.loc[:, usd_tickers].div(fx, axis=0)

    converted = converted.dropna(how="any")
    if converted.empty:
        raise ValueError("empty price index after EUR conversion")
    if converted.isna().any().any():
        raise ValueError("EUR prices contains missing values")
    return converted


def to_log_returns(prices_eur: pd.DataFrame) -> pd.DataFrame:
    """Calcule les log-returns journaliers r_t = ln(P_t / P_{t-1}).

    La première ligne (NaN) est supprimée. Returns = DataFrame (cf. contrats).
    """
    if prices_eur.empty:
        raise ValueError("prices_eur must not be empty")
    if prices_eur.isna().any().any():
        raise ValueError("prices_eur contains missing values")

    returns = np.log(prices_eur / prices_eur.shift(1)).iloc[1:]
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("empty returns index after computing log-returns")
    return returns


def _slice_dates(frame, start: str, end: str | None):
    """Restreint à [start, end[ (end exclusif, convention yfinance)."""
    mask = frame.index >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.index < pd.Timestamp(end)
    return frame.loc[mask]


def _read_cache(
    prices_path: Path,
    fx_path: Path,
    tickers: list[str],
    start: str,
    end: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Lit prix + FX depuis le snapshot CSV et restreint à la période demandée."""
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(
            f"cached prices at {prices_path} lack tickers {missing}; "
            "delete the cache files to force a fresh download"
        )
    fx_frame = pd.read_csv(fx_path, index_col=0, parse_dates=True)
    eurusd = fx_frame.iloc[:, 0]

    prices = _slice_dates(prices.loc[:, tickers], start, end).dropna(how="any")
    eurusd = _slice_dates(eurusd, start, end).dropna()
    if prices.empty:
        raise ValueError(f"cached prices at {prices_path} have no rows in [{start}, {end})")
    return prices, eurusd


def load_returns(
    tickers: list[str],
    currencies: dict[str, str],
    start: str,
    end: str | None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline data complet (façade pour cli.py).

    Enchaîne download_prices -> download_fx -> convert_to_eur -> to_log_returns.
    Si `cache_dir` est fourni (cache write-through, SPEC.md B1.4) : lit
    `prices.csv` / `eurusd.csv` s'ils existent (snapshot figé, restreint à
    [start, end[), sinon télécharge puis écrit ces fichiers. Le snapshot
    committé dans data/cache/ rend l'étude et les tests rejouables offline.

    Returns:
        (prices_eur, returns).
    """
    missing = [ticker for ticker in tickers if ticker not in currencies]
    if missing:
        raise ValueError(f"missing currencies for tickers: {missing}")

    if cache_dir is not None:
        cache = Path(cache_dir)
        prices_path = cache / "prices.csv"
        fx_path = cache / "eurusd.csv"
        if prices_path.is_file() and fx_path.is_file():
            prices, eurusd = _read_cache(prices_path, fx_path, tickers, start, end)
        else:
            prices = download_prices(tickers, start, end)
            eurusd = download_fx("EURUSD=X", start, end)
            cache.mkdir(parents=True, exist_ok=True)
            prices.to_csv(prices_path)
            eurusd.to_csv(fx_path)
    else:
        prices = download_prices(tickers, start, end)
        eurusd = download_fx("EURUSD=X", start, end)

    prices_eur = convert_to_eur(prices, currencies, eurusd)
    returns = to_log_returns(prices_eur)
    return prices_eur, returns
