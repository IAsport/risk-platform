"""Chargement et validation de la configuration YAML du portefeuille de référence.

Le YAML (`config/portfolio.yaml`) est la source de vérité du portefeuille :
positions (ticker, devise, poids optionnels — équipondéré par défaut),
benchmark HORS poids (série de marché non investie, cf. SPEC.md B0.3),
période, niveaux de confiance, horizon et notional. Le reste du code ne voit
jamais le YAML : `load_config` renvoie un `RunConfig` gelé.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from riskplatform.portfolio import Portfolio, make_equal_weight

_SUPPORTED_CURRENCIES = {"EUR", "USD"}
_WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RunConfig:
    """Configuration d'exécution du pipeline (immutable).

    benchmark_ticker/benchmark_currency : série de marché de référence chargée
    hors portefeuille (poids nul) ; inutilisée par le calcul de VaR en brique 0,
    réservée aux stress tests (brique 3) et aux comparaisons.
    """

    name: str
    portfolio: Portfolio
    start: str
    end: str | None
    alphas: tuple[float, ...]
    horizon_days: int
    benchmark_ticker: str | None = None
    benchmark_currency: str | None = None


def _validate_date(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO date (YYYY-MM-DD): {value!r}") from exc
    return value


def _build_portfolio(raw: dict) -> Portfolio:
    positions = raw.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("config must define a non-empty 'positions' list")

    tickers: list[str] = []
    currencies: dict[str, str] = {}
    weights: dict[str, float] = {}
    for position in positions:
        if not isinstance(position, dict) or "ticker" not in position or "currency" not in position:
            raise ValueError(f"each position needs 'ticker' and 'currency': {position!r}")
        ticker = str(position["ticker"])
        currency = str(position["currency"])
        if currency not in _SUPPORTED_CURRENCIES:
            raise ValueError(f"unsupported currency {currency!r} for {ticker} (EUR|USD)")
        tickers.append(ticker)
        currencies[ticker] = currency
        if "weight" in position:
            weights[ticker] = float(position["weight"])

    if len(set(tickers)) != len(tickers):
        raise ValueError("position tickers must be unique")

    notional = float(raw.get("notional_eur", 1_000_000.0))
    if notional <= 0:
        raise ValueError("notional_eur must be positive")

    if not weights:
        return make_equal_weight(tickers, currencies, notional_eur=notional)
    if len(weights) != len(tickers):
        missing = sorted(set(tickers) - set(weights))
        raise ValueError(f"either all positions define 'weight' or none; missing: {missing}")

    series = pd.Series(weights, dtype=float).loc[tickers]
    if (series < 0).any():
        raise ValueError("weights must be non-negative")
    if abs(float(series.sum()) - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"weights must sum to 1.0, got {float(series.sum())!r}")
    return Portfolio(weights=series, currencies=currencies, notional_eur=notional)


def _build_benchmark(raw: dict) -> tuple[str | None, str | None]:
    benchmark = raw.get("benchmark")
    if benchmark is None:
        return None, None
    if not isinstance(benchmark, dict) or "ticker" not in benchmark or "currency" not in benchmark:
        raise ValueError(f"benchmark needs 'ticker' and 'currency': {benchmark!r}")
    currency = str(benchmark["currency"])
    if currency not in _SUPPORTED_CURRENCIES:
        raise ValueError(f"unsupported benchmark currency {currency!r} (EUR|USD)")
    return str(benchmark["ticker"]), currency


def load_config(path: str | Path) -> RunConfig:
    """Charge et valide le YAML de configuration. Lève ValueError si invalide."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a YAML mapping")

    start = _validate_date(raw.get("start"), "start")
    end_raw = raw.get("end")
    end = None if end_raw is None else _validate_date(end_raw, "end")
    if end is not None and end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    alphas_raw = raw.get("alphas", [0.95, 0.99])
    if not isinstance(alphas_raw, list) or not alphas_raw:
        raise ValueError("alphas must be a non-empty list")
    alphas = tuple(float(alpha) for alpha in alphas_raw)
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError(f"alphas must be in ]0, 1[: {alphas}")

    horizon_days = int(raw.get("horizon_days", 1))
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    benchmark_ticker, benchmark_currency = _build_benchmark(raw)

    return RunConfig(
        name=str(raw.get("name", "reference")),
        portfolio=_build_portfolio(raw),
        start=start,
        end=end,
        alphas=alphas,
        horizon_days=horizon_days,
        benchmark_ticker=benchmark_ticker,
        benchmark_currency=benchmark_currency,
    )
