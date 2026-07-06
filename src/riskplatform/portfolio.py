"""Module portfolio — définition du portefeuille et agrégation des rendements.

Responsabilité : représenter le portefeuille (poids, devises, notional) et
agréger les log-returns titres en rendement de portefeuille + matrice de
covariance. Voir ARCHITECTURE.md §2 et SPEC.md §1.2-1.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Portfolio:
    """Définition immuable d'un portefeuille.

    Attributs:
        weights: pd.Series index=tickers, somme=1.0.
        currencies: dict ticker -> "EUR" | "USD".
        notional_eur: valeur de marché totale en EUR.
    """

    weights: pd.Series
    currencies: dict[str, str]
    notional_eur: float = 1_000_000.0


def make_equal_weight(
    tickers: list[str],
    currencies: dict[str, str],
    notional_eur: float = 1_000_000.0,
) -> Portfolio:
    """Construit un portefeuille équipondéré (1/N par titre)."""
    if not tickers:
        raise ValueError("tickers must not be empty")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique")

    missing = [ticker for ticker in tickers if ticker not in currencies]
    if missing:
        raise ValueError(f"missing currencies for tickers: {missing}")

    weight = 1.0 / len(tickers)
    weights = pd.Series(weight, index=tickers, dtype=float)
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")
    return Portfolio(weights=weights, currencies=currencies, notional_eur=notional_eur)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Agrège les log-returns titres en log-return de portefeuille.

    r_p,t ≈ sum_i w_i * r_i,t (poids constants, rebalancement quotidien).
    Limite (additivité inter-actifs) documentée dans SPEC.md §1.2.

    Returns:
        PortfolioReturns (pd.Series indexée par date).
    """
    if returns.empty:
        raise ValueError("returns must not be empty")
    if weights.empty:
        raise ValueError("weights must not be empty")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    missing = [ticker for ticker in weights.index if ticker not in returns.columns]
    if missing:
        raise ValueError(f"missing returns for tickers: {missing}")

    aligned = returns.loc[:, weights.index]
    if aligned.isna().any().any():
        raise ValueError("returns contains missing values")
    return aligned.mul(weights, axis=1).sum(axis=1)


def covariance_matrix(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Matrice de variance-covariance des log-returns (base journalière).

    Sert à var.py pour sigma_p^2 = w' Σ w (effet de diversification).
    """
    if returns.empty:
        raise ValueError("returns must not be empty")
    if returns.isna().any().any():
        raise ValueError("returns contains missing values")

    if weights is not None:
        if weights.empty:
            raise ValueError("weights must not be empty")
        if abs(float(weights.sum()) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        missing = [ticker for ticker in weights.index if ticker not in returns.columns]
        if missing:
            raise ValueError(f"missing returns for tickers: {missing}")
        returns = returns.loc[:, weights.index]

    cov = returns.cov()
    if cov.empty:
        raise ValueError("empty covariance matrix")
    if cov.isna().any().any():
        raise ValueError("covariance matrix contains missing values")
    return cov
