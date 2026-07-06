"""Validateurs et utilitaires privés du package backtest.

Déplacés tels quels depuis l'ancien src/backtest.py lors de la migration brique 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in ]0, 1[")


def validate_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    return series


def validate_exceptions(exceptions: pd.Series) -> pd.Series:
    clean = validate_series(exceptions, "exceptions").astype(int)
    values = set(clean.unique())
    if not values.issubset({0, 1}):
        raise ValueError("exceptions must contain only 0/1 values")
    return clean


def log_term(count: int, probability: float) -> float:
    """Return count * log(probability), with 0 * log(0) defined as 0."""
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -np.inf
    return float(count * np.log(probability))
