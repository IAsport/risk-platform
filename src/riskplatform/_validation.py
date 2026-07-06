"""Validateurs privés partagés par les modules de calcul (var/, es).

Déplacés tels quels depuis l'ancien src/var.py lors de la migration brique 0.
"""

from __future__ import annotations

import pandas as pd

QUANTILE_METHOD = "linear"


def validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in ]0, 1[")


def validate_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    return series.astype(float)


def validate_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("returns must not be empty")
    if returns.isna().any().any():
        raise ValueError("returns contains missing values")
    return returns.astype(float)
