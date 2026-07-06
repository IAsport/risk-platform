"""Student-t standardisée (variance 1) : quantile et MLE du degré de liberté.

Si T ~ t_nu (nu > 2), Var(T) = nu/(nu-2). La version STANDARDISÉE est
eps = T · sqrt((nu-2)/nu), de variance 1 : elle se branche sur sigma_t sans
changer l'échelle (VaR_t = |q_std|·sigma_t). Quand nu → ∞, on retrouve la
normale. Densité standardisée : f_eps(x) = f_nu(x/s)/s avec s = sqrt((nu-2)/nu).

Utilisé par var/monte_carlo.py, var/conditional.py et es.py (SPEC.md B2.1).
Source : McNeil-Frey-Embrechts, Quantitative Risk Management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from riskplatform._validation import validate_series

_MIN_OBS = 50


def _std_scale(df: float) -> float:
    """s = sqrt((df-2)/df) : facteur ramenant la t_df à variance 1."""
    if df <= 2.0:
        raise ValueError("df must be > 2 (finite variance required)")
    return float(np.sqrt((df - 2.0) / df))


def student_quantile_std(p: float, df: float) -> float:
    """Quantile de la t standardisée : t⁻¹_df(p) · sqrt((df-2)/df)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in ]0, 1[")
    return float(student_t.ppf(p, df) * _std_scale(df))


def fit_student_df(
    standardized: pd.Series,
    bounds: tuple[float, float] = (2.05, 100.0),
) -> float:
    """MLE du degré de liberté d'une t standardisée sur une série de variance ~1.

    Une valeur estimée en butée haute signifie « données ≈ gaussiennes »
    (ce n'est pas une erreur). ValueError si série invalide (< 50 points, NaN).
    """
    clean = validate_series(standardized, "standardized")
    if len(clean) < _MIN_OBS:
        raise ValueError(f"standardized must contain at least {_MIN_OBS} points")
    if not 2.0 < bounds[0] < bounds[1]:
        raise ValueError("bounds must satisfy 2 < lower < upper")

    values = clean.to_numpy()

    def negative_loglik(df: float) -> float:
        scale = _std_scale(df)
        return -float(np.sum(student_t.logpdf(values / scale, df) - np.log(scale)))

    result = minimize_scalar(
        negative_loglik, bounds=bounds, method="bounded", options={"xatol": 1e-4}
    )
    return float(result.x)
