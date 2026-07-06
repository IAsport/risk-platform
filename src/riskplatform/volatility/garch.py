"""GARCH(1,1) — estimation par maximum de vraisemblance gaussienne (à la main).

Modèle (moyenne nulle, SPEC.md B1.2) :

    r_t = sigma_t · eps_t,   eps_t ~ N(0, 1) i.i.d.
    sigma²_t = omega + alpha · r²_{t-1} + beta · sigma²_{t-1}

Contraintes : omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
(stationnarité). Variance de long terme : sigma²_LT = omega / (1-alpha-beta).

Log-vraisemblance gaussienne (maximisée) :

    l(omega, alpha, beta) = -1/2 · sum_t [ ln(2·pi) + ln(sigma²_t) + r²_t / sigma²_t ]

avec la récursion filtrée initialisée à la variance empirique de l'échantillon.
Prévision à horizon h (mean-reversion géométrique) :

    sigma²_{t+h} = sigma²_LT + (alpha+beta)^(h-1) · (sigma²_{t+1} - sigma²_LT)

Sources : Bollerslev (1986) ; Hull ; McNeil-Frey-Embrechts §4. La lib `arch`
n'est utilisée que comme oracle de validation dans les tests (SPEC.md B1.6),
jamais importée par le code de production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from riskplatform._validation import validate_series

_LOG_2PI = float(np.log(2.0 * np.pi))
_PERSISTENCE_CAP = 1.0 - 1e-6
_START_ALPHA = 0.05
_START_BETA = 0.90


@dataclass(frozen=True)
class GarchParams:
    """Paramètres GARCH(1,1) estimés (immutables)."""

    omega: float
    alpha: float
    beta: float
    loglik: float
    n_obs: int

    @property
    def persistence(self) -> float:
        """alpha + beta — proche de 1 = chocs très persistants (IGARCH à la limite)."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """sigma²_LT = omega / (1 - alpha - beta), défini car alpha + beta < 1."""
        return self.omega / (1.0 - self.alpha - self.beta)


def _filter_variance(
    values: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    sigma2_init: float,
) -> np.ndarray:
    """Récursion sigma²_t : sigma2[t] est la prévision pour t (info <= t-1)."""
    sigma2 = np.empty(len(values))
    sigma2[0] = sigma2_init
    for t in range(1, len(values)):
        sigma2[t] = omega + alpha * values[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _negative_loglik(params: np.ndarray, values: np.ndarray, sigma2_init: float) -> float:
    omega, alpha, beta = params
    sigma2 = _filter_variance(values, omega, alpha, beta, sigma2_init)
    if np.any(sigma2 <= 0.0) or not np.all(np.isfinite(sigma2)):
        return np.inf
    return 0.5 * float(np.sum(_LOG_2PI + np.log(sigma2) + values**2 / sigma2))


def fit_garch(returns: pd.Series, min_obs: int = 250) -> GarchParams:
    """Estime GARCH(1,1) par MLE gaussien (SLSQP sous contraintes).

    Point de départ par variance targeting : alpha0=0.05, beta0=0.90,
    omega0 = s²·(1-alpha0-beta0) avec s² la variance empirique — démarrage
    dans la région plausible. Le targeting ne sert QU'AU point initial, les
    trois paramètres sont estimés librement.

    Raises:
        ValueError: série invalide (courte, NaN, constante).
        RuntimeError: optimiseur non convergé (jamais de paramètres silencieux).
    """
    clean = validate_series(returns, "returns")
    if len(clean) < min_obs:
        raise ValueError(f"returns must contain at least min_obs = {min_obs} points")

    values = clean.to_numpy()
    sample_variance = float(np.var(values, ddof=1))
    if sample_variance <= 0.0:
        raise ValueError("returns are constant: GARCH likelihood is degenerate")

    # Standardisation interne (conditionnement de l'optimiseur) : sur r/s les
    # trois paramètres sont d'échelle comparable (omega ~ 1-alpha-beta au lieu
    # de ~1e-6). alpha et beta sont invariants d'échelle ; omega = omega_std·s².
    scale = float(np.sqrt(sample_variance))
    standardized = values / scale

    start = np.array([1.0 - _START_ALPHA - _START_BETA, _START_ALPHA, _START_BETA])
    bounds = [(1e-12, None), (0.0, _PERSISTENCE_CAP), (0.0, _PERSISTENCE_CAP)]
    constraints = [{"type": "ineq", "fun": lambda p: _PERSISTENCE_CAP - p[1] - p[2]}]

    result = minimize(
        _negative_loglik,
        start,
        args=(standardized, 1.0),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"GARCH MLE did not converge: {result.message}")

    omega_std, alpha, beta = (float(v) for v in result.x)
    # Log-vraisemblance en unités d'origine : l_r = l_x - T·ln(s) (jacobien).
    loglik = -float(result.fun) - len(values) * float(np.log(scale))
    return GarchParams(
        omega=omega_std * sample_variance,
        alpha=alpha,
        beta=beta,
        loglik=loglik,
        n_obs=len(values),
    )


def garch_variance(returns: pd.Series, params: GarchParams) -> pd.Series:
    """Filtre sigma²_t sur toute la série (prévision pour t, info <= t-1).

    Initialisation à la variance empirique de la série fournie (convention
    d'estimation standard ; en usage rolling, la série est l'échantillon
    d'estimation, antérieur aux dates prévues).
    """
    clean = validate_series(returns, "returns")
    if len(clean) < 2:
        raise ValueError("returns must contain at least two observations")
    values = clean.to_numpy()
    sigma2 = _filter_variance(
        values, params.omega, params.alpha, params.beta, float(np.var(values, ddof=1))
    )
    return pd.Series(sigma2, index=clean.index, name="garch_variance")


def forecast_variance(params: GarchParams, sigma2_next: float, horizon: int) -> np.ndarray:
    """Prévisions [sigma²_{t+1}, ..., sigma²_{t+h}] par mean-reversion.

    sigma²_{t+h} = sigma²_LT + (alpha+beta)^(h-1) · (sigma²_{t+1} - sigma²_LT).
    La variance cumulée sur h jours est la somme du vecteur retourné (remplace
    la règle racine-du-temps, valable seulement si sigma²_{t+1} = sigma²_LT).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if sigma2_next < 0:
        raise ValueError("sigma2_next must be >= 0")
    steps = np.arange(1, horizon + 1)
    long_run = params.long_run_variance
    return long_run + params.persistence ** (steps - 1) * (sigma2_next - long_run)
