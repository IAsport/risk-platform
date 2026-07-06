"""Traffic light bâlois : zones verte/jaune/rouge sur les exceptions à 250 j.

Cadre : Comité de Bâle, *Supervisory framework for the use of "backtesting"
in conjunction with the internal models approach* (janvier 1996). Sur les 250
dernières observations, le nombre d'exceptions de la VaR 99 % classe le
modèle et fixe le plus-factor du multiplicateur de capital (3 + plus).

Les bornes de zone sont DÉRIVÉES de la CDF binomiale X ~ B(window, 1-alpha)
(SPEC.md B3.10 #7, pas de bornes codées en dur) :
    verte  : P(X <= k) < 0.95
    jaune  : 0.95 <= P(X <= k) < 0.9999
    rouge  : P(X <= k) >= 0.9999
À (99 %, 250 j) cela redonne les bornes canoniques 0-4 / 5-9 / >= 10 de la
table de Bâle (vérifié par test). La table du plus-factor n'est définie par
Bâle que pour cette configuration canonique — plus_factor vaut None ailleurs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from riskplatform.backtest._common import validate_alpha, validate_exceptions

_GREEN_MAX_CUM_PROB = 0.95
_YELLOW_MAX_CUM_PROB = 0.9999
_CANONICAL_ALPHA = 0.99
_CANONICAL_WINDOW = 250
# Bâle 1996, table 2 (250 jours, VaR 99 %) : plus-factor de la zone jaune.
_YELLOW_PLUS_FACTORS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_zone_bounds(alpha: float = 0.99, window: int = 250) -> tuple[int, int]:
    """(green_max, yellow_max) dérivés de la CDF binomiale.

    green_max = plus grand k tel que P(X <= k) < 0.95 (peut valoir -1 si même
    0 exception est déjà improbable sous H0) ; yellow_max = plus grand k tel
    que P(X <= k) < 0.9999. Zone rouge au-delà.
    """
    validate_alpha(alpha)
    if window < 1:
        raise ValueError("window must be >= 1")
    cdf = binom.cdf(np.arange(window + 1), window, 1.0 - alpha)
    green_max = int(np.searchsorted(cdf, _GREEN_MAX_CUM_PROB, side="left")) - 1
    yellow_max = int(np.searchsorted(cdf, _YELLOW_MAX_CUM_PROB, side="left")) - 1
    return green_max, yellow_max


def _zone(n_exceptions: int, green_max: int, yellow_max: int) -> str:
    if n_exceptions <= green_max:
        return "green"
    if n_exceptions <= yellow_max:
        return "yellow"
    return "red"


def _is_canonical(alpha: float, window: int) -> bool:
    return abs(alpha - _CANONICAL_ALPHA) < 1e-12 and window == _CANONICAL_WINDOW


def _plus_factor(n_exceptions: int, green_max: int) -> float:
    if n_exceptions <= green_max:
        return 0.0
    return _YELLOW_PLUS_FACTORS.get(n_exceptions, 1.0)


def traffic_light(exceptions: pd.Series, alpha: float = 0.99, window: int = 250) -> dict:
    """Zone bâloise sur les `window` DERNIERS points de la série d'exceptions.

    Retourne : n_obs, n_exceptions, zone, cum_prob (P(X <= x) sous H0),
    plus_factor et multiplier (3 + plus) — None hors config canonique
    (250 j, 99 %), la table de Bâle n'étant pas définie ailleurs.
    ValueError si moins de `window` observations ou série non binaire.
    """
    clean = validate_exceptions(exceptions)
    green_max, yellow_max = basel_zone_bounds(alpha, window)
    if len(clean) < window:
        raise ValueError(f"exceptions needs at least {window} observations, got {len(clean)}")

    tail = clean.iloc[-window:]
    n_exceptions = int(tail.sum())
    zone = _zone(n_exceptions, green_max, yellow_max)
    cum_prob = float(binom.cdf(n_exceptions, window, 1.0 - alpha))

    plus_factor: float | None = None
    multiplier: float | None = None
    if _is_canonical(alpha, window):
        plus_factor = _plus_factor(n_exceptions, green_max)
        multiplier = 3.0 + plus_factor

    return {
        "n_obs": window,
        "n_exceptions": n_exceptions,
        "zone": zone,
        "cum_prob": cum_prob,
        "plus_factor": plus_factor,
        "multiplier": multiplier,
    }


def rolling_traffic_light(
    exceptions: pd.Series, alpha: float = 0.99, window: int = 250
) -> pd.DataFrame:
    """Compte glissant des exceptions sur `window` jours + zone, par date.

    Alimente le graphe à bandes de l'étude (SPEC.md B3.7). Première date
    produite = la `window`-ième observation. ValueError si série trop courte.
    """
    clean = validate_exceptions(exceptions)
    green_max, yellow_max = basel_zone_bounds(alpha, window)
    if len(clean) < window:
        raise ValueError(f"exceptions needs at least {window} observations, got {len(clean)}")

    counts = clean.rolling(window).sum().dropna().astype(int)
    zones = np.select(
        [counts <= green_max, counts <= yellow_max], ["green", "yellow"], default="red"
    )
    return pd.DataFrame({"n_exceptions": counts, "zone": zones}, index=counts.index)
