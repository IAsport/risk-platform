"""Package volatility — volatilité conditionnelle EWMA et GARCH(1,1).

Convention de datation (SPEC.md B1.0) : sigma²_t est la PRÉVISION de variance
pour le jour t, construite avec les rendements jusqu'à t-1 inclus. Aucune
grandeur datée t ne dépend de r_t (pas de look-ahead).
"""

from riskplatform.volatility.ewma import ewma_variance, ewma_volatility
from riskplatform.volatility.garch import (
    GarchParams,
    fit_garch,
    forecast_variance,
    garch_variance,
)

__all__ = [
    "GarchParams",
    "ewma_variance",
    "ewma_volatility",
    "fit_garch",
    "forecast_variance",
    "garch_variance",
]
