"""EWMA (RiskMetrics) — variance conditionnelle à poids exponentiels.

Récursion (moyenne nulle, SPEC.md B1.1) :

    sigma²_t = lam · sigma²_{t-1} + (1 - lam) · r²_{t-1},   lam = 0.94 journalier

C'est une moyenne mobile des r² à poids (1-lam)·lam^k : le choc d'hier pèse
1-lam = 6 %, demi-vie de l'information ln(0.5)/ln(0.94) ≈ 11 jours. Cas
particulier de GARCH(1,1) avec omega=0, alpha=1-lam, beta=lam (persistance 1 :
pas de retour vers une variance de long terme).

Datation : sigma²_t est la prévision pour le jour t, construite avec les
rendements jusqu'à t-1 (cf. volatility/__init__.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform._validation import validate_series

TRADING_DAYS_PER_YEAR = 252


def ewma_variance(
    returns: pd.Series,
    lam: float = 0.94,
    init_window: int = 30,
) -> pd.Series:
    """Variance conditionnelle EWMA, prévision un pas en avant.

    Initialisation : sigma² au premier point produit = variance empirique
    (ddof=1) des `init_window` premiers rendements — démarrage stable sans
    regarder le futur. La série produite commence donc à l'indice
    `init_window` (les `init_window` premières dates servent à l'amorçage).

    Args:
        returns: log-returns journaliers (PortfolioReturns).
        lam: facteur de décroissance, dans ]0, 1[ (0.94 = RiskMetrics).
        init_window: nombre de points d'amorçage.

    Returns:
        pd.Series sigma²_t indexée par returns.index[init_window:].
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in ]0, 1[")
    if init_window < 2:
        raise ValueError("init_window must be >= 2")
    clean = validate_series(returns, "returns")
    if len(clean) < init_window + 2:
        raise ValueError(f"returns must contain at least {init_window + 2} points")

    values = clean.to_numpy()
    n_out = len(values) - init_window
    sigma2 = np.empty(n_out)
    sigma2[0] = float(np.var(values[:init_window], ddof=1))
    for i in range(1, n_out):
        previous_return = values[init_window + i - 1]
        sigma2[i] = lam * sigma2[i - 1] + (1.0 - lam) * previous_return**2
    return pd.Series(sigma2, index=clean.index[init_window:], name="ewma_variance")


def ewma_volatility(
    returns: pd.Series,
    lam: float = 0.94,
    init_window: int = 30,
    annualize: bool = False,
) -> pd.Series:
    """Volatilité conditionnelle EWMA : sqrt(sigma²_t), fois sqrt(252) si annualize."""
    variance = ewma_variance(returns, lam=lam, init_window=init_window)
    volatility = np.sqrt(variance)
    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    return volatility.rename("ewma_volatility")
