"""Backtest d'Expected Shortfall — statistique Z2 d'Acerbi-Székely (2014).

    Z2 = sum_t [ L_t · I(L_t > VaR_t) ] / [ T · (1-alpha) · ES_t ]  -  1

E[Z2] = 0 sous H0 (le modèle prédit correctement fréquence ET sévérité des
pertes de queue) ; Z2 > 0 signifie des pertes de queue plus lourdes que l'ES
annoncé (sous-estimation). Test conjoint fréquence × sévérité, contrairement
à Kupiec qui ne compte que les violations.

p-value par SIMULATION sous H0 : on rejoue B trajectoires de pertes depuis
les prévisions du modèle (L_t = sigma_t · eps_t, eps ~ N(0,1) ou t_df
standardisée), on recalcule Z2 sur chacune, p = fraction des Z2 simulés >=
Z2 observé (unilatéral : on cherche la sous-estimation). Choix justifié
SPEC.md B2.6 (alternative Emmer-Kratz-Tasche rejetée : indirecte).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskplatform.backtest._common import validate_alpha, validate_series


def acerbi_szekely_z2(
    realized_returns: pd.Series,
    var_series: pd.Series,
    es_series: pd.Series,
    sigma_series: pd.Series,
    alpha: float = 0.99,
    df: float | None = None,
    n_sims: int = 5_000,
    seed: int | None = 42,
) -> dict:
    """Test Z2 d'Acerbi-Székely sur des prévisions (VaR_t, ES_t, sigma_t).

    Les quatre séries sont alignées sur l'intersection de leurs dates.
    df : degré de liberté des innovations sous H0 (None = normales) — doit
    être la loi que le modèle backtesté utilise réellement.

    Returns:
        dict: z_stat, p_value, reject (seuil 5 %), n_exceptions, n_obs.
    """
    validate_alpha(alpha)
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    realized = validate_series(realized_returns, "realized_returns").astype(float)
    var = validate_series(var_series, "var_series").astype(float)
    es = validate_series(es_series, "es_series").astype(float)
    sigma = validate_series(sigma_series, "sigma_series").astype(float)

    common = realized.index.intersection(var.index).intersection(es.index)
    common = common.intersection(sigma.index)
    if common.empty:
        raise ValueError("input series have an empty date intersection")
    realized, var, es, sigma = (s.loc[common] for s in (realized, var, es, sigma))
    if (es <= 0).any():
        raise ValueError("es_series must be strictly positive")
    if (sigma <= 0).any():
        raise ValueError("sigma_series must be strictly positive")

    n_obs = len(common)
    tail_probability = 1.0 - alpha

    def z2_statistic(losses: np.ndarray) -> np.ndarray:
        """Z2 par trajectoire ; losses de forme (..., T)."""
        indicators = losses > var.to_numpy()
        contributions = losses * indicators / es.to_numpy()
        return contributions.sum(axis=-1) / (n_obs * tail_probability) - 1.0

    observed_losses = (-realized).to_numpy()
    z_stat = float(z2_statistic(observed_losses))
    n_exceptions = int((observed_losses > var.to_numpy()).sum())

    rng = np.random.default_rng(seed)
    if df is None:
        eps = rng.standard_normal(size=(n_sims, n_obs))
    else:
        if df <= 2.0:
            raise ValueError("df must be > 2 (finite variance required)")
        eps = rng.standard_t(df, size=(n_sims, n_obs)) * np.sqrt((df - 2.0) / df)
    simulated_losses = sigma.to_numpy() * eps  # L = -r = -sigma·eps ~ sigma·eps (symétrie)
    z_sims = z2_statistic(simulated_losses)
    p_value = float(np.mean(z_sims >= z_stat))

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "n_exceptions": n_exceptions,
        "n_obs": n_obs,
    }
