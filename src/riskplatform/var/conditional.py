"""VaR conditionnelle : sigma_t (EWMA/GARCH) au lieu du sigma inconditionnel.

    VaR_t(alpha) = |q_{1-alpha}| · sigma_t · notional     (mu = 0)

où q est le quantile de la loi standardisée des innovations : normale
(q = z_{1-alpha}) ou Student-t standardisée de degré nu (SPEC.md B2.3) —
même sigma_t, seul le quantile change. Le GARCH reste estimé en gaussien
(QMLE, convergent pour omega/alpha/beta sous innovations non gaussiennes) ;
nu est estimé dans un second temps par MLE sur les résidus standardisés
r_t/sigma_t de la fenêtre d'estimation.

La volatilité conditionnelle est modélisée sur la série de portefeuille
AGRÉGÉE (univariée, SPEC.md B1.3). Convention perte positive (var/__init__).
Datation : sigma²_t et nu_t = info <= t-1 (volatility/__init__.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskplatform._validation import QUANTILE_METHOD, validate_alpha, validate_series
from riskplatform.distributions import fit_student_df, student_quantile_std
from riskplatform.volatility.ewma import ewma_variance
from riskplatform.volatility.garch import fit_garch, garch_variance

_EWMA_INIT_WINDOW = 30


def _quantile_abs(alpha: float, df: float | None) -> float:
    """|quantile 1-alpha| de la loi standardisée : normale si df=None, t sinon."""
    if df is None:
        return float(-norm.ppf(1.0 - alpha))
    return float(-student_quantile_std(1.0 - alpha, df))


def var_conditional(
    sigma: float | pd.Series,
    alpha: float = 0.99,
    notional: float = 1.0,
    df: float | None = None,
) -> float | pd.Series:
    """VaR paramétrique conditionnelle |q_{1-alpha}|·sigma·notional.

    df=None : innovations normales ; df>2 : Student-t standardisée.
    Accepte un scalaire (une date) ou une série sigma_t (même index en sortie).
    """
    validate_alpha(alpha)
    quantile = _quantile_abs(alpha, df)
    if isinstance(sigma, pd.Series):
        clean = validate_series(sigma, "sigma")
        if (clean < 0).any():
            raise ValueError("sigma must be >= 0")
        return (quantile * clean * notional).rename("var_conditional")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return float(quantile * sigma * notional)


def var_conditional_monte_carlo(
    sigma_t: float,
    alpha: float = 0.99,
    notional: float = 1.0,
    n_sims: int = 50_000,
    seed: int | None = 42,
    df: float | None = None,
) -> float:
    """VaR conditionnelle par simulation : r = sigma_t·eps.

    eps ~ N(0,1) si df=None, sinon t_df standardisée (variance 1). Converge
    vers var_conditional (mêmes hypothèses).
    """
    validate_alpha(alpha)
    if sigma_t < 0:
        raise ValueError("sigma_t must be >= 0")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    rng = np.random.default_rng(seed)
    if df is None:
        eps = rng.standard_normal(n_sims)
    else:
        if df <= 2.0:
            raise ValueError("df must be > 2 (finite variance required)")
        eps = rng.standard_t(df, size=n_sims) * np.sqrt((df - 2.0) / df)
    simulated_losses = -sigma_t * eps * notional
    value = np.quantile(simulated_losses, alpha, method=QUANTILE_METHOD)
    return max(0.0, float(value))


def rolling_var_conditional(
    pnl_returns: pd.Series,
    vol_method: str,
    alpha: float = 0.99,
    window: int = 1000,
    refit_every: int = 20,
    lam: float = 0.94,
    notional: float = 1.0,
    dist: str = "normal",
    df: float | None = None,
) -> pd.Series:
    """VaR conditionnelle out-of-sample pour le backtest.

    vol_method :
    - "ewma"  : filtre pur (lam fixé). En dist="normal", la série produite
      commence après l'amorçage EWMA (30 points) et `window`/`refit_every`
      sont ignorés. En dist="student", nu est réestimé tous les `refit_every`
      jours sur les `window` derniers résidus standardisés r/sigma (la série
      produite commence donc à 30+window).
    - "garch" : réestimation MLE sur fenêtre glissante `window` tous les
      `refit_every` jours ; en dist="student", nu est réestimé au même rythme
      sur les résidus standardisés de la fenêtre d'estimation (QMLE 2 étapes).

    dist : "normal" ou "student". df : fixe nu (sinon MLE par refit).
    Aucune date ne voit r_t dans son sigma²_t ni dans son nu_t.
    """
    validate_alpha(alpha)
    clean = validate_series(pnl_returns, "pnl_returns")
    if vol_method not in {"ewma", "garch"}:
        raise ValueError("vol_method must be 'ewma' or 'garch'")
    if dist not in {"normal", "student"}:
        raise ValueError("dist must be 'normal' or 'student'")
    if dist == "normal" and df is not None:
        raise ValueError("df is only valid with dist='student'")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")

    if vol_method == "ewma":
        sigma = np.sqrt(ewma_variance(clean, lam=lam, init_window=_EWMA_INIT_WINDOW))
        if dist == "normal":
            result = var_conditional(sigma, alpha=alpha, notional=notional)
            assert isinstance(result, pd.Series)
            return result.rename("var_ewma")
        # dist="student" : nu réestimé sur fenêtre glissante de résidus r/sigma.
        residuals = clean.loc[sigma.index] / sigma
        if window >= len(residuals):
            raise ValueError("window must be < number of EWMA residuals")
        values: list[float] = []
        quantile = 0.0
        for k in range(window, len(residuals)):
            if (k - window) % refit_every == 0:
                nu = df if df is not None else fit_student_df(residuals.iloc[k - window : k])
                quantile = _quantile_abs(alpha, nu)
            values.append(quantile * float(sigma.iloc[k]) * notional)
        return pd.Series(values, index=sigma.index[window:], name="var_ewma_student")

    if window >= len(clean):
        raise ValueError("window must be < len(pnl_returns)")

    series_values = clean.to_numpy()
    variances: list[float] = []
    quantiles: list[float] = []
    params = None
    quantile = 0.0
    sigma2_previous = 0.0

    for i in range(window, len(series_values)):
        if (i - window) % refit_every == 0:
            estimation_sample = clean.iloc[i - window : i]
            params = fit_garch(estimation_sample)
            filtered = garch_variance(estimation_sample, params)
            sigma2_previous = float(filtered.iloc[-1])
            if dist == "student":
                residuals = estimation_sample / np.sqrt(filtered)
                nu = df if df is not None else fit_student_df(residuals)
                quantile = _quantile_abs(alpha, nu)
            else:
                quantile = _quantile_abs(alpha, None)
        assert params is not None
        sigma2_forecast = (
            params.omega + params.alpha * series_values[i - 1] ** 2 + params.beta * sigma2_previous
        )
        variances.append(sigma2_forecast)
        quantiles.append(quantile)
        sigma2_previous = sigma2_forecast

    var_values = np.array(quantiles) * np.sqrt(np.array(variances)) * notional
    name = "var_garch" if dist == "normal" else "var_garch_student"
    return pd.Series(var_values, index=clean.index[window:], name=name)
