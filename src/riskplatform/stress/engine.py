"""Moteur de stress testing : application des scénarios au portefeuille courant.

Conventions (SPEC.md B3.1) : chocs = rendements arithmétiques, chocs
instantanés sur le portefeuille d'aujourd'hui (poids w, notional N), sans
rebalancement ni réaction de gestion. P&L par position = N·w_i·R_i (signé,
perte < 0) ; perte stressée L = -P&L (positive, convention VaR). Les pertes
sont rapportées à la VaR 99 % 1 j et au proxy de capital IMA 3·sqrt(10)·VaR
(multiplicateur plancher bâlois, horizon 10 j en racine de t — ordre de
grandeur documenté, pas un calcul de capital réglementaire).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskplatform._validation import validate_returns_frame, validate_series
from riskplatform.es import es_conditional
from riskplatform.portfolio import covariance_matrix, portfolio_returns
from riskplatform.stress.scenarios import (
    DEFAULT_SCENARIOS,
    HistoricalWindow,
    IndexShock,
    PriceShock,
    RiskParamShock,
    Scenario,
)
from riskplatform.var.conditional import var_conditional
from riskplatform.var.historical import var_historical

_CAPITAL_MULTIPLIER = 3.0 * np.sqrt(10.0)


@dataclass(frozen=True)
class StressResult:
    """Résultat d'un scénario de P&L (historique, choc de prix, choc d'indice)."""

    name: str
    kind: str  # "historical" | "price" | "index"
    pnl_by_position: pd.Series  # EUR, signé (perte < 0)
    pnl_total: float
    loss: float  # -pnl_total (perte positive, convention VaR)


@dataclass(frozen=True)
class StressedRiskResult:
    """Résultat d'un choc de paramètres : VaR/ES paramétriques stressées."""

    name: str
    var_base: float
    var_stressed: float
    es_base: float
    es_stressed: float
    ratio: float  # var_stressed / var_base


@dataclass(frozen=True)
class StressSuite:
    """Sortie agrégée de run_stress_suite (les deux panneaux, SPEC.md B3.5)."""

    pnl_table: pd.DataFrame  # par scénario : loss_eur, pct_notional, ratio_var, ratio_capital
    pnl_by_position: pd.DataFrame  # scénarios x tickers, EUR signé
    risk_table: pd.DataFrame  # par scénario : var_base, var_stressed, es_stressed, ratio
    worst: str | None  # pire scénario de P&L (None si aucun scénario de P&L)
    var_ref: float  # référence VaR des ratios (SPEC.md B3.1)
    capital_ref: float  # proxy capital 3·sqrt(10)·var_ref


def _validate_weights(weights: pd.Series) -> pd.Series:
    if weights.empty:
        raise ValueError("weights must not be empty")
    clean = weights.astype(float)
    if abs(float(clean.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")
    return clean


def _pnl_result(
    name: str,
    kind: str,
    shocks: pd.Series,
    weights: pd.Series,
    notional: float,
) -> StressResult:
    pnl = (notional * weights * shocks.loc[weights.index]).rename("pnl_eur")
    total = float(pnl.sum())
    return StressResult(
        name=name, kind=kind, pnl_by_position=pnl, pnl_total=total, loss=-total
    )


def worst_window(portfolio_pnl: pd.Series, horizon: int = 20) -> HistoricalWindow:
    """Pire fenêtre glissante de `horizon` jours (rendement cumulé minimal).

    Extraite des données pour PROUVER où est le pire épisode au lieu de le
    supposer (SPEC.md B3.2). ValueError si la série est plus courte que horizon.
    """
    clean = validate_series(portfolio_pnl, "portfolio_pnl")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(clean) < horizon:
        raise ValueError("portfolio_pnl must have at least `horizon` observations")

    cumulative = clean.rolling(horizon).sum()
    end = cumulative.idxmin()
    end_position = int(clean.index.get_loc(end))
    start = clean.index[end_position - horizon + 1]
    start_iso = pd.Timestamp(start).date().isoformat()
    end_iso = pd.Timestamp(end).date().isoformat()
    return HistoricalWindow(
        name=f"Pire fenetre {horizon} j ({start_iso} -> {end_iso})",
        start=start_iso,
        end=end_iso,
    )


def replay_window(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: HistoricalWindow,
    notional: float = 1.0,
) -> StressResult:
    """Rejoue une fenêtre historique : R_i = exp(sum r_i) - 1, buy-and-hold.

    Conversion arithmétique EXACTE : l'approximation log sum(w·r) surestime
    la perte de plusieurs points sur un choc de -38 % (SPEC.md B3.10 #1).
    """
    frame = validate_returns_frame(returns)
    clean_weights = _validate_weights(weights)
    missing = [ticker for ticker in clean_weights.index if ticker not in frame.columns]
    if missing:
        raise ValueError(f"missing returns for tickers: {missing}")

    window = frame.loc[scenario.start : scenario.end, clean_weights.index]
    if window.empty:
        raise ValueError(
            f"scenario {scenario.name!r}: no return dates in [{scenario.start}, {scenario.end}]"
        )
    shocks = pd.Series(np.expm1(window.sum(axis=0)), index=clean_weights.index)
    return _pnl_result(scenario.name, "historical", shocks, clean_weights, notional)


def apply_price_shock(
    weights: pd.Series,
    scenario: PriceShock,
    notional: float = 1.0,
) -> StressResult:
    """Choc de prix instantané : uniforme, ou par ticker (absents -> 0)."""
    clean_weights = _validate_weights(weights)
    if isinstance(scenario.shock, float | int):
        shocks = pd.Series(float(scenario.shock), index=clean_weights.index)
    else:
        unknown = [ticker for ticker in scenario.shock if ticker not in clean_weights.index]
        if unknown:
            raise ValueError(
                f"scenario {scenario.name!r}: shocked tickers not in portfolio: {unknown}"
            )
        shocks = pd.Series(0.0, index=clean_weights.index)
        for ticker, value in scenario.shock.items():
            shocks.loc[ticker] = float(value)
    return _pnl_result(scenario.name, "price", shocks, clean_weights, notional)


def estimate_betas(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """Bêtas OLS pleine période : beta_i = Cov(r_i, r_b) / Var(r_b).

    Estimés sur l'intersection des dates. Limite (SPEC.md B3.3, amendement
    #5) : sous-estiment la propagation en crise (bêtas conditionnels hors
    périmètre).
    """
    frame = validate_returns_frame(returns)
    benchmark = validate_series(benchmark_returns, "benchmark_returns")

    common = frame.index.intersection(benchmark.index)
    if len(common) < 2:
        raise ValueError("returns and benchmark_returns need >= 2 common dates")
    aligned = frame.loc[common]
    bench = benchmark.loc[common]

    bench_var = float(bench.var(ddof=1))
    if bench_var < 1e-18:
        raise ValueError("benchmark_returns variance is zero (constant series)")
    covariances = aligned.apply(lambda column: column.cov(bench))
    return (covariances / bench_var).rename("beta")


def apply_index_shock(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    weights: pd.Series,
    scenario: IndexShock,
    notional: float = 1.0,
) -> StressResult:
    """Choc d'indice propagé par bêtas : R_i = beta_i · index_return."""
    if benchmark_returns is None:
        raise ValueError(
            f"scenario {scenario.name!r} requires benchmark_returns (index shock via betas)"
        )
    clean_weights = _validate_weights(weights)
    betas = estimate_betas(returns.loc[:, clean_weights.index], benchmark_returns)
    shocks = betas * scenario.index_return
    return _pnl_result(scenario.name, "index", shocks, clean_weights, notional)


def stressed_var_parametric(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: RiskParamShock,
    alpha: float = 0.99,
    notional: float = 1.0,
) -> StressedRiskResult:
    """VaR/ES paramétriques sous paramètres stressés (SPEC.md B3.4).

    Sigma = D·R·D décomposée ; R_s = (1-s)·R + s·J (PSD pour tout s), D_k =
    diag(k·sigma_i) ; sigma_p* = sqrt(w' D_k R_s D_k w), puis VaR/ES fermées
    normales sur sigma_p*.
    """
    cov = covariance_matrix(returns, weights)
    aligned_weights = weights.loc[cov.index].astype(float)
    cov_array = cov.to_numpy()
    sigma_vec = np.sqrt(np.diag(cov_array))
    if np.any(sigma_vec < 1e-12):
        raise ValueError("returns contain a constant column (zero variance)")
    corr = cov_array / np.outer(sigma_vec, sigma_vec)

    s = scenario.corr_shift
    k = scenario.vol_multiplier
    corr_stressed = (1.0 - s) * corr + s * np.ones_like(corr)
    sigma_stressed_vec = k * sigma_vec
    cov_stressed = np.outer(sigma_stressed_vec, sigma_stressed_vec) * corr_stressed

    w = aligned_weights.to_numpy()
    sigma_p_base = float(np.sqrt(w @ cov_array @ w))
    sigma_p_stressed = float(np.sqrt(w @ cov_stressed @ w))

    var_base = var_conditional(sigma_p_base, alpha=alpha, notional=notional)
    var_stressed = var_conditional(sigma_p_stressed, alpha=alpha, notional=notional)
    es_base = es_conditional(sigma_p_base, alpha=alpha, notional=notional)
    es_stressed = es_conditional(sigma_p_stressed, alpha=alpha, notional=notional)
    assert isinstance(var_base, float) and isinstance(es_base, float)
    assert isinstance(var_stressed, float) and isinstance(es_stressed, float)

    return StressedRiskResult(
        name=scenario.name,
        var_base=var_base,
        var_stressed=var_stressed,
        es_base=es_base,
        es_stressed=es_stressed,
        ratio=var_stressed / var_base,
    )


def run_stress_suite(
    returns: pd.DataFrame,
    weights: pd.Series,
    notional: float = 1.0,
    benchmark_returns: pd.Series | None = None,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    add_worst_window: bool = True,
    horizon: int = 20,
    alpha: float = 0.99,
    var_ref: float | None = None,
) -> StressSuite:
    """Applique le catalogue de scénarios et agrège les deux panneaux.

    var_ref=None : VaR historique `alpha` plein échantillon sur r_p (référence
    « capital VaR » du PLAN §3-B3) ; ratio_capital = loss / (3·sqrt(10)·var_ref).
    IndexShock présent sans benchmark_returns -> ValueError (levée à
    l'application du scénario).
    """
    frame = validate_returns_frame(returns)
    clean_weights = _validate_weights(weights)
    portfolio_pnl = portfolio_returns(frame, clean_weights)

    if var_ref is None:
        var_ref = var_historical(portfolio_pnl, alpha=alpha, notional=notional)
    if var_ref <= 0.0:
        raise ValueError("var_ref must be positive")
    capital_ref = _CAPITAL_MULTIPLIER * var_ref

    scenario_list: list[Scenario] = list(scenarios)
    if add_worst_window:
        scenario_list.append(worst_window(portfolio_pnl, horizon=horizon))

    pnl_results: list[StressResult] = []
    risk_results: list[StressedRiskResult] = []
    for scenario in scenario_list:
        if isinstance(scenario, HistoricalWindow):
            pnl_results.append(replay_window(frame, clean_weights, scenario, notional))
        elif isinstance(scenario, PriceShock):
            pnl_results.append(apply_price_shock(clean_weights, scenario, notional))
        elif isinstance(scenario, IndexShock):
            pnl_results.append(
                apply_index_shock(frame, benchmark_returns, clean_weights, scenario, notional)
            )
        elif isinstance(scenario, RiskParamShock):
            risk_results.append(
                stressed_var_parametric(frame, clean_weights, scenario, alpha, notional)
            )
        else:
            raise ValueError(f"unsupported scenario type: {type(scenario).__name__}")

    pnl_table = pd.DataFrame(
        {
            "kind": [result.kind for result in pnl_results],
            "loss_eur": [result.loss for result in pnl_results],
            "pct_notional": [result.loss / notional for result in pnl_results],
            "ratio_var": [result.loss / var_ref for result in pnl_results],
            "ratio_capital": [result.loss / capital_ref for result in pnl_results],
        },
        index=pd.Index([result.name for result in pnl_results], name="scenario"),
    )
    pnl_by_position = pd.DataFrame(
        {result.name: result.pnl_by_position for result in pnl_results}
    ).T.rename_axis("scenario")
    risk_table = pd.DataFrame(
        {
            "var_base": [result.var_base for result in risk_results],
            "var_stressed": [result.var_stressed for result in risk_results],
            "es_stressed": [result.es_stressed for result in risk_results],
            "ratio": [result.ratio for result in risk_results],
        },
        index=pd.Index([result.name for result in risk_results], name="scenario"),
    )

    worst = str(pnl_table["loss_eur"].idxmax()) if not pnl_table.empty else None
    return StressSuite(
        pnl_table=pnl_table,
        pnl_by_position=pnl_by_position,
        risk_table=risk_table,
        worst=worst,
        var_ref=float(var_ref),
        capital_ref=float(capital_ref),
    )
