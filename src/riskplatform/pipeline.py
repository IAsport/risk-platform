"""Pipeline de calcul réutilisable — la source unique de la CLI, du dashboard
Streamlit et du rapport quotidien (SPEC.md B4.1).

`run_analysis` reprend exactement les calculs historiques de la CLI :
data -> portefeuille -> VaR/ES multi-méthodes -> backtests 250 j (Kupiec,
Christoffersen, traffic light) -> suite de stress. Il est **silencieux**
(aucun print, aucune écriture fichier) : la CLI imprime, le dashboard
affiche, le rapport met en page — tous depuis le même `RiskAnalysis` gelé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import pandas as pd

from riskplatform import backtest, data, es, portfolio, stress, var
from riskplatform.config import RunConfig
from riskplatform.stress import StressSuite

TRAFFIC_LIGHT_WINDOW = 250
_WORST_WINDOW_HORIZON = 20
_STRESS_ALPHA = 0.99


@dataclass(frozen=True)
class RiskAnalysis:
    """Résultat complet d'un run (immutable) — SPEC.md B4.1.

    var_results : schéma historique de la CLI (method, alpha, horizon_days,
    var, es). backtest_results : clé "{method}_{int(alpha*100)}", valeurs
    Kupiec + cc_* + séries (realized_returns, var_series, exceptions) + tl_*
    si >= 250 points. skipped_scenarios : paires (nom, raison) des scénarios
    de stress écartés (fenêtre hors échantillon, benchmark absent).
    """

    config: RunConfig
    returns: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series | None
    var_results: list[dict]
    backtest_results: dict[str, dict]
    stress: StressSuite | None
    skipped_scenarios: tuple[tuple[str, str], ...]
    as_of: pd.Timestamp


def _raise_market_data_error(exc: Exception, tickers: list[str]) -> NoReturn:
    message = str(exc)
    faulty = [ticker for ticker in tickers if ticker in message]
    if faulty:
        raise RuntimeError(
            f"Failed to load market data for ticker(s) {faulty}: {message}"
        ) from exc
    raise RuntimeError(
        f"Failed to load market data for reference portfolio tickers {tickers}: {message}"
    ) from exc


def _var_and_backtests(
    config: RunConfig,
    returns: pd.DataFrame,
    portfolio_rets: pd.Series,
) -> tuple[list[dict], dict[str, dict]]:
    """VaR/ES 1 j remises à l'échelle + backtests rolling 250 j par alpha."""
    reference = config.portfolio
    horizon_days = config.horizon_days

    var_results: list[dict] = []
    backtest_results: dict[str, dict] = {}

    for alpha in config.alphas:
        hist_1d = var.var_historical(portfolio_rets, alpha=alpha)
        param_1d = var.var_parametric_portfolio(returns, reference.weights, alpha=alpha)
        mc_1d = var.var_monte_carlo(returns, reference.weights, alpha=alpha)
        es_1d = es.expected_shortfall(portfolio_rets, alpha=alpha)

        method_values = {
            "historical": hist_1d,
            "parametric": param_1d,
            "monte_carlo": mc_1d,
        }
        es_scaled = var.scale_var(es_1d, horizon_days)
        for method, value_1d in method_values.items():
            var_results.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "horizon_days": horizon_days,
                    "var": var.scale_var(value_1d, horizon_days),
                    "es": es_scaled,
                }
            )

        for method in ("historical", "parametric"):
            var_series = var.rolling_var(portfolio_rets, method=method, alpha=alpha)
            exceptions = backtest.count_exceptions(portfolio_rets, var_series)
            kupiec = backtest.kupiec_pof(exceptions, alpha=alpha)
            cc = backtest.christoffersen_cc(exceptions, alpha=alpha)
            key = f"{method}_{int(alpha * 100)}"
            backtest_results[key] = {
                **kupiec,
                "cc_lr_stat": cc["lr_stat"],
                "cc_p_value": cc["p_value"],
                "cc_reject": cc["reject"],
                "realized_returns": portfolio_rets,
                "var_series": var_series,
                "exceptions": exceptions,
            }

            # Traffic light bâlois sur les 250 dernières dates prévues (SPEC.md B3.6).
            if len(exceptions) >= TRAFFIC_LIGHT_WINDOW:
                light = backtest.traffic_light(
                    exceptions, alpha=alpha, window=TRAFFIC_LIGHT_WINDOW
                )
                backtest_results[key].update(
                    {
                        "tl_zone": light["zone"],
                        "tl_exceptions_250d": light["n_exceptions"],
                        "tl_plus_factor": light["plus_factor"],
                        "tl_multiplier": light["multiplier"],
                    }
                )

    return var_results, backtest_results


def _stress_analysis(
    config: RunConfig,
    returns: pd.DataFrame,
    portfolio_rets: pd.Series,
    cache_dir: str | None,
) -> tuple[pd.Series | None, StressSuite | None, tuple[tuple[str, str], ...]]:
    """Suite de stress B3 : benchmark chargé si configuré, scénarios applicables.

    Les fenêtres historiques hors échantillon et le scénario indiciel sans
    benchmark sont écartés et listés dans skipped (le moteur, lui, reste
    strict : ValueError).
    """
    benchmark_rets: pd.Series | None = None
    if config.benchmark_ticker is not None:
        assert config.benchmark_currency is not None
        _, bench_returns = data.load_returns(
            [config.benchmark_ticker],
            {config.benchmark_ticker: config.benchmark_currency},
            config.start,
            config.end,
            cache_dir=cache_dir,
        )
        benchmark_rets = bench_returns[config.benchmark_ticker]

    scenarios = []
    skipped: list[tuple[str, str]] = []
    for scenario in stress.DEFAULT_SCENARIOS:
        if isinstance(scenario, stress.IndexShock) and benchmark_rets is None:
            skipped.append((scenario.name, "no benchmark configured"))
            continue
        if isinstance(scenario, stress.HistoricalWindow):
            if returns.loc[scenario.start : scenario.end].empty:
                skipped.append((scenario.name, "window outside sample"))
                continue
        scenarios.append(scenario)

    add_worst_window = len(portfolio_rets) >= _WORST_WINDOW_HORIZON
    if not scenarios and not add_worst_window:
        return benchmark_rets, None, tuple(skipped)

    suite = stress.run_stress_suite(
        returns,
        config.portfolio.weights,
        notional=config.portfolio.notional_eur,
        benchmark_returns=benchmark_rets,
        scenarios=tuple(scenarios),
        add_worst_window=add_worst_window,
        alpha=_STRESS_ALPHA,
    )
    return benchmark_rets, suite, tuple(skipped)


def run_analysis(config: RunConfig, cache_dir: str | None = "data/cache") -> RiskAnalysis:
    """Data -> portefeuille -> VaR/ES -> backtests -> stress, sans effet de bord.

    cache_dir : cache CSV write-through (SPEC.md B1.4) ; None = téléchargement
    direct sans cache. Lève RuntimeError (données) ou ValueError (entrées),
    comme la CLI historique.
    """
    reference = config.portfolio
    tickers = list(reference.weights.index)

    try:
        _prices_eur, returns = data.load_returns(
            tickers, reference.currencies, config.start, config.end, cache_dir=cache_dir
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        _raise_market_data_error(exc, tickers)

    portfolio_rets = portfolio.portfolio_returns(returns, reference.weights)
    var_results, backtest_results = _var_and_backtests(config, returns, portfolio_rets)
    benchmark_rets, suite, skipped = _stress_analysis(
        config, returns, portfolio_rets, cache_dir=cache_dir
    )

    return RiskAnalysis(
        config=config,
        returns=returns,
        portfolio_returns=portfolio_rets,
        benchmark_returns=benchmark_rets,
        var_results=var_results,
        backtest_results=backtest_results,
        stress=suite,
        skipped_scenarios=skipped,
        as_of=pd.Timestamp(portfolio_rets.index[-1]),
    )
