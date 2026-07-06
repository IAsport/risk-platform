"""Verrouillage du résultat phare (SPEC.md B1.5) sur le snapshot committé.

Règle du projet : pas de résultat affiché sans test. Les verdicts du tableau README
(étude 2019–2021, VaR 99 %) sont vérifiés ici sur le snapshot figé
data/cache/ (2026-07-06) — offline, déterministe. Le rolling GARCH complet
tourne en ~2 s : inclus en CI (amendement B1.8 #8, seuil 5 min).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskplatform.backtest import (
    christoffersen_cc,
    christoffersen_independence,
    count_exceptions,
    kupiec_pof,
)
from riskplatform.config import load_config
from riskplatform.data import load_returns
from riskplatform.portfolio import portfolio_returns
from riskplatform.var import rolling_var, rolling_var_conditional

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.99
STUDY = slice("2019-01-01", "2021-12-31")


@pytest.fixture(scope="module")
def pnl() -> pd.Series:
    config = load_config(REPO / "config" / "portfolio.yaml")
    tickers = list(config.portfolio.weights.index)
    _, returns = load_returns(
        tickers,
        config.portfolio.currencies,
        config.start,
        config.end,
        cache_dir=REPO / "data" / "cache",
    )
    return portfolio_returns(returns, config.portfolio.weights)


def _backtest(pnl: pd.Series, var_series: pd.Series) -> dict:
    var_study = var_series.loc[STUDY]
    exceptions = count_exceptions(pnl.loc[var_study.index], var_study)
    return {
        "kupiec": kupiec_pof(exceptions, alpha=ALPHA),
        "independence": christoffersen_independence(exceptions),
        "cc": christoffersen_cc(exceptions, alpha=ALPHA),
    }


def test_parametric_unconditional_fails_coverage_and_clustering(pnl: pd.Series):
    """La VaR gaussienne inconditionnelle (250 j) : trop d'exceptions ET en grappes."""
    result = _backtest(pnl, rolling_var(pnl, "parametric", alpha=ALPHA, window=250))

    assert result["kupiec"]["n_exceptions"] == 17  # ~7.5 attendues sur 751 jours
    assert result["kupiec"]["reject"] is True
    assert result["independence"]["reject"] is True  # grappe février-avril 2020
    assert result["cc"]["reject"] is True


def test_ewma_fixes_clustering_but_not_coverage(pnl: pd.Series):
    """EWMA : l'indépendance passe (plus de grappes), la couverture reste
    rejetée — les résidus standardisés sont leptokurtiques, |z_0.01|=2.33 est
    trop petit. C'est le pont vers la brique 2 (innovations Student-t)."""
    result = _backtest(pnl, rolling_var_conditional(pnl, "ewma", alpha=ALPHA))

    assert result["independence"]["reject"] is False
    assert result["kupiec"]["n_exceptions"] == 21
    assert result["kupiec"]["reject"] is True


def test_garch_fixes_clustering_but_not_coverage(pnl: pd.Series):
    """GARCH(1,1) refit 20 j / fenêtre 1000 j : même diagnostic que l'EWMA."""
    result = _backtest(
        pnl, rolling_var_conditional(pnl, "garch", alpha=ALPHA, window=1000, refit_every=20)
    )

    assert result["independence"]["reject"] is False
    assert result["kupiec"]["n_exceptions"] == 21
    assert result["kupiec"]["reject"] is True


def test_conditional_var_reacts_to_march_2020_unconditional_does_not(pnl: pd.Series):
    """Le cœur du résultat phare : en mars 2020 la VaR conditionnelle est
    multipliée par >2.5 vs la paramétrique 250 j, qui reste quasi aveugle."""
    parametric = rolling_var(pnl, "parametric", alpha=ALPHA, window=250)
    ewma = rolling_var_conditional(pnl, "ewma", alpha=ALPHA)

    max_parametric = parametric.loc["2020-03"].max()
    max_ewma = ewma.loc["2020-03"].max()

    assert max_ewma > 2.5 * max_parametric


def test_historical_unconditional_passes_on_this_window(pnl: pd.Series):
    """Contrepoint honnête : l'historique 250 j passe sur 2019-2021 (10 exc.),
    mais avec 10 exceptions le test d'indépendance a peu de puissance — et
    l'historique échoue à Kupiec sur l'échantillon complet 2014-2026 (README)."""
    result = _backtest(pnl, rolling_var(pnl, "historical", alpha=ALPHA, window=250))

    assert result["kupiec"]["n_exceptions"] == 10
    assert result["cc"]["reject"] is False


# ---------- Brique 2 : Student-t (verdicts constatés sur snapshot, B2.9 #8) ----------


def test_student_t_reduces_exceptions_but_kupiec_still_rejects(pnl: pd.Series):
    """Résultat B2, documenté tel quel : le quantile t (nu estimé ~6-8 par MLE
    sur les résidus) monte la VaR d'environ +10 % et réduit les exceptions
    (21 -> 19 EWMA, 21 -> 18 GARCH sur 2019-2021), mais Kupiec rejette encore
    à 99 % : le MLE de nu calibre TOUTE la densité, pas la queue à 1 %, et les
    sauts jour-1 depuis un régime calme restent sous-couverts par tout filtre
    à retard d'un jour. L'indépendance, elle, tient toujours."""
    ewma_t = _backtest(
        pnl,
        rolling_var_conditional(
            pnl, "ewma", alpha=ALPHA, window=1000, refit_every=20, dist="student"
        ),
    )
    garch_t = _backtest(
        pnl,
        rolling_var_conditional(
            pnl, "garch", alpha=ALPHA, window=1000, refit_every=20, dist="student"
        ),
    )

    assert ewma_t["kupiec"]["n_exceptions"] == 19  # 21 en normal
    assert garch_t["kupiec"]["n_exceptions"] == 18  # 21 en normal
    assert ewma_t["kupiec"]["reject"] is True
    assert garch_t["kupiec"]["reject"] is True
    assert ewma_t["independence"]["reject"] is False
    assert garch_t["independence"]["reject"] is False


def test_student_t_halves_the_kupiec_gap_on_full_sample(pnl: pd.Series):
    """Sur toutes les dates prévues (~2018-2026) : 48 exceptions en normal,
    37 en Student-t (20.8 attendues) — l'écart à l'attendu est presque divisé
    par deux, la p-value Kupiec passe de <1e-6 à ~1e-3."""
    normal = rolling_var_conditional(
        pnl, "ewma", alpha=ALPHA, window=1000, refit_every=20, dist="normal"
    )
    student = rolling_var_conditional(
        pnl, "ewma", alpha=ALPHA, window=1000, refit_every=20, dist="student"
    )
    # Même périmètre de dates pour comparer (le mode normal démarre plus tôt).
    normal = normal.loc[student.index.intersection(normal.index)]

    exc_normal = count_exceptions(pnl.loc[normal.index], normal)
    exc_student = count_exceptions(pnl.loc[student.index], student)
    kupiec_normal = kupiec_pof(exc_normal, alpha=ALPHA)
    kupiec_student = kupiec_pof(exc_student, alpha=ALPHA)

    assert kupiec_student["n_exceptions"] < kupiec_normal["n_exceptions"]
    assert kupiec_student["p_value"] > 100 * kupiec_normal["p_value"]


def test_acerbi_szekely_rejects_conditional_es_on_crisis_window(pnl: pd.Series):
    """ES 97.5 % conditionnel (sigma EWMA, nu estimé hors échantillon sur les
    résidus <= 2018) : AS Z2 rejette sur 2019-2021 — les pertes de queue
    réalisées dépassent l'ES annoncé même sous t. Verdict honnête, verrouillé ;
    la variante t reste moins mauvaise que la normale (z plus petit)."""
    from riskplatform.backtest import acerbi_szekely_z2
    from riskplatform.distributions import fit_student_df
    from riskplatform.es import es_conditional
    from riskplatform.var import var_conditional
    from riskplatform.volatility import ewma_variance

    sigma = np.sqrt(ewma_variance(pnl))
    residuals = pnl.loc[sigma.index] / sigma
    nu = fit_student_df(residuals.loc[:"2018-12-31"])
    sigma_study = sigma.loc[STUDY]

    results = {}
    for df_h0 in (None, nu):
        var_series = var_conditional(sigma_study, alpha=0.975, df=df_h0)
        es_series = es_conditional(sigma_study, alpha=0.975, df=df_h0)
        results[df_h0] = acerbi_szekely_z2(
            pnl.loc[sigma_study.index], var_series, es_series, sigma_study,
            alpha=0.975, df=df_h0,
        )

    assert 8.0 < nu < 10.0  # ~9.1 sur le snapshot
    assert results[None]["reject"] is True
    assert results[nu]["reject"] is True
    assert results[nu]["z_stat"] < results[None]["z_stat"]  # t moins mauvais
