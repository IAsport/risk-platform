"""Colle UI partagée du dashboard Streamlit (SPEC.md B4.2).

Tout le calcul vient de `riskplatform` (pipeline, var, backtest) ; ce module
n'apporte que le cache Streamlit, le bandeau commun (date du snapshot,
amendement de validation (a)) et le registre des modèles de backtest. Les
chemins sont résolus depuis la racine du repo (parent de `app/`) pour
fonctionner en local, sous AppTest et sur Streamlit Community Cloud.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from riskplatform import backtest
from riskplatform.config import load_config
from riskplatform.pipeline import RiskAnalysis, run_analysis
from riskplatform.var import rolling_var, rolling_var_conditional, var_monte_carlo_student

ROOT = Path(__file__).resolve().parents[1]

# Les 4 modèles du récit B3 (décision B4.9 #9), mêmes paramètres que l'étude
# etude_stress_traffic_light : fenêtre 1000 j, refit 20 j, innovations Student-t.
BACKTEST_MODELS = ("Historique 250 j", "Paramétrique 250 j", "EWMA-t", "GARCH-t")
_CONDITIONAL_WINDOW = 1000
_REFIT_EVERY = 20
_TRAFFIC_LIGHT_WINDOW = 250


@st.cache_data(show_spinner="Pipeline complet sur le snapshot (une seule fois)...")
def load_analysis() -> RiskAnalysis:
    """RiskAnalysis sur le snapshot committé — offline, décision B4.9 #2."""
    config = load_config(ROOT / "config" / "portfolio.yaml")
    return run_analysis(config, cache_dir=str(ROOT / "data" / "cache"))


@st.cache_data(show_spinner="Backtest du modèle sélectionné...")
def backtest_model(model: str, alpha: float) -> dict:
    """VaR rolling + Kupiec/Christoffersen/traffic light du modèle choisi."""
    pnl = load_analysis().portfolio_returns
    if model == "Historique 250 j":
        var_series = rolling_var(pnl, method="historical", alpha=alpha)
    elif model == "Paramétrique 250 j":
        var_series = rolling_var(pnl, method="parametric", alpha=alpha)
    elif model == "EWMA-t":
        var_series = rolling_var_conditional(
            pnl, "ewma", alpha=alpha, window=_CONDITIONAL_WINDOW,
            refit_every=_REFIT_EVERY, dist="student",
        )
    elif model == "GARCH-t":
        var_series = rolling_var_conditional(
            pnl, "garch", alpha=alpha, window=_CONDITIONAL_WINDOW,
            refit_every=_REFIT_EVERY, dist="student",
        )
    else:
        raise ValueError(f"unknown backtest model: {model!r}")

    exceptions = backtest.count_exceptions(pnl, var_series)
    result = {
        "var_series": var_series,
        "exceptions": exceptions,
        "kupiec": backtest.kupiec_pof(exceptions, alpha=alpha),
        "christoffersen": backtest.christoffersen_cc(exceptions, alpha=alpha),
        "traffic_light": None,
        "rolling_traffic_light": None,
    }
    if len(exceptions) >= _TRAFFIC_LIGHT_WINDOW:
        result["traffic_light"] = backtest.traffic_light(
            exceptions, alpha=alpha, window=_TRAFFIC_LIGHT_WINDOW
        )
        result["rolling_traffic_light"] = backtest.rolling_traffic_light(
            exceptions, alpha=alpha, window=_TRAFFIC_LIGHT_WINDOW
        )
    return result


@st.cache_data(show_spinner="Monte Carlo Student-t (50 000 tirages)...")
def mc_student_var(alpha: float) -> float:
    """VaR MC Student-t plein échantillon (nu par MLE, SPEC.md B2.2), 1 j, fraction."""
    analysis = load_analysis()
    return var_monte_carlo_student(
        analysis.returns, analysis.config.portfolio.weights, alpha=alpha
    )


def page_header(title: str) -> RiskAnalysis:
    """Titre + bandeau as-of commun ; retourne l'analyse.

    Le bandeau affiche la date du snapshot sur CHAQUE page (amendement (a)) :
    l'app lit des données committées, pas un flux live. `set_page_config`
    appartient au routeur (streamlit_app.py).
    """
    analysis = load_analysis()
    config = analysis.config
    st.title(title)
    st.caption(
        f"Portefeuille « {config.name} » ({len(config.portfolio.weights)} titres, "
        f"notional {config.portfolio.notional_eur:,.0f} EUR) · "
        f"snapshot offline du **{analysis.as_of.date().isoformat()}** · "
        f"{len(analysis.portfolio_returns)} rendements journaliers depuis {config.start}. "
        f"Rafraîchissement des données : `riskplatform --no-cache` (CLI), pas depuis l'app."
    )
    return analysis
