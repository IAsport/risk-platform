"""Page 2 — VaR & Expected Shortfall par méthode (SPEC.md B4.2).

Table des méthodes au niveau choisi, histogramme des rendements avec seuils,
zoom queue gauche normal vs Student-t (le message de la brique 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from _shared import backtest_model, mc_student_var, page_header  # noqa: E402

from riskplatform.es import expected_shortfall  # noqa: E402
from riskplatform.reporting import plot_return_distribution  # noqa: E402
from riskplatform.var import scale_var  # noqa: E402

analysis = page_header("VaR & Expected Shortfall")
config = analysis.config
notional = config.portfolio.notional_eur
pnl = analysis.portfolio_returns

alpha_label = st.radio("Niveau de confiance", ["95 %", "99 %"], index=1, horizontal=True)
alpha = 0.95 if alpha_label == "95 %" else 0.99
horizon = st.slider("Horizon (jours ouvrés, échelle √t)", 1, 10, 1)


def _to_1d(value: float) -> float:
    """var_results est à l'horizon de la config ; retour à 1 j avant re-scaling."""
    return value / np.sqrt(config.horizon_days)


rows = {row["method"]: row for row in analysis.var_results if row["alpha"] == alpha}
var_mc_t = mc_student_var(alpha)
var_ewma_t = float(backtest_model("EWMA-t", alpha)["var_series"].iloc[-1])
var_garch_t = float(backtest_model("GARCH-t", alpha)["var_series"].iloc[-1])
es_hist = _to_1d(rows["historical"]["es"])
es_frtb = expected_shortfall(pnl, alpha=0.975)

table = pd.DataFrame(
    [
        ("Historique", "plein échantillon, non paramétrique", _to_1d(rows["historical"]["var"])),
        ("Paramétrique normale", "sigma inconditionnel", _to_1d(rows["parametric"]["var"])),
        ("Monte Carlo normal", "50 000 tirages, Cholesky", _to_1d(rows["monte_carlo"]["var"])),
        ("Monte Carlo Student-t", "nu par MLE, mélange partagé", var_mc_t),
        ("EWMA-t (conditionnelle)", "sigma_t du dernier jour du snapshot", var_ewma_t),
        ("GARCH-t (conditionnelle)", "sigma_t du dernier jour du snapshot", var_garch_t),
    ],
    columns=["Méthode", "Hypothèses", "VaR 1 j"],
)
table[f"VaR {horizon} j"] = table["VaR 1 j"].apply(lambda value: scale_var(value, horizon))
table["EUR"] = table[f"VaR {horizon} j"] * notional

var_hist_h = scale_var(_to_1d(rows["historical"]["var"]), horizon)
left, middle, right = st.columns(3)
left.metric(f"VaR historique {alpha_label} ({horizon} j)", f"{var_hist_h:.2%}")
middle.metric(f"ES historique {alpha_label} ({horizon} j)", f"{scale_var(es_hist, horizon):.2%}")
right.metric("ES 97,5 % (référence FRTB, 1 j)", f"{es_frtb:.2%}")

st.dataframe(
    table.style.format({"VaR 1 j": "{:.2%}", f"VaR {horizon} j": "{:.2%}", "EUR": "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Les VaR conditionnelles dépendent du régime de volatilité du moment (fin "
    "de snapshot calme → VaR basse) là où les méthodes plein échantillon moyennent "
    "12 ans ; l'échelle √t est une approximation documentée (SPEC.md §4)."
)

st.subheader("Distribution des rendements et seuils (1 j)")
markers = {
    f"VaR hist {alpha_label}": _to_1d(rows["historical"]["var"]),
    f"VaR param {alpha_label}": _to_1d(rows["parametric"]["var"]),
    f"VaR MC-t {alpha_label}": var_mc_t,
    f"ES hist {alpha_label}": es_hist,
}
st.pyplot(plot_return_distribution(pnl, markers))

st.subheader("Zoom queue gauche — normale vs Student-t (message B2)")
tail_fig = plot_return_distribution(
    pnl,
    {
        f"VaR param (normale) {alpha_label}": _to_1d(rows["parametric"]["var"]),
        f"VaR MC Student-t {alpha_label}": var_mc_t,
    },
)
tail_fig.axes[0].set_xlim(float(pnl.min()) * 1.05, -0.5 * _to_1d(rows["parametric"]["var"]))
tail_fig.axes[0].set_ylim(0, 6)
st.pyplot(tail_fig)
st.caption(
    "La normale sous-estime la queue : le quantile Student-t (nu ≈ 6,5 estimé "
    "par MLE sur les résidus) est plus à gauche, surtout à 99 %. C'est ce qui divise par ~2 "
    "l'écart de couverture au backtest (brique 2)."
)
