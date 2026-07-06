"""Page 3 — Backtesting interactif (SPEC.md B4.2).

Les 4 modèles du récit B3, exceptions sur le graphe de pertes, p-values
Kupiec/Christoffersen, traffic light bâlois courant et rolling.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st  # noqa: E402
from _shared import BACKTEST_MODELS, backtest_model, page_header  # noqa: E402

from riskplatform.reporting import plot_traffic_light, plot_var_backtest  # noqa: E402

analysis = page_header("Backtesting interactif")
notional = analysis.config.portfolio.notional_eur

left, right = st.columns(2)
model = left.selectbox("Modèle de VaR", BACKTEST_MODELS, index=0)
alpha_label = right.radio("Niveau de confiance", ["95 %", "99 %"], index=1, horizontal=True)
alpha = 0.95 if alpha_label == "95 %" else 0.99

result = backtest_model(model, alpha)
kupiec = result["kupiec"]
christoffersen = result["christoffersen"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Points de backtest", kupiec["n_obs"])
col2.metric(
    "Exceptions",
    kupiec["n_exceptions"],
    delta=f"attendu {kupiec['expected']:.1f}",
    delta_color="off",
)
col3.metric(
    "p-value Kupiec",
    f"{kupiec['p_value']:.3f}",
    delta="REJET" if kupiec["reject"] else "OK",
    delta_color="inverse" if kupiec["reject"] else "normal",
)
col4.metric(
    "p-value Christoffersen",
    f"{christoffersen['p_value']:.3f}",
    delta="REJET" if christoffersen["reject"] else "OK",
    delta_color="inverse" if christoffersen["reject"] else "normal",
)

st.subheader("Pertes réalisées vs VaR — exceptions en rouge")
st.pyplot(
    plot_var_backtest(
        analysis.portfolio_returns.loc[result["var_series"].index],
        result["var_series"] * notional,
        result["exceptions"],
        notional=notional,
    )
)

st.subheader("Traffic light bâlois (fenêtre 250 j)")
light = result["traffic_light"]
if light is None:
    st.info("Série trop courte pour le traffic light (moins de 250 points).")
else:
    zone_labels = {"green": "VERTE", "yellow": "JAUNE", "red": "ROUGE"}
    tl1, tl2, tl3 = st.columns(3)
    tl1.metric("Zone actuelle", zone_labels[light["zone"]])
    tl2.metric("Exceptions / 250 j", light["n_exceptions"])
    multiplier = light["multiplier"]
    tl3.metric(
        "Multiplicateur de capital",
        f"{multiplier:.2f}" if multiplier is not None else "n/a (hors config canonique)",
    )
    st.pyplot(plot_traffic_light(result["rolling_traffic_light"], alpha=alpha))
    st.caption(
        "Zones dérivées de la CDF binomiale (Bâle 1996) : verte 0-4, jaune 5-9 "
        "(plus-factor 0,40 → 0,85), rouge ≥ 10 (multiplicateur 4,0) à 99 % / 250 j. "
        "Le plus-factor n'est défini par Bâle que pour cette configuration."
    )

st.caption(
    "Fil rouge du projet : la paramétrique 250 j reste en zone rouge de "
    "nov. 2018 à fév. 2021 (~483 j) ; EWMA-t et GARCH-t touchent le rouge au pic "
    "COVID mais en sortent en 5 mois au lieu de 27 — voir la page Méthodologie."
)
