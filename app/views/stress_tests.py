"""Page 4 — Stress tests (SPEC.md B4.2).

Les deux panneaux de la StressSuite (P&L / VaR-ES stressées), graphe barres
vs VaR et proxy capital, décomposition par position du scénario choisi.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st  # noqa: E402
from _shared import page_header  # noqa: E402

from riskplatform.reporting import plot_stress_pnl  # noqa: E402

analysis = page_header("Stress tests")
suite = analysis.stress

if suite is None or suite.pnl_table.empty:
    st.warning("Suite de stress indisponible sur cet échantillon.")
    st.stop()

worst_row = suite.pnl_table.loc[suite.worst]
left, middle, right = st.columns(3)
left.metric("VaR 99 % 1 j (référence)", f"{suite.var_ref:,.0f} EUR")
middle.metric("Proxy capital 3·√10·VaR", f"{suite.capital_ref:,.0f} EUR")
right.metric(
    f"Pire scénario : {suite.worst}",
    f"{worst_row['loss_eur']:,.0f} EUR",
    delta=f"{worst_row['ratio_capital']:.2f}× le proxy capital",
    delta_color="inverse",
)

for name, reason in analysis.skipped_scenarios:
    st.caption(f"Scénario écarté : {name} ({reason}).")

st.subheader("Scénarios de P&L — replay historiques et chocs de prix")
st.dataframe(
    suite.pnl_table.style.format(
        {
            "loss_eur": "{:,.0f}",
            "pct_notional": "{:.1%}",
            "ratio_var": "{:.1f}",
            "ratio_capital": "{:.2f}",
        }
    ),
    use_container_width=True,
)
st.pyplot(plot_stress_pnl(suite.pnl_table, suite.var_ref, suite.capital_ref))

st.subheader("Chocs de paramètres — VaR/ES paramétriques stressées")
st.dataframe(
    suite.risk_table.style.format(
        {
            "var_base": "{:,.0f}",
            "var_stressed": "{:,.0f}",
            "es_stressed": "{:,.0f}",
            "ratio": "{:.2f}",
        }
    ),
    use_container_width=True,
)
st.caption(
    "Un choc de sigma/rho ne bouge aucun prix : sa sortie est une VaR stressée, "
    "pas un P&L (panneaux séparés, SPEC.md B3.4). « Corrélations → 1 » tue la "
    "diversification : sigma_p tend vers la moyenne pondérée des vols."
)

st.subheader("Décomposition par position")
scenario = st.selectbox("Scénario", list(suite.pnl_by_position.index))
by_position = suite.pnl_by_position.loc[scenario].sort_values()
st.bar_chart(by_position)
st.caption(
    "P&L signé par position (EUR, perte < 0) — hypothèse buy-and-hold sur le "
    "portefeuille courant, sans rebalancement ni réaction de gestion."
)
