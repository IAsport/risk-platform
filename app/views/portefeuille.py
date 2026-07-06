"""Page 1 — Portefeuille & données (SPEC.md B4.2).

Positions, prix base 100, stats des rendements, corrélations. Vue par
défaut du routeur `app/streamlit_app.py`.
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
from _shared import page_header  # noqa: E402

analysis = page_header("Portefeuille & données")
config = analysis.config
weights = config.portfolio.weights

left, middle, right = st.columns(3)
portfolio_vol = float(analysis.portfolio_returns.std() * np.sqrt(252))
left.metric("Titres en portefeuille", len(weights))
middle.metric("Vol. portefeuille (annualisée)", f"{portfolio_vol:.1%}")
right.metric("Kurtosis excédentaire", f"{float(analysis.portfolio_returns.kurt()):.1f}")

st.subheader("Positions (équipondérées, converties EUR)")
positions = pd.DataFrame(
    {
        "Devise": pd.Series(config.portfolio.currencies),
        "Poids": weights,
        "Notional (EUR)": weights * config.portfolio.notional_eur,
    }
).rename_axis("Ticker")
st.dataframe(
    positions.style.format({"Poids": "{:.1%}", "Notional (EUR)": "{:,.0f}"}),
    use_container_width=True,
)
if config.benchmark_ticker is not None:
    st.caption(
        f"Benchmark **{config.benchmark_ticker}** hors poids (indice de prix non "
        f"investissable, décision B0) — sert au scénario indiciel des stress tests."
    )

st.subheader("Trajectoires base 100 (depuis les log-returns EUR)")
base_100 = pd.DataFrame(
    {"Portefeuille": 100.0 * np.exp(analysis.portfolio_returns.cumsum())}
)
if analysis.benchmark_returns is not None:
    aligned = analysis.benchmark_returns.reindex(analysis.returns.index).fillna(0.0)
    base_100[str(config.benchmark_ticker)] = 100.0 * np.exp(aligned.cumsum())
if st.checkbox("Afficher les titres individuels (échelle dominée par NVDA, ×700)"):
    base_100 = base_100.join(100.0 * np.exp(analysis.returns.cumsum()))
st.line_chart(base_100)

st.subheader("Statistiques des rendements journaliers")
stats = pd.DataFrame(
    {
        "Vol. annualisée": analysis.returns.std() * np.sqrt(252),
        "Skewness": analysis.returns.skew(),
        "Kurtosis excédentaire": analysis.returns.kurt(),
        "Pire jour": analysis.returns.min(),
    }
).rename_axis("Ticker")
st.dataframe(
    stats.style.format(
        {
            "Vol. annualisée": "{:.1%}",
            "Skewness": "{:.2f}",
            "Kurtosis excédentaire": "{:.1f}",
            "Pire jour": "{:.2%}",
        }
    ),
    use_container_width=True,
)
st.caption(
    "Kurtosis excédentaire > 0 sur tous les titres : les queues sont plus épaisses "
    "que la normale — c'est la motivation des pages VaR/ES (Student-t) et Méthodologie."
)

st.subheader("Corrélations des rendements")
corr = analysis.returns.corr()
st.dataframe(
    corr.style.background_gradient(cmap="RdYlGn_r", vmin=-1.0, vmax=1.0).format("{:.2f}"),
    use_container_width=True,
)
st.caption(
    "Le bloc US (AAPL/MSFT/NVDA) et le bloc Euronext se distinguent ; en crise, "
    "ces corrélations montent — voir le choc « corrélations → 1 » des stress tests."
)
