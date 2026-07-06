"""Entrypoint du dashboard (SPEC.md B4.2) — routeur `st.navigation`.

`streamlit run app/streamlit_app.py` (nom attendu par Streamlit Community
Cloud). Les cinq vues vivent dans `app/views/` ; chaque vue reste un script
autonome testable individuellement par AppTest.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="risk-platform", layout="wide")

st.navigation(
    [
        st.Page("views/portefeuille.py", title="Portefeuille & données", default=True),
        st.Page("views/var_es.py", title="VaR & Expected Shortfall", url_path="var-es"),
        st.Page("views/backtesting.py", title="Backtesting interactif", url_path="backtesting"),
        st.Page("views/stress_tests.py", title="Stress tests", url_path="stress-tests"),
        st.Page("views/methodologie.py", title="Méthodologie", url_path="methodologie"),
    ]
).run()
