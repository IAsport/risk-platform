"""Tests du dashboard Streamlit via AppTest (SPEC.md B4.5, décision B4.9 #6).

Headless, sans navigateur ni réseau : les pages tournent sur le snapshot
committé (data/cache). Smoke test par page + deux interactions clés.
Le calcul lourd est mis en cache par _shared (st.cache_data), partagé entre
les tests d'un même process.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = [
    APP_DIR / "views" / "portefeuille.py",
    APP_DIR / "views" / "var_es.py",
    APP_DIR / "views" / "backtesting.py",
    APP_DIR / "views" / "stress_tests.py",
    APP_DIR / "views" / "methodologie.py",
]
_TIMEOUT = 600


def _run_page(path: Path) -> AppTest:
    app_test = AppTest.from_file(str(path), default_timeout=_TIMEOUT)
    app_test.run()
    return app_test


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_page_runs_without_exception(page):
    app_test = _run_page(page)

    assert not app_test.exception
    assert len(app_test.title) == 1


def test_router_entrypoint_renders_default_view():
    # `streamlit run app/streamlit_app.py` : le routeur st.navigation sert la
    # vue Portefeuille par défaut.
    app_test = _run_page(APP_DIR / "streamlit_app.py")

    assert not app_test.exception
    assert app_test.title[0].value == "Portefeuille & données"


def test_every_page_shows_snapshot_date(page_paths=PAGES):
    # Amendement de validation (a) : la date du snapshot est affichée en tête
    # de chaque page (bandeau page_header).
    for page in page_paths:
        app_test = _run_page(page)
        banner = app_test.caption[0].value
        assert "snapshot offline du" in banner
        assert re.search(r"\d{4}-\d{2}-\d{2}", banner)


def test_var_page_alpha_switch_moves_the_var():
    app_test = _run_page(PAGES[1])

    def historical_var() -> float:
        return float(app_test.metric[0].value.replace("%", ""))

    var_99 = historical_var()
    app_test.radio[0].set_value("95 %").run()
    var_95 = historical_var()

    assert not app_test.exception
    assert var_99 > var_95 > 0  # le quantile 99 % domine strictement le 95 %


def test_backtesting_page_model_switch_changes_series():
    app_test = _run_page(PAGES[2])

    n_obs_historical = str(app_test.metric[0].value)
    app_test.selectbox[0].select("EWMA-t").run()
    n_obs_ewma = str(app_test.metric[0].value)

    assert not app_test.exception
    # Historique démarre après 250 j, EWMA-t après 30 + 1000 j : les séries
    # de backtest n'ont pas la même longueur.
    assert n_obs_historical != n_obs_ewma


def test_methodology_page_tells_the_five_acts():
    app_test = _run_page(PAGES[4])

    headers = [header.value for header in app_test.header]
    assert len(headers) == 5
    assert headers[0].startswith("Acte 1")
    assert headers[4].startswith("Acte 5")
