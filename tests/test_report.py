from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from riskplatform.reporting import (
    plot_return_distribution,
    plot_stress_pnl,
    plot_traffic_light,
    plot_var_backtest,
    render_report,
    render_stress_report,
    summary_table,
)
from riskplatform.stress import StressSuite


def _var_results() -> list[dict]:
    return [
        {"method": "parametric", "alpha": 0.99, "horizon_days": 10, "var": 0.20, "es": 0.25},
        {"method": "historical", "alpha": 0.99, "horizon_days": 1, "var": 0.10, "es": 0.12},
        {"method": "historical", "alpha": 0.95, "horizon_days": 1, "var": 0.07, "es": 0.09},
    ]


def _backtest_series() -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=4)
    realized_returns = pd.Series([-0.02, -0.05, 0.01, -0.08], index=dates)
    var_series = pd.Series([0.03, 0.04, 0.02, 0.06], index=dates)
    exceptions = pd.Series([0, 1, 0, 1], index=dates)
    return realized_returns, var_series, exceptions


def test_summary_table_has_expected_columns_and_preserves_values():
    table = summary_table(_var_results())

    assert list(table.columns) == ["method", "alpha", "horizon_days", "var", "es", "es_method"]
    assert table.loc[0, "method"] == "historical"
    assert table.loc[0, "alpha"] == 0.95
    assert table.loc[0, "var"] == 0.07
    assert table.loc[0, "es"] == 0.09
    assert table.loc[0, "es_method"] == "historical"


def test_summary_table_sorted_by_method_and_alpha():
    table = summary_table(_var_results())

    assert list(table["method"]) == ["historical", "historical", "parametric"]
    assert list(table["alpha"]) == [0.95, 0.99, 0.99]


def test_plot_var_backtest_returns_figure_saves_png_and_marks_exceptions(tmp_path):
    realized_returns, var_series, exceptions = _backtest_series()
    out_path = tmp_path / "backtest.png"

    fig = plot_var_backtest(realized_returns, var_series, exceptions, out_path=str(out_path))

    assert isinstance(fig, Figure)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    scatter = fig.axes[0].collections[0]
    assert len(scatter.get_offsets()) == int(exceptions.sum())
    plt.close(fig)


def test_render_report_writes_expected_files(tmp_path):
    realized_returns, var_series, exceptions = _backtest_series()
    backtest_results = {
        "historical": {
            "n_obs": 4,
            "n_exceptions": 2,
            "expected": 0.04,
            "lr_stat": 1.23,
            "p_value": 0.27,
            "reject": False,
            "realized_returns": realized_returns,
            "var_series": var_series,
            "exceptions": exceptions,
        }
    }

    render_report(_var_results(), backtest_results, out_dir=str(tmp_path))

    expected_files = [
        "var_summary.csv",
        "var_summary.md",
        "backtest_summary.csv",
        "backtest_summary.md",
        "historical_backtest.png",
    ]
    for filename in expected_files:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    var_summary = pd.read_csv(tmp_path / "var_summary.csv")
    assert "es_method" in var_summary.columns
    assert set(var_summary["es_method"]) == {"historical"}


def _stress_suite() -> StressSuite:
    scenarios = pd.Index(["COVID-19", "Uniforme -20 %"], name="scenario")
    pnl_table = pd.DataFrame(
        {
            "kind": ["historical", "price"],
            "loss_eur": [150_000.0, 200_000.0],
            "pct_notional": [0.15, 0.20],
            "ratio_var": [6.0, 8.0],
            "ratio_capital": [0.63, 0.84],
        },
        index=scenarios,
    )
    pnl_by_position = pd.DataFrame(
        {"AAA": [-100_000.0, -100_000.0], "BBB": [-50_000.0, -100_000.0]}, index=scenarios
    )
    risk_table = pd.DataFrame(
        {
            "var_base": [25_000.0],
            "var_stressed": [50_000.0],
            "es_stressed": [57_000.0],
            "ratio": [2.0],
        },
        index=pd.Index(["Volatilites x2"], name="scenario"),
    )
    return StressSuite(
        pnl_table=pnl_table,
        pnl_by_position=pnl_by_position,
        risk_table=risk_table,
        worst="Uniforme -20 %",
        var_ref=25_000.0,
        capital_ref=25_000.0 * 3.0 * 10.0**0.5,
    )


def test_plot_stress_pnl_returns_figure_with_reference_lines(tmp_path):
    suite = _stress_suite()
    out_path = tmp_path / "stress_pnl.png"

    fig = plot_stress_pnl(suite.pnl_table, suite.var_ref, suite.capital_ref, str(out_path))

    assert isinstance(fig, Figure)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert len(fig.axes[0].patches) == len(suite.pnl_table)  # une barre par scénario
    assert len(fig.axes[0].lines) == 2  # VaR + proxy capital
    plt.close(fig)


def test_render_stress_report_writes_expected_files(tmp_path):
    render_stress_report(_stress_suite(), out_dir=str(tmp_path))

    for filename in [
        "stress_tests.csv",
        "stress_risk.csv",
        "stress_by_position.csv",
        "stress_tests.md",
        "stress_pnl.png",
    ]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    stress_table = pd.read_csv(tmp_path / "stress_tests.csv")
    assert list(stress_table["scenario"]) == ["COVID-19", "Uniforme -20 %"]
    markdown = (tmp_path / "stress_tests.md").read_text(encoding="utf-8")
    assert "Uniforme -20 %" in markdown
    assert "Chocs de paramètres" in markdown


def _rolling_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    counts = [2, 4, 5, 9, 10, 12]
    zones = ["green", "green", "yellow", "yellow", "red", "red"]
    return pd.DataFrame({"n_exceptions": counts, "zone": zones}, index=dates)


def test_plot_return_distribution_draws_one_line_per_marker(tmp_path):
    returns = pd.Series([0.01, -0.02, 0.005, -0.03, 0.015, -0.01] * 50)
    out_path = tmp_path / "distribution.png"

    fig = plot_return_distribution(
        returns,
        {"VaR 99 %": 0.025, "ES 97.5 %": 0.030},
        out_path=str(out_path),
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2  # une ligne verticale par marker
    assert out_path.exists() and out_path.stat().st_size > 0
    plt.close("all")


def test_plot_return_distribution_rejects_bad_inputs():
    with pytest.raises(ValueError, match="empty"):
        plot_return_distribution(pd.Series(dtype=float), {})
    with pytest.raises(ValueError, match="missing"):
        plot_return_distribution(pd.Series([0.01, float("nan")]), {})
    with pytest.raises(ValueError, match="finite"):
        plot_return_distribution(pd.Series([0.01, -0.02]), {"VaR": np.inf})
    plt.close("all")


def test_plot_traffic_light_draws_bands_and_saves(tmp_path):
    out_path = tmp_path / "traffic.png"

    fig = plot_traffic_light(_rolling_frame(), alpha=0.99, window=250, out_path=str(out_path))

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 3 bandes (axhspan -> patches) au-dessus de l'histogramme absent, et la
    # courbe du compte d'exceptions.
    assert len(ax.lines) == 1
    assert len(ax.patches) >= 3
    # Bornes canoniques (4, 9) : le plafond couvre la zone rouge observée.
    assert ax.get_ylim()[1] >= 12 + 2
    assert out_path.exists() and out_path.stat().st_size > 0
    plt.close("all")


def test_plot_traffic_light_rejects_bad_inputs():
    with pytest.raises(ValueError, match="empty"):
        plot_traffic_light(pd.DataFrame())
    with pytest.raises(ValueError, match="missing columns"):
        plot_traffic_light(pd.DataFrame({"n_exceptions": [1, 2]}))
    plt.close("all")
