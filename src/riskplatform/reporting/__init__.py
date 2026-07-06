"""Package reporting — tables de synthèse, graphes de backtest, rendu fichiers."""

from riskplatform.reporting.daily_report import render_daily_report
from riskplatform.reporting.report import (
    plot_return_distribution,
    plot_stress_pnl,
    plot_traffic_light,
    plot_var_backtest,
    render_report,
    render_stress_report,
    summary_table,
)

__all__ = [
    "plot_return_distribution",
    "render_daily_report",
    "plot_stress_pnl",
    "plot_traffic_light",
    "plot_var_backtest",
    "render_report",
    "render_stress_report",
    "summary_table",
]
