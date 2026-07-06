"""Rapport de risque quotidien — une page HTML autonome (SPEC.md B4.3).

Format type middle office : VaR/ES du jour, backtesting réglementaire
(traffic light, dernières exceptions datées), top risques stress, un graphe
250 jours. Le fichier est **autonome** : CSS inline, figure matplotlib
embarquée en base64 — envoyable par mail, imprimable en PDF par le
navigateur. Templating stdlib (`string.Template`), décision B4.9 #5.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskplatform import es, var
from riskplatform.pipeline import RiskAnalysis
from riskplatform.reporting.report import plot_var_backtest

_FRTB_ES_ALPHA = 0.975
_PLOT_WINDOW = 250
_TOP_POSITIONS = 3
_ZONE_COLORS = {"green": "#1a7f37", "yellow": "#b08800", "red": "#c1121f"}
_ZONE_LABELS = {"green": "VERTE", "yellow": "JAUNE", "red": "ROUGE"}

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport de risque quotidien — $name</title>
<style>
  body { font-family: "Segoe UI", Arial, sans-serif; color: #1f2328; margin: 24px auto;
         max-width: 860px; font-size: 14px; }
  h1 { font-size: 22px; border-bottom: 2px solid #1f2328; padding-bottom: 6px; }
  h2 { font-size: 16px; margin-top: 22px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: right; }
  th { background: #f6f8fa; }
  td:first-child, th:first-child { text-align: left; }
  .meta { color: #57606a; }
  .badge { color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .note { color: #57606a; font-style: italic; }
  footer { margin-top: 24px; border-top: 1px solid #d0d7de; padding-top: 8px;
           color: #57606a; font-size: 12px; }
  img { max-width: 100%; }
</style>
</head>
<body>
<h1>Rapport de risque quotidien — $name</h1>
<p class="meta">Date de marché : <strong>$as_of</strong> · Notional : $notional ·
Devise : EUR · Horizon VaR : $horizon j · $n_obs rendements journaliers</p>

<h2>1. VaR &amp; Expected Shortfall</h2>
$var_table
<p class="note">ES 97,5 % historique 1 j (référence FRTB) : $es_frtb.</p>

<h2>2. Backtesting réglementaire (fenêtre 250 j)</h2>
$backtest_section

<h2>3. Top risques — stress tests</h2>
$stress_section

<h2>4. Pertes vs VaR 99 % — 250 derniers jours</h2>
$figure_section

<footer>Méthodologie : log-returns journaliers EUR, VaR en perte positive,
backtests Kupiec/Christoffersen et traffic light bâlois (SPEC.md) —
détail interactif sur le dashboard Streamlit du projet.</footer>
</body>
</html>
"""
)


def _eur(value: float) -> str:
    return f"{value:,.0f} EUR"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _zone_badge(zone: str) -> str:
    color = _ZONE_COLORS.get(zone, "#57606a")
    return f'<span class="badge" style="background:{color}">{_ZONE_LABELS.get(zone, zone)}</span>'


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _var_section(analysis: RiskAnalysis, notional: float) -> str:
    rows = []
    ordered = sorted(analysis.var_results, key=lambda row: (row["alpha"], row["method"]))
    for row in ordered:
        rows.append(
            [
                row["method"],
                f"{row['alpha']:.0%}",
                _eur(row["var"] * notional),
                f"{row['var']:.2%}",
                _eur(row["es"] * notional),
            ]
        )
    return _html_table(
        ["Méthode", "Niveau", "VaR", "VaR (% notional)", "ES historique"], rows
    )


def _reference_backtest(analysis: RiskAnalysis) -> tuple[str, dict] | None:
    """Backtest de référence pour les exceptions et le graphe : historique au
    niveau le plus élevé, sinon le premier disponible."""
    for alpha in sorted(analysis.config.alphas, reverse=True):
        key = f"historical_{int(alpha * 100)}"
        if key in analysis.backtest_results:
            return key, analysis.backtest_results[key]
    if analysis.backtest_results:
        key = next(iter(analysis.backtest_results))
        return key, analysis.backtest_results[key]
    return None


def _backtest_section(analysis: RiskAnalysis, notional: float) -> str:
    if not analysis.backtest_results:
        return '<p class="note">Aucun backtest disponible (échantillon trop court).</p>'

    rows = []
    for alpha in analysis.config.alphas:
        for method in ("historical", "parametric"):
            result = analysis.backtest_results.get(f"{method}_{int(alpha * 100)}")
            if result is None:
                continue
            if "tl_zone" in result:
                zone = _zone_badge(result["tl_zone"])
                exceptions_250 = str(result["tl_exceptions_250d"])
                multiplier = result["tl_multiplier"]
                multiplier_label = f"{multiplier:.2f}" if multiplier is not None else "n/a"
            else:
                zone, exceptions_250, multiplier_label = "n/a (&lt; 250 pts)", "n/a", "n/a"
            rows.append(
                [
                    f"{method} {alpha:.0%}",
                    f"{result['n_exceptions']}/{result['n_obs']}",
                    f"{result['p_value']:.3f}",
                    f"{result['cc_p_value']:.3f}",
                    zone,
                    exceptions_250,
                    multiplier_label,
                ]
            )
    table = _html_table(
        [
            "Modèle",
            "Exceptions",
            "p Kupiec",
            "p Christoffersen",
            "Zone",
            "Exc./250 j",
            "Multiplicateur",
        ],
        rows,
    )

    reference = _reference_backtest(analysis)
    assert reference is not None
    key, result = reference
    exception_dates = result["exceptions"].index[result["exceptions"] == 1]
    if len(exception_dates) == 0:
        recent = f'<p class="note">Aucune exception sur l\'échantillon ({key}).</p>'
    else:
        last = exception_dates[-5:]
        losses = -result["realized_returns"].loc[last] * notional
        items = "".join(
            f"<li>{date.date().isoformat()} — perte {_eur(loss)}</li>"
            for date, loss in losses.items()
        )
        recent = (
            f"<p>Dernières exceptions ({key}, {len(exception_dates)} au total) :</p>"
            f"<ul>{items}</ul>"
        )
    return table + recent


def _stress_section(analysis: RiskAnalysis) -> str:
    suite = analysis.stress
    if suite is None or suite.pnl_table.empty:
        return '<p class="note">Suite de stress indisponible sur cet échantillon.</p>'

    worst_row = suite.pnl_table.loc[suite.worst]
    contributions = suite.pnl_by_position.loc[suite.worst].nsmallest(_TOP_POSITIONS)
    position_rows = [
        [ticker, _eur(-pnl), f"{-pnl / worst_row['loss_eur']:.0%}"]
        for ticker, pnl in contributions.items()
    ]
    return (
        f"<p>Pire scénario : <strong>{suite.worst}</strong> — perte "
        f"{_eur(worst_row['loss_eur'])} ({worst_row['pct_notional']:.1%} du notional), "
        f"soit <strong>{worst_row['ratio_var']:.1f}×</strong> la VaR 99 % 1 j et "
        f"<strong>{worst_row['ratio_capital']:.2f}×</strong> le proxy capital 3·√10·VaR.</p>"
        + _html_table(["Position", "Perte dans ce scénario", "Part"], position_rows)
    )


def _figure_section(analysis: RiskAnalysis, notional: float) -> str:
    reference = _reference_backtest(analysis)
    if reference is None:
        return '<p class="note">Pas de série de backtest à tracer.</p>'
    _key, result = reference
    index = result["var_series"].index[-_PLOT_WINDOW:]
    fig = plot_var_backtest(
        result["realized_returns"].loc[index],
        result["var_series"].loc[index] * notional,
        result["exceptions"].loc[index],
        notional=notional,
    )
    encoded = _fig_to_base64(fig)
    return f'<img alt="Pertes vs VaR" src="data:image/png;base64,{encoded}">'


def render_daily_report(
    analysis: RiskAnalysis,
    out_path: str | None = "outputs/daily_report.html",
) -> str:
    """Génère le rapport quotidien ; écrit `out_path` si fourni, retourne le HTML."""
    notional = analysis.config.portfolio.notional_eur
    es_frtb_1d = es.expected_shortfall(analysis.portfolio_returns, alpha=_FRTB_ES_ALPHA)
    es_frtb = var.scale_var(es_frtb_1d, analysis.config.horizon_days)

    html = _PAGE.substitute(
        name=analysis.config.name,
        as_of=analysis.as_of.date().isoformat(),
        notional=_eur(notional),
        horizon=analysis.config.horizon_days,
        n_obs=len(analysis.portfolio_returns),
        var_table=_var_section(analysis, notional),
        es_frtb=f"{_eur(es_frtb * notional)} ({es_frtb:.2%})",
        backtest_section=_backtest_section(analysis, notional),
        stress_section=_stress_section(analysis),
        figure_section=_figure_section(analysis, notional),
    )

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html
