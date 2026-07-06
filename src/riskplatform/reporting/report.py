"""Presentation helpers: VaR/ES tables, backtest plots, file rendering.

This module formats already-computed risk and backtest outputs. It does not
perform risk calculations.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskplatform.backtest import basel_zone_bounds

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from riskplatform.stress import StressSuite


_BASE_VAR_COLUMNS = ["method", "alpha", "horizon_days", "var", "es"]
_VAR_COLUMNS = ["method", "alpha", "horizon_days", "var", "es", "es_method"]


def _to_markdown_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_to_markdown_text(df) + "\n", encoding="utf-8")


def summary_table(var_results: list[dict]) -> pd.DataFrame:
    """Compile VaRResult dictionaries into a readable, deterministic table.

    Columns: method, alpha, horizon_days, var, es, es_method. Sorted by method
    then alpha. ES is historical, so es_method makes that explicit on every row.
    """
    table = pd.DataFrame(var_results)
    if table.empty:
        return pd.DataFrame(columns=_VAR_COLUMNS)

    missing = [column for column in _BASE_VAR_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"missing VaR result fields: {missing}")

    if "es_method" not in table.columns:
        table["es_method"] = "historical"

    table = table.loc[:, _VAR_COLUMNS]
    return table.sort_values(["method", "alpha"], kind="mergesort").reset_index(drop=True)


def plot_var_backtest(
    realized_returns: pd.Series,
    var_series: pd.Series,
    exceptions: pd.Series,
    notional: float = 1.0,
    out_path: str | None = None,
) -> Figure:
    """Plot realized losses against VaR and mark exceptions in red."""
    if realized_returns.empty:
        raise ValueError("realized_returns must not be empty")
    if var_series.empty:
        raise ValueError("var_series must not be empty")
    if exceptions.empty:
        raise ValueError("exceptions must not be empty")

    common_index = realized_returns.index.intersection(var_series.index)
    common_index = common_index.intersection(exceptions.index)
    if common_index.empty:
        raise ValueError("realized_returns, var_series and exceptions have an empty intersection")

    realized = realized_returns.loc[common_index].astype(float)
    var = var_series.loc[common_index].astype(float)
    exc = exceptions.loc[common_index].astype(int)
    if realized.isna().any() or var.isna().any() or exc.isna().any():
        raise ValueError("plot inputs contain missing values")

    losses = -realized * notional

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(common_index, losses, label="Perte réalisée", color="tab:blue", linewidth=1.5)
    ax.plot(common_index, var, label="VaR", color="tab:orange", linewidth=1.5)

    exception_dates = common_index[exc.to_numpy() == 1]
    if len(exception_dates) > 0:
        ax.scatter(
            exception_dates,
            losses.loc[exception_dates],
            color="red",
            label="Exceptions",
            zorder=3,
        )

    ax.set_title("Backtest de la VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Perte")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")

    return fig


def render_report(
    var_results: list[dict],
    backtest_results: dict[str, dict],
    out_dir: str = "outputs",
) -> None:
    """Write summary tables and backtest figures into out_dir."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    var_table = summary_table(var_results)
    var_table.to_csv(output / "var_summary.csv", index=False)
    _write_markdown(var_table, output / "var_summary.md")

    backtest_rows = []
    for name, result in backtest_results.items():
        row = {
            "name": name,
            **{
                key: value
                for key, value in result.items()
                if not isinstance(value, (pd.Series, pd.DataFrame))
            },
        }
        backtest_rows.append(row)

        required_plot_keys = {"realized_returns", "var_series", "exceptions"}
        if required_plot_keys.issubset(result):
            plot_var_backtest(
                result["realized_returns"],
                result["var_series"],
                result["exceptions"],
                notional=result.get("notional", 1.0),
                out_path=str(output / f"{name}_backtest.png"),
            )
            plt.close("all")

    backtest_table = pd.DataFrame(backtest_rows)
    backtest_table.to_csv(output / "backtest_summary.csv", index=False)
    _write_markdown(backtest_table, output / "backtest_summary.md")


def plot_return_distribution(
    returns: pd.Series,
    markers: Mapping[str, float],
    bins: int = 80,
    out_path: str | None = None,
) -> Figure:
    """Histogramme des rendements + lignes verticales VaR/ES (SPEC.md B4.2).

    markers : étiquette -> perte positive (convention VaR) ; chaque ligne est
    tracée dans l'espace des rendements à x = -valeur (queue gauche).
    """
    if returns.empty:
        raise ValueError("returns must not be empty")
    clean = returns.astype(float)
    if clean.isna().any():
        raise ValueError("returns contain missing values")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clean, bins=bins, density=True, color="tab:blue", alpha=0.55, label="Rendements")

    colors = ["tab:orange", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    for i, (label, loss) in enumerate(markers.items()):
        value = float(loss)
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"marker {label!r} must be finite, got {loss!r}")
        ax.axvline(-value, color=colors[i % len(colors)], linestyle="--", label=label)

    ax.set_title("Distribution des rendements et seuils de perte")
    ax.set_xlabel("Rendement journalier")
    ax.set_ylabel("Densité")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")
    return fig


def plot_traffic_light(
    rolling: pd.DataFrame,
    alpha: float = 0.99,
    window: int = 250,
    out_path: str | None = None,
) -> Figure:
    """Compte d'exceptions rolling sur bandes verte/jaune/rouge (SPEC.md B4.2).

    rolling : sortie de `backtest.rolling_traffic_light` (colonnes
    n_exceptions, zone). Les bornes de bandes sont re-dérivées de la CDF
    binomiale (`basel_zone_bounds`) pour le couple (alpha, window).
    """
    if rolling.empty:
        raise ValueError("rolling must not be empty")
    missing = [column for column in ("n_exceptions", "zone") if column not in rolling.columns]
    if missing:
        raise ValueError(f"rolling is missing columns: {missing}")

    green_max, yellow_max = basel_zone_bounds(alpha, window)
    top = max(float(rolling["n_exceptions"].max()) + 2.0, yellow_max + 4.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(0.0, green_max + 0.5, color="tab:green", alpha=0.15, label="Zone verte")
    ax.axhspan(green_max + 0.5, yellow_max + 0.5, color="gold", alpha=0.20, label="Zone jaune")
    ax.axhspan(yellow_max + 0.5, top, color="tab:red", alpha=0.15, label="Zone rouge")
    ax.plot(
        rolling.index,
        rolling["n_exceptions"],
        color="tab:blue",
        linewidth=1.5,
        label=f"Exceptions / {window} j",
    )
    ax.set_ylim(0.0, top)
    ax.set_title(f"Traffic light bâlois (alpha={alpha:.0%}, fenêtre {window} j)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre d'exceptions")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")
    return fig


def plot_stress_pnl(
    pnl_table: pd.DataFrame,
    var_ref: float,
    capital_ref: float,
    out_path: str | None = None,
) -> Figure:
    """Barres horizontales des pertes stressées vs VaR 99 % et proxy capital."""
    if pnl_table.empty:
        raise ValueError("pnl_table must not be empty")

    ordered = pnl_table.sort_values("loss_eur")
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(ordered) + 2))
    ax.barh(ordered.index, ordered["loss_eur"], color="tab:red", alpha=0.75)
    ax.axvline(var_ref, color="tab:orange", linestyle="--", label="VaR 99 % (1 j)")
    ax.axvline(capital_ref, color="tab:blue", linestyle="--", label="Proxy capital 3·√10·VaR")
    ax.set_title("Stress tests — pertes par scénario")
    ax.set_xlabel("Perte (EUR)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")
    return fig


def render_stress_report(suite: StressSuite, out_dir: str = "outputs") -> None:
    """Écrit la suite de stress : CSV des deux panneaux + markdown + graphe.

    Fichiers produits : stress_tests.csv (P&L par scénario), stress_risk.csv
    (VaR/ES stressées), stress_by_position.csv, stress_tests.md (les deux
    panneaux), stress_pnl.png. SPEC.md B3.5/B3.10 #9.
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    pnl_table = suite.pnl_table.reset_index()
    risk_table = suite.risk_table.reset_index()
    pnl_table.to_csv(output / "stress_tests.csv", index=False)
    risk_table.to_csv(output / "stress_risk.csv", index=False)
    suite.pnl_by_position.to_csv(output / "stress_by_position.csv")

    sections = [
        "# Stress tests",
        "",
        f"Référence : VaR 99 % (1 j) = {suite.var_ref:,.0f} EUR ; "
        f"proxy capital 3·√10·VaR = {suite.capital_ref:,.0f} EUR. "
        f"Pire scénario : **{suite.worst}**.",
        "",
        "## Scénarios de P&L (chocs de prix)",
        "",
        _to_markdown_text(pnl_table),
        "",
        "## Chocs de paramètres (VaR/ES paramétriques stressées)",
        "",
        _to_markdown_text(risk_table),
    ]
    (output / "stress_tests.md").write_text("\n".join(sections) + "\n", encoding="utf-8")

    if not suite.pnl_table.empty:
        plot_stress_pnl(
            suite.pnl_table,
            suite.var_ref,
            suite.capital_ref,
            out_path=str(output / "stress_pnl.png"),
        )
        plt.close("all")
