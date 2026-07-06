"""Entry point: prints and renders the full risk-platform pipeline.

Le portefeuille de référence vient de `config/portfolio.yaml` (cf. SPEC.md
B0.3) ; les flags CLI (--start, --end, --alphas, --horizon-days) priment sur
le YAML. Depuis la brique 4 le calcul vit dans `riskplatform.pipeline`
(SPEC.md B4.1) : la CLI ne fait plus qu'imprimer le résumé et écrire les
rapports depuis le `RiskAnalysis`.
"""

from __future__ import annotations

import argparse
import dataclasses

from riskplatform import pipeline
from riskplatform.config import RunConfig, load_config
from riskplatform.pipeline import RiskAnalysis
from riskplatform.reporting import daily_report, report

DEFAULT_CONFIG_PATH = "config/portfolio.yaml"


def _print_summary(analysis: RiskAnalysis) -> None:
    """Résumé console : VaR/ES, backtests, traffic light, stress."""
    config = analysis.config
    horizon_days = config.horizon_days

    print(f"Computed {len(analysis.portfolio_returns)} daily portfolio returns.")

    for alpha in config.alphas:
        rows = [row for row in analysis.var_results if row["alpha"] == alpha]
        if rows:
            print(
                f"ES historical alpha={alpha:.2%} horizon={horizon_days}d: "
                f"{rows[0]['es']:.6f}"
            )
        for row in rows:
            print(
                f"VaR {row['method']:<11} alpha={alpha:.2%} horizon={horizon_days}d: "
                f"{row['var']:.6f}"
            )

        for method in ("historical", "parametric"):
            result = analysis.backtest_results.get(f"{method}_{int(alpha * 100)}")
            if result is None:
                continue
            print(
                f"Backtest {method:<10} alpha={alpha:.2%}: "
                f"exceptions={result['n_exceptions']}/{result['n_obs']} "
                f"Kupiec={'REJECT' if result['reject'] else 'OK'} "
                f"CC={'REJECT' if result['cc_reject'] else 'OK'}"
            )
            if "tl_zone" in result:
                multiplier = result["tl_multiplier"]
                multiplier_label = f"{multiplier:.2f}" if multiplier is not None else "n/a"
                print(
                    f"Traffic light {method:<10} alpha={alpha:.2%}: "
                    f"zone={result['tl_zone'].upper()} "
                    f"({result['tl_exceptions_250d']} exceptions/250 j, "
                    f"multiplicateur={multiplier_label})"
                )

    for name, reason in analysis.skipped_scenarios:
        print(f"Stress: scenario {name!r} skipped ({reason}).")
    suite = analysis.stress
    if suite is not None and not suite.pnl_table.empty:
        print(
            f"Stress: {len(suite.pnl_table)} P&L scenarios, worst = {suite.worst} "
            f"(loss {suite.pnl_table['loss_eur'].max():,.0f} EUR, "
            f"VaR99 ref {suite.var_ref:,.0f} EUR)."
        )


def run(config: RunConfig, cache_dir: str | None = "data/cache") -> None:
    """Run data -> portfolio -> VaR (x3) -> backtest -> stress -> report.

    cache_dir: cache CSV write-through (SPEC.md B1.4) ; None = téléchargement
    direct sans cache.
    """
    tickers = list(config.portfolio.weights.index)
    end_label = config.end if config.end is not None else "today"
    print(f"Loading market data from {config.start} to {end_label} for {len(tickers)} tickers...")
    if config.benchmark_ticker is not None:
        print(f"Loading benchmark {config.benchmark_ticker}...")

    analysis = pipeline.run_analysis(config, cache_dir=cache_dir)
    _print_summary(analysis)

    if analysis.stress is not None:
        report.render_stress_report(analysis.stress, out_dir="outputs")
    report.render_report(analysis.var_results, analysis.backtest_results, out_dir="outputs")
    daily_report.render_daily_report(analysis, out_path="outputs/daily_report.html")
    print("Report written to outputs/ (daily_report.html included).")


def main() -> None:
    """Argparse CLI: charge le YAML puis applique les overrides de ligne de commande."""
    parser = argparse.ArgumentParser(description="Run the risk-platform reference pipeline.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides YAML).")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (overrides YAML).")
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated confidence levels, e.g. 0.95,0.99 (overrides YAML).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="VaR horizon in trading days (overrides YAML).",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="CSV market data cache directory (write-through).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the CSV cache and download fresh data.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    overrides: dict[str, object] = {}
    if args.start is not None:
        overrides["start"] = args.start
    if args.end is not None:
        overrides["end"] = args.end
    if args.alphas is not None:
        overrides["alphas"] = tuple(
            float(value.strip()) for value in args.alphas.split(",") if value.strip()
        )
    if args.horizon_days is not None:
        overrides["horizon_days"] = args.horizon_days
    if overrides:
        config = dataclasses.replace(config, **overrides)

    run(config, cache_dir=None if args.no_cache else args.cache_dir)


if __name__ == "__main__":
    main()
