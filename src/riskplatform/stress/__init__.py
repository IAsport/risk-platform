"""Package stress — scénarios historiques/hypothétiques et moteur (SPEC.md B3).

Deux familles de scénarios, deux sorties : les chocs de prix produisent un
P&L stressé (perte positive, convention VaR), les chocs de paramètres
(sigma, correlations) produisent des VaR/ES paramétriques stressées.
"""

from riskplatform.stress.engine import (
    StressedRiskResult,
    StressResult,
    StressSuite,
    apply_index_shock,
    apply_price_shock,
    estimate_betas,
    replay_window,
    run_stress_suite,
    stressed_var_parametric,
    worst_window,
)
from riskplatform.stress.scenarios import (
    DEFAULT_SCENARIOS,
    HistoricalWindow,
    IndexShock,
    PriceShock,
    RiskParamShock,
    Scenario,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "HistoricalWindow",
    "IndexShock",
    "PriceShock",
    "RiskParamShock",
    "Scenario",
    "StressResult",
    "StressSuite",
    "StressedRiskResult",
    "apply_index_shock",
    "apply_price_shock",
    "estimate_betas",
    "replay_window",
    "run_stress_suite",
    "stressed_var_parametric",
    "worst_window",
]
