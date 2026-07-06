"""Définitions des scénarios de stress + catalogue par défaut (SPEC.md B3.2-B3.4).

Deux familles, deux sorties distinctes :
- chocs de PRIX (HistoricalWindow, PriceShock, IndexShock) -> P&L stressé ;
- chocs de PARAMÈTRES (RiskParamShock : sigma x k, correlations -> 1) -> VaR/ES
  stressées, car aucun prix ne bouge (SPEC.md B3.4).

Les dataclasses valident leurs paramètres à la construction (fail fast) ;
l'application des scénarios vit dans stress/engine.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class HistoricalWindow:
    """Fenêtre historique [start, end] rejouée sur le portefeuille courant.

    Choc par titre = rendement arithmétique exact cumulé sur la fenêtre :
    R_i = exp(sum r_i,t) - 1 (buy-and-hold, SPEC.md B3.2).
    """

    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        try:
            start = date.fromisoformat(self.start)
            end = date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(
                f"scenario {self.name!r}: start/end must be ISO dates (YYYY-MM-DD)"
            ) from exc
        if start >= end:
            raise ValueError(f"scenario {self.name!r}: start ({self.start}) must be < end")


@dataclass(frozen=True)
class PriceShock:
    """Choc de prix instantané : uniforme (float) ou par ticker (mapping).

    Les tickers absents du mapping sont choqués à 0 (SPEC.md B3.3). Les chocs
    sont des rendements arithmétiques (ex. -0.20 = -20 %).
    """

    name: str
    shock: float | Mapping[str, float]


@dataclass(frozen=True)
class IndexShock:
    """Choc d'indice propagé aux positions par bêtas OLS vs le benchmark.

    R_i = beta_i * index_return, avec beta_i = Cov(r_i, r_b) / Var(r_b) estimé
    sur l'échantillon commun. Limite documentée (amendement de validation
    B3.10 #5) : des bêtas pleine période sous-estiment la propagation en crise
    (les bêtas montent avec les corrélations) — le choc de corrélation
    (RiskParamShock) capture cet effet par ailleurs.
    """

    name: str
    index_return: float


@dataclass(frozen=True)
class RiskParamShock:
    """Choc de paramètres de risque : sigma_i -> k·sigma_i, R -> (1-s)·R + s·J.

    Le mélange convexe avec J (matrice de 1) garantit une matrice de
    corrélation semi-définie positive pour tout s dans [0, 1] ; s=1 est le
    scénario « corrélations -> 1 » (SPEC.md B3.4). Sortie = VaR/ES
    paramétriques stressées.
    """

    name: str
    vol_multiplier: float = 1.0
    corr_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.vol_multiplier <= 0.0:
            raise ValueError(f"scenario {self.name!r}: vol_multiplier must be > 0")
        if not 0.0 <= self.corr_shift <= 1.0:
            raise ValueError(f"scenario {self.name!r}: corr_shift must be in [0, 1]")


Scenario = HistoricalWindow | PriceShock | IndexShock | RiskParamShock

# Catalogue par défaut (SPEC.md B3.5). Fenêtres datées justifiées en B3.2 ;
# la pire fenêtre 20 j est extraite des données par run_stress_suite.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    HistoricalWindow("COVID-19 (19/02-18/03/2020)", "2020-02-19", "2020-03-18"),
    HistoricalWindow("Hausse des taux 2022 (03/01-12/10)", "2022-01-03", "2022-10-12"),
    PriceShock("Actions uniformes -20 %", -0.20),
    PriceShock("Tech US -30 %", {"AAPL": -0.30, "MSFT": -0.30, "NVDA": -0.30}),
    IndexShock("Euro Stoxx 50 -15 % (betas)", -0.15),
    RiskParamShock("Volatilites x2", vol_multiplier=2.0),
    RiskParamShock("Correlations -> 1", corr_shift=1.0),
    RiskParamShock("Crise systemique (sigma x2, rho -> 1)", vol_multiplier=2.0, corr_shift=1.0),
)
