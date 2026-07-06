"""Page 5 — Méthodologie : le récit B0 → B3 (SPEC.md B4.2, amendement (c)).

Cinq actes, les formules en second niveau (expanders). Les chiffres cités
sont les verdicts d'étude verrouillés par tests (test_etude_2020,
test_etude_stress) — y compris ceux qui contredisent l'attendu initial.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st  # noqa: E402
from _shared import page_header  # noqa: E402

page_header("Méthodologie — le récit en cinq actes")

st.markdown(
    """
Conventions : log-returns journaliers, prix convertis en EUR avant calcul
(risque de change endogène à la covariance), VaR en **perte positive**,
horizon 1 j, niveaux 95 % et 99 %. Sources : Hull, Jorion, McNeil-Frey-
Embrechts, RiskMetrics (1996), Comité de Bâle (1996, 2018).
"""
)

st.header("Acte 1 — Rien ne tient")
st.markdown(
    """
Sur 2014-2026 (2 862 points de backtest), la VaR 99 % **paramétrique** est
rejetée par Kupiec (59 exceptions pour ~29 attendues) — et la VaR
**historique** aussi (44 exceptions). Christoffersen rejette partout :
les exceptions arrivent **en grappes** (mars 2020, 2022), signature d'une
volatilité qui varie dans le temps quand le modèle la suppose constante.
"""
)
with st.expander("Formules — les trois VaR et le test de Kupiec"):
    st.latex(r"r_t = \ln(P_t / P_{t-1}), \qquad L_t = -r_t \cdot N")
    st.latex(
        r"\text{VaR}^{hist}_\alpha = \text{quantile}_\alpha(L), \qquad "
        r"\text{VaR}^{param}_\alpha = |z_{1-\alpha}|\,\sigma_p\,N, \qquad "
        r"\sigma_p^2 = w^\top \Sigma\, w"
    )
    st.latex(
        r"LR_{POF} = -2\ln\frac{(1-p)^{T-x}p^{x}}{(1-\hat{p})^{T-x}\hat{p}^{x}}"
        r" \sim \chi^2_1, \qquad \hat{p} = x/T"
    )
    st.markdown(
        "Kupiec teste la **couverture** (le bon nombre d'exceptions), "
        "Christoffersen ajoute l'**indépendance** (pas de grappes) via une "
        "chaîne de Markov à 2 états ; le test conditionnel combine les deux."
    )

st.header("Acte 2 — Le timing est réparé (EWMA / GARCH)")
st.markdown(
    """
La volatilité conditionnelle (brique 1) remplace le sigma moyen par un
sigma **du jour**. Sur 2019-2021, l'indépendance de Christoffersen passe de
p = 0,005 à p = 0,61 : plus de grappes, la VaR triple en quelques jours de
crise. **Mais** Kupiec rejette encore : 21 exceptions pour 7,5 attendues —
le *niveau* reste trop bas, seul le *timing* est réparé.
"""
)
with st.expander("Formules — EWMA (RiskMetrics) et GARCH(1,1)"):
    st.latex(
        r"\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2, \qquad \lambda = 0{,}94"
    )
    st.latex(
        r"\sigma_t^2 = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma_{t-1}^2, \qquad "
        r"\omega > 0,\ \alpha, \beta \ge 0,\ \alpha + \beta < 1"
    )
    st.latex(
        r"\sigma^2_{LT} = \frac{\omega}{1 - \alpha - \beta}, \qquad "
        r"\mathbb{E}[\sigma^2_{t+h}] = \sigma^2_{LT} + (\alpha+\beta)^h(\sigma^2_t - \sigma^2_{LT})"
    )
    st.markdown(
        "Estimation par maximum de vraisemblance gaussien (QMLE) maison, "
        "validée contre la librairie `arch` en oracle de test. "
        "α + β ≈ 0,98 sur nos données : chocs très persistants."
    )

st.header("Acte 3 — Le niveau est à moitié réparé (Student-t)")
st.markdown(
    """
Les résidus standardisés r/sigma restent leptokurtiques : la brique 2
remplace le quantile normal par un quantile **Student-t standardisée**
(nu ≈ 6,5 par MLE). L'écart de couverture est **divisé par ~2** (plein
échantillon : 48 → 37 exceptions, p-value Kupiec ×1000) — mais le rejet
subsiste à 99 % en fenêtre de crise. À 95 %, tout passe.
"""
)
with st.expander("Formules — t standardisée, ES et FRTB"):
    st.latex(
        r"\varepsilon \sim t_\nu \cdot \sqrt{\tfrac{\nu-2}{\nu}} \quad (\text{variance } 1), "
        r"\qquad \text{VaR}_t = |t^{std}_{1-\alpha,\nu}|\,\sigma_t\, N"
    )
    st.latex(
        r"\text{ES}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha] \ \ge\ \text{VaR}_\alpha"
    )
    st.latex(r"\text{ES}^{normale}_{97{,}5\,\%} \approx \text{VaR}^{normale}_{99\,\%}")
    st.markdown(
        "L'ES est **subadditive** (cohérente au sens d'Artzner) là où la VaR "
        "ne l'est pas toujours — c'est le calibrage retenu par FRTB. "
        "Backtest d'ES : Acerbi-Székely Z₂ par simulation."
    )

st.header("Acte 4 — Le risque résiduel est un risque de saut")
st.markdown(
    """
Les exceptions restantes sont des **sauts jour-1 depuis un régime calme**
(le filtre à retard 1 ne peut pas les voir venir) et le MLE de nu calibre
toute la densité, pas la queue à 1 % (il faudrait nu ≈ 3, le centre
gaussien l'interdit). Diagnostic assumé : c'est une **limite structurelle**
des filtres de volatilité, pas un bug — la perspective serait EVT/POT.
"""
)

st.header("Acte 5 — D'où le stress testing")
st.markdown(
    """
La VaR répond « combien au seuil alpha sous la distribution estimée » ; le
stress répond « **combien si ce scénario se réalise** », sans probabilité.
Le replay COVID (19/02 → 18/03/2020) perd 36,9 % du notional : **11,2× la
VaR 99 % 1 j et 1,19× le proxy capital 3·√10·VaR** — un scénario réellement
advenu dépasse le capital calibré sur la VaR. Au traffic light bâlois, la
paramétrique 250 j reste **rouge ~483 jours** (nov. 2018 → fév. 2021) ;
EWMA-t et GARCH-t touchent le rouge au pic COVID mais en sortent en
5 mois au lieu de 27.
"""
)
with st.expander("Formules — stress et traffic light"):
    st.latex(r"R_i = e^{\sum_{t \in [d_0, d_1]} r_{i,t}} - 1, \qquad PnL = N \sum_i w_i R_i")
    st.latex(
        r"R_s = (1-s)\,R + s\,J \quad (\text{PSD } \forall s \in [0,1]), \qquad "
        r"\sigma_i \to k\,\sigma_i"
    )
    st.latex(
        r"X \sim B(250,\ 1-\alpha) : \text{verte } P(X \le k) < 0{,}95, \quad "
        r"\text{rouge } P(X \le k) \ge 0{,}9999"
    )
    st.markdown(
        "Replay en rendements arithmétiques **exacts** (l'approximation log "
        "surestime la perte de plusieurs points à −38 %). Multiplicateur de "
        "capital = 3 + plus-factor (0,40 → 0,85 en zone jaune, 1,00 en rouge)."
    )

st.divider()
st.markdown(
    """
**Limites documentées** (défendues telles quelles) : couverture 99 % encore
rejetée en crise (acte 4) ; bêtas OLS pleine période qui sous-estiment la
propagation en crise (scénario indiciel) ; approximation √t pour l'horizon ;
poids constants dans l'agrégation. Le détail, formules et sources : `SPEC.md`
du repo ; les chiffres cités sont verrouillés par `tests/test_etude_*.py`.
"""
)
