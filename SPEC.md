# SPEC.md — Spécification méthodologique `risk-platform`

> Document de référence méthodologique. Toute formule du code doit pouvoir
> être retrouvée ici. Notation cohérente avec `ARCHITECTURE.md`.
> **Convention centrale : la VaR est une PERTE POSITIVE.**
>
> Organisation (spec avant code, une section par brique) :
> - **Partie I — Socle méthodologique** : la spec d'origine de `var-engine`
>   (3 VaR, change, ES, backtesting), entièrement implémentée et testée.
>   Conservée telle quelle.
> - **Partie II — Briques `risk-platform`** : une section `B<N>` par brique,
>   rédigée avant implémentation puis validée.

---

# Partie I — Socle méthodologique (hérité de `var-engine`, implémenté)

---

## 0. Notations & conventions

| Symbole        | Sens                                                              |
|----------------|-------------------------------------------------------------------|
| `P_{i,t}`      | prix du titre i à la date t (clôture ajustée, **en EUR**)         |
| `r_{i,t}`      | log-return journalier du titre i                                  |
| `w_i`          | poids du titre i dans le portefeuille (Σ w_i = 1)                 |
| `r_{p,t}`      | log-return du portefeuille à t                                    |
| `L_t`          | perte du portefeuille = `-r_{p,t}` (× notional)                  |
| `α`            | niveau de confiance (0.95 ou 0.99)                                |
| `p = 1 - α`    | probabilité de queue (5 % ou 1 %)                                 |
| `z_p`          | quantile p de la loi normale standard (z_{0.01} ≈ -2.326)         |
| `Σ`            | matrice de variance-covariance des r_{i,t} (journalière)          |
| `h`            | horizon en jours                                                  |

**Conventions d'unité et de signe :**
- Rendements = **log-returns** : `r_t = ln(P_t / P_{t-1})`. Avantages :
  additivité temporelle (`r_{t→t+k} = Σ r`), symétrie, cohérence avec
  l'hypothèse gaussienne. Limite : non additifs **entre actifs** (cf. §3).
- VaR exprimée en **perte positive** : `VaR_α` est le montant tel que la perte
  ne dépasse `VaR_α` qu'avec probabilité `p = 1 - α`. Une VaR 99 % de 1 000 €
  signifie : 1 jour sur 100 en moyenne, la perte excède 1 000 €.
- Devise de référence : **EUR**. Horizon de base : **1 jour**.

---

## 1. Construction des données

### 1.1 Log-returns
Pour chaque titre :
```
r_{i,t} = ln(P_{i,t}) - ln(P_{i,t-1})
```
On utilise les **clôtures ajustées** (dividendes/splits neutralisés) pour ne pas
introduire de faux sauts de prix.

### 1.2 Rendement de portefeuille
Approximation par somme pondérée (rebalancement quotidien, poids constants) :
```
r_{p,t} ≈ Σ_i w_i · r_{i,t}
```
**Limite assumée :** les log-returns ne sont pas exactement additifs entre
actifs (la vraie agrégation passe par les rendements arithmétiques
`R_{i,t} = e^{r}-1`, puis `R_{p,t} = Σ w_i R_{i,t}`). Pour des rendements
journaliers de l'ordre de quelques %, l'écart est du second ordre et négligeable.
On documente ce choix ; on peut basculer en rendements arithmétiques si besoin.

### 1.3 Matrice de variance-covariance
```
Σ_{ij} = Cov(r_i, r_j),   Σ estimée sur l'échantillon (estimateur empirique).
```
Variance du portefeuille :
```
σ_p² = wᵀ Σ w
```
C'est le terme clé qui capture la **diversification** : la covariance entre
titres réduit `σ_p` sous la moyenne pondérée des `σ_i`.

---

## 2. Conversion de change EUR/USD

Les titres US (AAPL, MSFT, NVDA) cotent en USD. Pour un investisseur EUR, le
risque réel inclut le risque de change. On convertit **les prix** avant de
calculer les rendements :
```
P^{EUR}_{i,t} = P^{USD}_{i,t} / S_t      où S_t = taux EURUSD (USD pour 1 EUR)
```
Le log-return EUR d'un titre US devient alors :
```
r^{EUR}_{i,t} = r^{USD}_{i,t} - Δ ln(S_t)
```
Autrement dit le rendement perçu en EUR = rendement local **moins** la variation
de l'EURUSD. Le risque de change est ainsi **endogène** aux rendements EUR : pas
besoin de l'ajouter séparément, il alimente directement Σ. C'est un point fort à
souligner (le portefeuille mixte crée une covariance non triviale).

**Trous de cotation FX (jours sans taux EURUSD).** Le taux `S_t` est
forward-fillé (dernier taux connu reporté) sur les dates de prix où aucune
cotation FX n'est disponible. *Justification :* les calendriers de cotation
actions et change ne coïncident pas toujours (jours fériés locaux différents,
décalages de fuseau), et le marché des changes ne « ferme » jamais vraiment au
sens où un taux récent reste une estimation raisonnable du taux courant —
reporter le dernier taux connu évite de perdre une date de prix valide pour un
simple trou de flux FX.

**Suppression de ligne en l'absence de taux exploitable.** Dès qu'un titre USD
figure dans le portefeuille, toute date où aucun taux FX exploitable n'existe
(même après forward-fill, typiquement en tout début d'historique) fait
supprimer **toute la ligne**, y compris les colonnes EUR qui n'avaient pourtant
pas besoin de conversion. *Justification :* la VaR porte sur le portefeuille
valorisé **dans son ensemble** à une date donnée — on a besoin d'un axe de
dates commun à tous les titres (EUR et USD) pour calculer un rendement de
portefeuille cohérent ce jour-là ; une ligne partiellement valorisable n'a pas
de sens économique et serait trompeuse si on la gardait partiellement.

---

## 3. Les trois VaR

On travaille sur la distribution des **pertes** `L = -r_p`. La `VaR_α` est le
**quantile α** des pertes (= quantile `1-α` des rendements, au signe près).

### 3.1 VaR historique (non paramétrique)
**Principe.** Aucune hypothèse de loi : on prend le quantile empirique des
rendements passés.
```
VaR_α^{hist} = - Quantile_{1-α}( {r_{p,t}} ) · notional
```
Concrètement : on trie les rendements, on lit le `(1-α)`-ième percentile (ex. le
1er percentile pour α=99 %), on change le signe.

- **Hypothèses :** le passé est représentatif du futur ; stationnarité de la
  distribution sur la fenêtre.
- **Forces :** capture les queues épaisses, l'asymétrie, sans supposer de loi.
- **Limites :** entièrement tournée vers le passé ; aveugle aux scénarios absents
  de l'échantillon ; sensible à la taille de fenêtre ; un choc sort de
  l'échantillon brutalement (effet « ghost »).

### 3.2 VaR paramétrique gaussienne (variance-covariance)
**Principe.** On suppose `r_p ~ N(μ, σ²)`.
```
VaR_α^{param} = -(μ + z_{1-α} · σ) · notional
```
avec `z_{1-α} < 0` (ex. `z_{0.01} = -2.326`, `z_{0.05} = -1.645`). En posant
souvent `μ = 0` (horizon court) :
```
VaR_α^{param} = -z_{1-α} · σ · notional = |z_{1-α}| · σ · notional
```
Version **multivariée** (variance-covariance de Markowitz) :
```
σ_p = sqrt(wᵀ Σ w),   puis VaR = |z_{1-α}| · σ_p · notional
```

- **Hypothèses :** normalité des rendements, μ connu/nul, σ stable.
- **Forces :** rapide, analytique, peu de données ; met en valeur la
  diversification via Σ ; standard historique (RiskMetrics).
- **Limites :** **sous-estime les queues** (les rendements réels sont
  leptokurtiques) → sous-estime la VaR en crise ; non valable pour des
  portefeuilles non linéaires (options).

### 3.3 VaR Monte Carlo
**Principe.** On simule un grand nombre `N` de scénarios de rendements multivariés
et on lit le quantile empirique des pertes simulées.
1. Estimer `μ_vec` et `Σ` sur l'historique.
2. Décomposer `Σ = L Lᵀ` (Cholesky).
3. Tirer `N` vecteurs `z ~ N(0, I)`, former `r^{(k)} = μ_vec + L z^{(k)}`.
4. Agréger `r_p^{(k)} = wᵀ r^{(k)}`.
5. `VaR_α^{MC} = -Quantile_{1-α}({r_p^{(k)}}) · notional`.

- **Hypothèses :** ici, loi multinormale (mêmes hypothèses que paramétrique mais
  par simulation) ; extensible à d'autres lois (Student, mélanges).
- **Forces :** flexible (toute loi, toute fonction de payoff non linéaire),
  gère naturellement de nombreux facteurs.
- **Limites :** coût de calcul ; **dépend du modèle de loi choisi** (ici
  gaussien → mêmes faiblesses de queue que le paramétrique) ; bruit de
  simulation → fixer `seed` pour la reproductibilité.

> Note de défense : sur des actifs linéaires et sous loi normale, MC et
> paramétrique doivent **converger** quand N→∞. Si on veut que MC apporte
> quelque chose de plus, il faut changer la loi (ex. Student-t) — à mentionner.

---

## 4. Mise à l'échelle temporelle (horizon)

Sous i.i.d. et μ=0, la volatilité croît en √t :
```
VaR_h = VaR_{1j} · sqrt(h)
```
- **Hypothèses :** rendements i.i.d., pas d'autocorrélation, moyenne nulle.
- **Limite :** en présence d'autocorrélation ou de mean-reversion, la règle √t
  biaise (sur- ou sous-estime) ; Bâle impose souvent 10 jours, parfois calculés
  directement plutôt que mis à l'échelle.

---

## 5. Expected Shortfall (ES / CVaR)

```
ES_α = E[ L | L > VaR_α ]     (moyenne des pertes au-delà de la VaR)
```
Estimateur historique : moyenne des pertes pires que `VaR_α`.

- **VaR vs ES :** la VaR répond « combien je peux perdre au seuil α », mais **ne
  dit rien de la sévérité au-delà** ; elle n'est **pas sous-additive** (la VaR
  d'un portefeuille peut dépasser la somme des VaR → pénalise la
  diversification, incohérent). L'ES, lui, est une **mesure de risque cohérente**
  (sous-additive) et capture l'épaisseur de la queue.
- **Pourquoi Bâle est passé à l'ES :** la revue FRTB (Fundamental Review of the
  Trading Book) remplace la VaR 99 % par l'**ES 97,5 %** pour mieux capter le
  risque extrême et garantir la sous-additivité. À horizon 10 jours, ES 97,5 %
  gaussien ≈ VaR 99 % — calibrage volontaire pour une transition douce.

---

## 6. Backtesting

Le backtesting confronte les VaR **prévues** (out-of-sample, via fenêtre
glissante) aux pertes **réalisées**. Une exception (« violation ») survient quand
la perte réalisée dépasse la VaR prévue :
```
Exception_t = 1  si  L_t > VaR_t      (sinon 0)
```
Sur `T` observations, on attend en moyenne `p·T` exceptions (p = 1-α).

### 6.1 Test de Kupiec — Proportion Of Failures (POF)
- **Hypothèse testée H0 :** le taux d'exception observé `π = x/T` est égal au
  taux théorique `p = 1-α` (couverture **non conditionnelle** correcte).
- **Statistique (rapport de vraisemblance) :**
```
LR_POF = -2 ln[ (1-p)^{T-x} · p^{x} ]  +  2 ln[ (1-π)^{T-x} · π^{x} ]
       = -2 ln[ ( (1-p)^{T-x} p^x ) / ( (1-π)^{T-x} π^x ) ]
```
avec `x` = nombre d'exceptions, `π = x/T`.
- **Loi sous H0 :** `LR_POF ~ χ²(1)`.
- **Règle de décision :** rejeter H0 si `LR_POF > χ²_{0.95}(1) = 3.841`
  (équivalent `p_value < 0.05`). Rejet ⇒ la VaR est mal calibrée (trop
  d'exceptions = VaR sous-estimée ; trop peu = VaR trop conservatrice).

> Cas limite : `x = 0` ou `x = T` → traiter les termes log proprement
> (convention `0·ln 0 = 0`) pour éviter les NaN. À implémenter avec soin.

### 6.2 Test de Christoffersen — indépendance (bonus)
Le Kupiec ne regarde que **le nombre** d'exceptions, pas leur **regroupement**.
Or des exceptions en grappe (clustering) signalent que le modèle ne réagit pas
assez vite aux changements de volatilité.

- **Hypothèse H0 :** les exceptions sont **indépendantes** dans le temps —
  la probabilité d'une exception demain ne dépend pas d'une exception aujourd'hui.
  Formalisé par une chaîne de Markov d'ordre 1 : `π_{01} = π_{11}`.
- **Transitions :** `n_{ij}` = nb de passages de l'état i à l'état j (0=pas
  d'exception, 1=exception). Estimateurs :
```
π_{01} = n_{01}/(n_{00}+n_{01}),  π_{11} = n_{11}/(n_{10}+n_{11}),
π     = (n_{01}+n_{11}) / (n_{00}+n_{01}+n_{10}+n_{11})
```
- **Statistique :**
```
LR_ind = -2 ln[ (1-π)^{n00+n10} π^{n01+n11} ]
         + 2 ln[ (1-π01)^{n00} π01^{n01} (1-π11)^{n10} π11^{n11} ]
```
  `LR_ind ~ χ²(1)`.
- **Couverture conditionnelle (CC) :** combine couverture + indépendance :
```
LR_cc = LR_POF + LR_ind ~ χ²(2)     (seuil 95 % : 5.991)
```

### 6.3 Lecture des résultats
| Résultat                     | Interprétation                                        |
|------------------------------|-------------------------------------------------------|
| Kupiec non rejeté            | bon nombre d'exceptions (couverture correcte)         |
| Kupiec rejeté, trop d'except.| VaR sous-estimée → modèle dangereux                   |
| Christoffersen ind. rejeté   | exceptions en grappe → modèle lent à réagir à la vol  |
| CC non rejeté                | VaR fiable en niveau ET dans le temps                 |

---

## 7. Points clés du socle

1. **Convention de signe + définition exacte de la VaR** (perte positive,
   quantile, « 1 jour sur 100 »).
2. **Les 3 méthodes : hypothèse / force / limite de chacune**, et le fait que
   MC gaussien ≈ paramétrique (la valeur ajoutée de MC vient du choix de loi).
3. **Pourquoi la gaussienne sous-estime la VaR** (queues épaisses /
   leptokurtisme) et **risque de change endogène** aux rendements EUR.
4. **Kupiec : H0, statistique LR ~ χ²(1), seuil 3.841**, et l'apport de
   Christoffersen (clustering). Plus **VaR vs ES** et le passage Bâle/FRTB à l'ES.

---

# Partie II — Briques `risk-platform`

---

## B0. Brique 0 — Refactoring & CI (fondations)

> **Statut : VALIDÉE le 2026-07-06** (arbitrages : benchmark hors poids,
> B0.7 #1–7 tels que proposés, archivage des résultats 2018–2024 avant
> régénération).
>
> Objectif : migrer le socle var-engine dans l'architecture cible,
> avec packaging `pyproject.toml`, portefeuille de référence en YAML et CI
> GitHub Actions. **Aucune logique de risque n'est modifiée** : les corps de
> fonctions sont déplacés à l'identique, seuls les imports et le point
> d'entrée changent.
>
> **Critère de fin** : les 67 tests existants passent dans
> la nouvelle structure, CI verte sur GitHub.

### B0.1 Plan de migration fichier par fichier

Package installable en **src-layout** : `src/riskplatform/` (nom d'import
`riskplatform`, sans tiret ni underscore — le nom PyPI/repo reste
`risk-platform`).

| Fichier actuel | Destination | Contenu déplacé | Remarques |
|---|---|---|---|
| `src/__init__.py` | `src/riskplatform/__init__.py` | docstring + `__version__ = "0.1.0"` | |
| `src/data.py` | `src/riskplatform/data/loader.py` | `download_prices`, `download_fx`, `convert_to_eur`, `to_log_returns`, `load_returns` | corps inchangés ; `data/__init__.py` ré-exporte les 5 fonctions |
| `src/portfolio.py` | `src/riskplatform/portfolio.py` | `Portfolio`, `make_equal_weight`, `portfolio_returns`, `covariance_matrix` | inchangé |
| `src/var.py` | `src/riskplatform/var/historical.py` | `var_historical` | la convention de signe (en-tête de `var.py`) est reprise dans `var/__init__.py` |
| | `src/riskplatform/var/parametric.py` | `var_parametric`, `var_parametric_portfolio` | |
| | `src/riskplatform/var/monte_carlo.py` | `var_monte_carlo` | |
| | `src/riskplatform/var/rolling.py` | `rolling_var`, `scale_var` | outillage transverse aux méthodes |
| | `src/riskplatform/es.py` | `expected_shortfall` | conforme à l'arbo cible (`es.py` au niveau racine du package) ; sera enrichi en brique 2 |
| `src/backtest.py` | `src/riskplatform/backtest/exceptions.py` | `count_exceptions` | |
| | `src/riskplatform/backtest/kupiec.py` | `kupiec_pof` | |
| | `src/riskplatform/backtest/christoffersen.py` | `christoffersen_independence`, `christoffersen_cc` | `LR_cc = LR_POF + LR_ind` → `christoffersen.py` importe `kupiec.py` |
| `src/report.py` | `src/riskplatform/reporting/report.py` | `summary_table`, `plot_var_backtest`, `render_report` | `reporting/__init__.py` ré-exporte |
| `main.py` | `src/riskplatform/cli.py` + `src/riskplatform/__main__.py` | `run`, `main` ; `build_default_portfolio` remplacé par le chargement YAML (B0.3) | `main.py` racine **supprimé** ; invocation : `riskplatform` (script console) ou `python -m riskplatform` |
| *(nouveau)* | `src/riskplatform/config.py` | `load_config(path) -> RunConfig` (portefeuille + période + alphas + notional depuis YAML) | seule fonctionnalité *nouvelle* de la brique |

Chaque sous-package (`var/`, `backtest/`, `data/`, `reporting/`) ré-exporte ses
fonctions publiques dans son `__init__.py`, de sorte que l'API reste plate :
`from riskplatform.var import var_historical, var_monte_carlo`.

**Non créés en brique 0** (anti sur-ingénierie) : `volatility/`,
`stress/`, `app/`, `notebooks/` — chacun naîtra avec sa brique.

**Tests** : les 6 fichiers existants sont conservés tels quels, seuls les
imports changent (`from src.var import …` → `from riskplatform.var import …`) ;
`tests/test_main.py` devient `tests/test_cli.py`. Le critère de fin étant « les
tests existants passent », on ne les éclate pas en brique 0 ; la règle « un
fichier de test par module source » s'appliquera aux modules nouveaux à partir
de la brique 1.

### B0.2 Packaging `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "risk-platform"
version = "0.1.0"
description = "Plateforme de mesure des risques de marché : VaR multi-méthodes, ES, backtesting"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26", "pandas>=2.1", "scipy>=1.11",
  "yfinance>=0.2.40", "matplotlib>=3.8", "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4", "mypy>=1.10", "types-PyYAML"]

[project.scripts]
riskplatform = "riskplatform.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/riskplatform"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]   # pycodestyle, pyflakes, isort, bugbear, pyupgrade

[tool.coverage.run]
source = ["riskplatform"]
```

- Installation dev : `pip install -e ".[dev]"`.
- `requirements.txt` **supprimé** (remplacé par `pyproject.toml`, source unique
  de vérité des dépendances) ; le README est mis à jour en conséquence.
- Nouvelle dépendance runtime : `PyYAML` (lecture de la config B0.3).

### B0.3 Portefeuille de référence en YAML

Fichier `config/portfolio.yaml`, chargé par `riskplatform.config.load_config`
(schéma validé à la lecture : poids ≥ 0 et somme = 1 à tolérance 1e-9, devises
∈ {EUR, USD}, dates ISO, 0 < alpha < 1) :

```yaml
name: reference
base_currency: EUR
notional_eur: 1000000
start: "2014-01-01"        # >= 10 ans, couvre COVID 2020 et la hausse des taux 2022
end: null                   # null = aujourd'hui
alphas: [0.95, 0.99]
horizon_days: 1
positions:                  # weight absent => équipondéré sur toutes les positions
  - {ticker: TTE.PA, currency: EUR}   # TotalEnergies
  - {ticker: MC.PA,  currency: EUR}   # LVMH
  - {ticker: SAN.PA, currency: EUR}   # Sanofi
  - {ticker: BNP.PA, currency: EUR}   # BNP Paribas
  - {ticker: AIR.PA, currency: EUR}   # Airbus
  - {ticker: AAPL,   currency: USD}   # Apple
  - {ticker: MSFT,   currency: USD}   # Microsoft
  - {ticker: NVDA,   currency: USD}   # NVIDIA
benchmark:                  # série de marché HORS portefeuille (poids nul)
  {ticker: ^STOXX50E, currency: EUR}  # Euro Stoxx 50 — indice de référence
```

- `load_config` renvoie un `RunConfig` (dataclass gelée : `portfolio: Portfolio`,
  `start`, `end`, `alphas`, `horizon_days`) — le reste du code ne voit jamais
  le YAML.
- La CLI prend `--config chemin.yaml` (défaut `config/portfolio.yaml`) ; les
  flags existants (`--start`, `--alphas`, …) restent et **priment** sur le YAML.
- **Indice = benchmark HORS poids** (arbitrage du 2026-07-06, amende la
  proposition initiale « 9ᵉ position équipondérée ») : le portefeuille reste à
  8 titres. Raisons : un indice n'est **pas investissable** (on ne détient pas
  « le Stoxx 50 », on détient un ETF), et son cours est un **indice de prix,
  pas total-return** — pas de dividendes, contrairement aux 8 actions en
  clôture ajustée, d'où un biais de rendement systématique s'il était pondéré.
  `^STOXX50E` est chargé comme série de marché séparée (`RunConfig.benchmark`),
  ignorée par le calcul de VaR en B0, et resservira aux stress tests (B3) et
  comme référence de comparaison. Alternatives rejetées : 9ᵉ position indice
  (biais ci-dessus), ETF réel type C50.PA (historique yfinance plus court, et
  redondant avec les 5 titres Euronext déjà en portefeuille).
- Choix de l'indice : **^STOXX50E** (Euro Stoxx 50) — en EUR, cohérent avec la
  devise de référence. Alternative rejetée : ^GSPC (S&P 500), qui aurait
  dupliqué le bloc USD.
- Passage 2018→2014 : historique ≥ 10 ans requis. Conséquence : les
  chiffres du README (calculés sur 2018–2024, 8 titres) seront **régénérés**
  en fin de brique 0 sur la nouvelle config (`python -m riskplatform`), après
  **archivage des résultats actuels** dans `docs/archive/` (pour pouvoir
  expliquer l'écart entre versions).

### B0.4 CI GitHub Actions

Fichier `.github/workflows/ci.yml` :

```yaml
name: CI
on:
  push: {branches: ["**"]}
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: {python-version: ["3.11", "3.12"]}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python-version }}"}
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: pytest --cov=riskplatform --cov-report=term-missing --cov-fail-under=85
```

- **Aucun accès réseau dans les tests** (déjà le cas : yfinance est mocké,
  séries fabriquées) → la CI passe offline. Le
  cache CSV des données de marché (pour notebooks/études) sera introduit en
  brique 1, quand l'étude 2019–2021 en aura besoin.
- Badge CI dans le README :
  `![CI](https://github.com/IAsport/risk-platform/actions/workflows/ci.yml/badge.svg)`.
- Seuil de couverture global **85 %** en garde-fou CI ; l'objectif > 90 % sur le
  cœur quantitatif (`var/`, `backtest/`, `es.py`) est vérifié dans le rapport
  `term-missing` (le seuil `fail-under` de coverage est global, pas par module).
- `mypy` : lancé **hors CI** dans un premier temps (`mypy src` en local) ; on le
  rend bloquant en CI quand la base est propre — décision à revalider en fin de
  brique.

### B0.5 Renommages `var-engine` → `risk-platform`

- ~~`SPEC.md`, `ARCHITECTURE.md` (titres)~~, ~~`main.py:134`
  (description argparse)~~, ~~`src/__init__.py` (docstring)~~ — **fait** en
  amont de cette spec (2026-07-06).
- README : retitré et mis à jour en fin de brique (badge CI, nouvelle
  arborescence, install `pip install -e .`, résultats régénérés).

### B0.6 Ordre d'implémentation (après validation)

Branche `brique-0-refactoring-ci`, commits atomiques :

1. `pyproject.toml` + suppression `requirements.txt` (structure encore plate,
   tests verts via `pip install -e .`).
2. Migration `src/` → `src/riskplatform/` selon B0.1, mise à jour des imports
   des tests. **Contrôle : 67/67 tests verts, aucune diff de logique**
   (`git diff` des corps de fonctions vide au déplacement près).
3. `config.py` + `config/portfolio.yaml` + branchement CLI (+ tests dédiés
   `tests/test_config.py` : schéma valide, poids ≠ 1 rejeté, défaut équipondéré).
4. `.github/workflows/ci.yml` + ruff (corrections lint éventuelles, sans toucher
   à la logique) + badge README.
5. Régénération `outputs/` sur la nouvelle config, mise à jour README,
   `ARCHITECTURE.md` réécrit sur la nouvelle arbo.
6. Merge dans `main` après relecture + CI verte.

### B0.7 Décisions (validées le 2026-07-06)

Arbitrages rendus : #1–7 validés tels que proposés ; #2 amendé (benchmark hors
poids, voir B0.3). En complément : archivage des résultats 2018–2024 avant
régénération, mutation tests manuels hors CI.

| # | Décision proposée | Alternative rejetée | Raison |
|---|---|---|---|
| 1 | Backend `hatchling`, src-layout | setuptools | config minimale, standard moderne, src-layout force l'installation (les tests testent le package installé, pas le dossier) |
| 2 | Indice : `^STOXX50E` | `^GSPC` | cohérence EUR, équilibre EUR/USD du portefeuille |
| 3 | Période par défaut `2014-01-01` → aujourd'hui | garder 2018–2024 | exigence d'un historique ≥ 10 ans couvrant 2020 et 2022 ; les chiffres README seront régénérés |
| 4 | Cache CSV des données différé à la brique 1 | l'introduire en brique 0 | les tests sont déjà offline ; le cache ne sert qu'aux études/notebooks (YAGNI) |
| 5 | Tests non éclatés (6 fichiers conservés, imports seuls modifiés) | un fichier par nouveau module | critère de fin = « tests existants passent » ; l'éclatement s'appliquera aux briques suivantes |
| 6 | `mypy` local d'abord, non bloquant en CI | bloquant immédiatement | éviter de mélanger refactoring et chasse aux types dans la même brique |
| 7 | Couverture : garde-fou global 85 % en CI | 90 % global | l'objectif 90 % porte sur le cœur quantitatif, pas sur `reporting/` ni la CLI |

---

## B1. Brique 1 — Volatilité conditionnelle (EWMA + GARCH(1,1))

> **Statut : VALIDÉE le 2026-07-06** — B1.8 #1–9 tels que proposés, avec
> amendement sur #8 : si le rolling GARCH sur snapshot tourne en < 5 min, il
> est inclus au test CI du résultat phare (marqué `slow` si besoin) ; sinon la
> validation manuelle est documentée en Décision.
>
> Sources : RiskMetrics Technical Document (J.P. Morgan, 1996) pour l'EWMA ;
> Bollerslev (1986) pour le GARCH ; Hull, *Options, Futures and Other
> Derivatives* (chap. volatilité) ; McNeil-Frey-Embrechts, *Quantitative Risk
> Management* §4.

### B1.0 Motivation (le chiffre qui commande la brique)

Le backtest B0 sur 2014–2026 rejette **Christoffersen partout** (les 4
configurations) et Kupiec à 99 % (historique ET paramétrique) : les exceptions
arrivent en grappes (mars 2020, 2022) parce que la VaR à volatilité constante
sur fenêtre 250 j réagit avec ~6 mois de retard à un changement de régime. La
réponse standard : remplacer σ (inconditionnel) par **σ_t (conditionnel)**,
réestimé chaque jour à partir de l'information disponible en t-1.

**Convention de datation (transverse à toute la brique) :**
`σ²_t` désigne la **prévision de variance pour le jour t, construite avec les
rendements jusqu'à t-1 inclus**. Aucune grandeur datée t ne dépend de r_t —
c'est ce qui rend le backtest out-of-sample honnête (pas de look-ahead).

### B1.1 EWMA (RiskMetrics)

**Récursion** (moyenne nulle, convention métier existante) :
```
σ²_t = λ · σ²_{t-1} + (1-λ) · r²_{t-1},      λ = 0.94 (journalier, RiskMetrics)
```
Interprétation : moyenne mobile des r² à poids exponentiels (1-λ)λ^k ; le choc
d'hier pèse 6 %, l'information a une demi-vie de ln(0.5)/ln(0.94) ≈ 11 jours.
C'est un GARCH(1,1) contraint (ω=0, α=1-λ, β=λ, persistance α+β=1 : la
variance ne revient pas vers un niveau de long terme).

- **Initialisation** : `σ²_{init_window+1}` = variance empirique (ddof=1) des
  `init_window = 30` premiers rendements ; la série σ²_t produite commence
  donc à l'indice init_window+1 (pas de valeur look-ahead, pas de σ² arbitraire).
- **Vol annualisée** : `σ_ann = σ_daily · √252` (convention 252 jours ouvrés).
- **Cas limites** : λ ∉ ]0,1[ → ValueError ; série < init_window+2 points →
  ValueError ; NaN → ValueError ; série constante → σ²_t = 0 (légal pour
  EWMA, la VaR conditionnelle vaut alors 0).

**Interface (`src/riskplatform/volatility/ewma.py`) :**
```python
def ewma_variance(returns: pd.Series, lam: float = 0.94, init_window: int = 30) -> pd.Series:
    """sigma²_t (prévision pour t, info jusqu'à t-1). Index ⊂ returns.index."""

def ewma_volatility(returns, lam=0.94, init_window=30, annualize: bool = False) -> pd.Series:
    """sqrt(ewma_variance), fois sqrt(252) si annualize."""
```

### B1.2 GARCH(1,1)

**Modèle** (moyenne nulle) : `r_t = σ_t·ε_t`, `ε_t ~ N(0,1)` i.i.d., et
```
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1},    ω > 0, α ≥ 0, β ≥ 0, α + β < 1
```
- **Variance de long terme** : `σ²_LT = ω / (1 - α - β)` (existe ssi α+β<1).
- α = réaction aux chocs, β = persistance ; α+β proche de 1 (typique : 0.98
  sur actions) = chocs très persistants. Si α+β ≥ 1 : IGARCH, variance non
  stationnaire, σ²_LT n'existe pas — le fit signale la butée (cf. edge cases).

**Log-vraisemblance gaussienne** (à maximiser, constante incluse) :
```
ℓ(ω,α,β) = -½ · Σ_t [ ln(2π) + ln(σ²_t) + r²_t / σ²_t ]
```
avec la récursion filtrée initialisée à `σ²_1 =` variance empirique de
l'échantillon d'estimation (standard ; pas de look-ahead **au sein du fit**
car l'échantillon d'estimation est de toute façon antérieur aux prévisions).

**Estimation (implémentée à la main)** :
- `scipy.optimize.minimize(method="SLSQP")` sur -ℓ, bornes
  `ω ∈ [1e-12, ∞[`, `α, β ∈ [0, 1[`, contrainte `α + β ≤ 1 - 1e-6`.
- **Point de départ par variance targeting** : `α₀=0.05, β₀=0.90,
  ω₀ = s²·(1-α₀-β₀)` où s² = variance empirique — démarrage dans la région
  plausible, convergence robuste.
- Sortie : `converged` (statut optimiseur) exposé, fit non convergé → RuntimeError
  explicite (pas de paramètres silencieusement faux).

**Prévision de variance à horizon h** (mean-reversion géométrique) :
```
σ²_{t+h} = σ²_LT + (α+β)^{h-1} · (σ²_{t+1} - σ²_LT),   h ≥ 1
```
et variance cumulée sur h jours = Σ_{k=1..h} σ²_{t+k} (remplace la règle √t
pour l'horizon ; la √t reste valable seulement si σ²_{t+1} = σ²_LT).

**Cas limites** : série constante (s²=0) → ValueError explicite (vraisemblance
dégénérée) ; série < 250 points → ValueError (MLE non fiable, α/β non
identifiés) ; NaN → ValueError ; α+β en butée (> 0.999) → fit valide mais
`persistence` le rend visible (à commenter dans l'étude).

**Interface (`src/riskplatform/volatility/garch.py`) :**
```python
@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float
    loglik: float
    n_obs: int
    @property
    def persistence(self) -> float:      # alpha + beta
    @property
    def long_run_variance(self) -> float # omega / (1 - alpha - beta)

def fit_garch(returns: pd.Series, min_obs: int = 250) -> GarchParams:
    """MLE gaussien ; RuntimeError si non convergé, ValueError si série invalide."""

def garch_variance(returns: pd.Series, params: GarchParams) -> pd.Series:
    """Filtre sigma²_t (prévision pour t, info ≤ t-1) sur toute la série."""

def forecast_variance(params: GarchParams, sigma2_next: float, horizon: int) -> np.ndarray:
    """[sigma²_{t+1}, ..., sigma²_{t+h}] par la formule de mean-reversion."""
```

### B1.3 Branchement sur la VaR (paramétrique et Monte Carlo conditionnelles)

La vol conditionnelle est modélisée sur la **série de portefeuille agrégée**
(univariée) — le GARCH multivarié (DCC) est explicitement hors périmètre
(sur-ingénierie). En conséquence l'effet de diversification vit
toujours dans r_p (l'agrégation), et σ_t capte le régime de volatilité.

```
VaR_t(α) = |z_{1-α}| · σ_t · notional        (paramétrique conditionnelle, μ=0)
```

**Interface (`src/riskplatform/var/conditional.py`) :**
```python
def var_conditional(sigma: float | pd.Series, alpha: float = 0.99, notional: float = 1.0):
    """|z_{1-alpha}| · sigma_t · notional. Scalaire ou série (même index)."""

def var_conditional_monte_carlo(sigma_t: float, alpha=0.99, notional=1.0,
                                n_sims=50_000, seed: int | None = 42) -> float:
    """Simule r = sigma_t · eps, eps ~ N(0,1) ; quantile empirique des pertes.
    Converge vers var_conditional (mêmes hypothèses) — le branchement existe
    pour recevoir les innovations Student-t en brique 2."""

def rolling_var_conditional(pnl_returns: pd.Series, vol_method: str,  # "ewma" | "garch"
                            alpha: float = 0.99, window: int = 1000,
                            refit_every: int = 20, lam: float = 0.94,
                            notional: float = 1.0) -> pd.Series:
    """VaR out-of-sample : sigma_t par EWMA (filtre pur, pas d'estimation) ou
    GARCH réestimé sur fenêtre glissante `window` tous les `refit_every` jours,
    filtrage quotidien entre deux réestimations. Index aligné sur les dates prévues."""
```

- EWMA : λ fixé (pas d'estimation) → filtre quotidien, coût nul.
- GARCH : refit MLE quotidien sur ~780 dates serait prohibitif → **réestimation
  tous les 20 jours ouvrés (~mensuelle, pratique desk), fenêtre glissante de
  1000 jours (~4 ans)**, le filtre σ²_t avance quotidiennement avec les
  derniers paramètres estimés. Aucune date ne voit r_t dans son σ²_t.

### B1.4 Données : cache CSV (préalable à l'étude)

`load_returns` gagne un cache write-through :
```python
def load_returns(tickers, currencies, start, end, cache_dir: str | Path | None = None):
    """Si cache_dir: lit prices/fx depuis CSV si présents, sinon télécharge et écrit."""
```
- Un **snapshot daté est committé** dans `data/cache/` (prix locaux + EURUSD,
  2014→date du snapshot) : le notebook d'étude et le test du résultat phare
  (B1.6) rejouent à l'identique, **sans réseau** — la CI reste hermétique.
- La CLI gagne `--cache-dir` (défaut : `data/cache`). Le README documente que
  le snapshot fige les résultats publiés (date en en-tête de fichier).

### B1.5 Étude 2019–2021 (LE résultat phare du projet)

Notebook `notebooks/etude_2020_var_conditionnelle.ipynb` + section README :

1. Backtest VaR 99 % sur les dates prévues de **2019-01 → 2021-12** (~780
   points), quatre modèles : historique 250 j, paramétrique 250 j
   (inconditionnels, déjà en B0), EWMA λ=0.94, GARCH(1,1) refit 20 j.
2. Tableau : exceptions observées/attendues, p-values Kupiec, indépendance,
   CC — **résultat attendu** : les inconditionnels échouent à Christoffersen
   (grappe de mars 2020), EWMA/GARCH passent (la VaR monte avec la vol dès
   les premiers jours du choc, les exceptions ne se suivent plus).
3. Graphes clés : (a) σ_t GARCH/EWMA vs |r_t| (le clustering se voit), (b)
   pertes vs les 4 lignes de VaR autour de mars 2020 avec exceptions
   marquées, (c) trajectoire des paramètres GARCH refités (stabilité).
4. Le verdict qualitatif du tableau est **verrouillé par un test** sur le
   snapshot (règle : pas de résultat affiché sans test) : voir B1.6.

### B1.6 Tests

`tests/test_ewma.py`, `tests/test_garch.py`, `tests/test_var_conditional.py`,
`tests/test_etude_2020.py` (un fichier par module, règle B0) :

- **EWMA** : récursion vérifiée à la main sur 4 points (valeurs calculées
  hors code) ; poids du dernier choc = 1-λ ; **pas de look-ahead** (modifier
  r_t ne change pas σ²_t, seulement σ²_{t+1}) ; annualisation ×√252 ; cas
  limites (λ hors borne, série courte, NaN, série constante → 0).
- **GARCH — validation croisée `arch`** (oracle uniquement, dép. dev) : fit
  des deux implémentations sur la même série simulée longue (T = 3000,
  ω=5e-6, α=0.08, β=0.90, seed fixe) ; **tolérances : |Δα|, |Δβ| ≤ 1e-3
  (absolu) ; ω comparé en relatif ≤ 5 % et via σ_LT ≤ 1 % relatif** — un
  seuil 1e-3 absolu serait inapplicable à ω (échelle 1e-6).
- **GARCH — cohérence interne** : paramètres retrouvés proches des vrais sur
  série simulée ; ℓ(params estimés) ≥ ℓ(params vrais) sur l'échantillon ;
  filtre σ²_t sans look-ahead ; forecast → σ²_LT quand h→∞, forecast constant
  si σ²_{t+1}=σ²_LT ; erreurs explicites (série constante, courte, NaN).
- **VaR conditionnelle** : `var_conditional(σ)` = |z|·σ exactement ; MC
  conditionnel converge vers la fermée (tolérance 2 % à 50 000 tirages, seed
  fixe) ; rolling : réestimation bien tous les `refit_every` jours (compteur
  de fits via monkeypatch), index de sortie correct.
- **Cache** : write-through sur tmp_path (2ᵉ appel sans réseau — yfinance
  mocké compté), lecture d'un snapshot existant.
- **Résultat phare (`test_etude_2020.py`)** : sur le snapshot committé,
  backtest 2019–2021 → asserts : paramétrique inconditionnelle 99 % rejetée
  par CC ; EWMA 99 % non rejetée par CC. (GARCH exclu de ce test CI pour le
  temps de calcul — vérifié dans le notebook ; si le run < 60 s en pratique,
  on l'ajoute.)
- **Tests de mutation (manuels, hors CI, documentés ici)** : inverser λ et
  1-λ ; utiliser r_t au lieu de r_{t-1} dans les récursions (look-ahead) ;
  ω/(1-α-β) → ω/(1+α+β) ; (α+β)^{h-1} → (α+β)^h ; retirer la contrainte
  α+β<1. Chacun doit faire échouer au moins un test ci-dessus.

### B1.7 Livrables & critère de fin

- `volatility/{ewma,garch}.py`, `var/conditional.py`, cache data, notebook,
  section README (tableau + graphe phare), `ARCHITECTURE.md` à jour.
- **Critère de fin** : tests verts (dont test du résultat phare sur snapshot),
  CI verte, couverture ≥ 85 % maintenue (> 90 % sur `volatility/`), étude
  notebook reproductible offline.

### B1.8 Décisions à valider avant implémentation

| # | Décision proposée | Alternative rejetée | Raison |
|---|---|---|---|
| 1 | Vol conditionnelle **univariée sur r_p agrégé** | GARCH multivarié (DCC), EWMA par titre + corrélations | anti sur-ingénierie ; la diversification vit déjà dans l'agrégation |
| 2 | Init EWMA : variance des 30 premiers points, série tronquée en conséquence | σ²₁ = r²₁ (bruité) ; variance de tout l'échantillon (look-ahead) | démarrage stable sans regarder le futur |
| 3 | GARCH : MLE 3 paramètres (SLSQP, α+β ≤ 1-1e-6), **départ** par variance targeting | variance targeting figé (2 params estimés) | le 3-params est l'exercice canonique ; le targeting ne sert qu'au point initial |
| 4 | Tolérance vs `arch` : 1e-3 absolu sur α/β, 5 % relatif sur ω, 1 % sur σ_LT | 1e-3 absolu partout | ω est d'échelle 1e-6 : 1e-3 absolu serait vide de sens |
| 5 | Backtest GARCH : fenêtre 1000 j, **refit tous les 20 j**, filtre quotidien | refit quotidien (~780 MLE) ; fenêtre expansive | coût CI/notebook ; 20 j = réestimation mensuelle réaliste ; 1000 j ≈ 4 ans stabilise α, β |
| 6 | Snapshot CSV committé dans `data/cache/` (prix + FX, daté) | pas de cache (résultats non rejouables) ; cache gitignoré seul | le test du résultat phare et le notebook doivent tourner offline et à l'identique |
| 7 | `arch` en dépendance **dev** uniquement | dépendance runtime | oracle de validation de tests, jamais importé par `src/` |
| 8 | Test CI du résultat phare : paramétrique-250 rejetée par CC vs EWMA non rejetée (GARCH vérifié au notebook) | tout tester en CI, GARCH compris | garder la CI < ~2 min ; à réévaluer si le rolling GARCH est rapide |
| 9 | Innovations **normales** en B1 (queues → B2) | passer à Student-t tout de suite | isoler l'effet « conditionnel vs inconditionnel » avant l'effet « queues » — c'est l'histoire du README |

---

## B2. Brique 2 — Monte Carlo Student-t + Expected Shortfall

> **Statut : VALIDÉE le 2026-07-06** — B2.9 #1–9 tels que proposés, avec
> ajout sur #3 : l'étude affiche le ν estimé et la sensibilité de la VaR à
> ν±2 (une ligne).
>
> Sources : McNeil-Frey-Embrechts, *Quantitative Risk Management* (§2.2 ES,
> §6.2 lois elliptiques/t multivariée) ; Acerbi & Székely, *Backtesting
> Expected Shortfall* (Risk, 2014) ; Comité de Bâle, *FRTB — Minimum capital
> requirements for market risk* (2016/2019) ; Hull.

### B2.0 Motivation (là où la brique 1 s'arrête)

L'étude B1 laisse un défaut ouvert et mesuré : la VaR conditionnelle 99 %
passe l'indépendance (p = 0,61) mais échoue à Kupiec (**21 exceptions vs
7,5**) car les résidus standardisés z_t = r_t/σ_t restent **leptokurtiques**
— le quantile gaussien 2,33 est trop court. B2 attaque le niveau par les
queues : **innovations Student-t** (dans le MC multivarié ET dans la VaR
conditionnelle, pour boucler le résultat phare), puis **Expected Shortfall**
(historique, fermé normal/t, MC), l'angle réglementaire **FRTB (ES 97,5 % vs
VaR 99 %)** et un **backtest d'ES** (Acerbi-Székely).

### B2.1 Student-t standardisée et estimation du degré de liberté

**Student-t standardisée** (variance 1, pour brancher sur σ_t sans changer
d'échelle) : si T ~ t_ν (ν > 2), alors Var(T) = ν/(ν−2) et
```
eps = T · sqrt((ν−2)/ν)          (variance 1)
q_std(p, ν) = t⁻¹_ν(p) · sqrt((ν−2)/ν)     (quantile standardisé)
```
Densité standardisée (pour le MLE) : `f_eps(x) = f_ν(x·sqrt(ν/(ν−2)))·sqrt(ν/(ν−2))`.
Quand ν → ∞ : q_std → z (retour au gaussien) — testé.

**Estimation de ν par MLE univarié** sur une série standardisée (résidus
z_t = r_t/σ_t, ou rendements/σ pour l'inconditionnel) :
`scipy.optimize.minimize_scalar(bounded, bornes [2.05, 100])` sur −Σ ln f_eps.
Borne haute atteinte ⇒ données ≈ gaussiennes (documenté, pas une erreur).

**Interface (`src/riskplatform/distributions.py`, nouveau module)** :
```python
def fit_student_df(standardized: pd.Series, bounds: tuple[float, float] = (2.05, 100.0)) -> float:
    """MLE du degré de liberté d'une t standardisée (variance 1). ValueError si série invalide."""

def student_quantile_std(p: float, df: float) -> float:
    """Quantile de la t standardisée : t⁻¹_df(p)·sqrt((df−2)/df). df > 2 exigé."""
```
Cas limites : série < 50 points → ValueError ; NaN → ValueError ; ν ≤ 2
interdit partout (variance infinie).

### B2.2 Monte Carlo multivarié Student-t

**Modèle** (t multivariée elliptique, McNeil-Frey-Embrechts §6.2) : Cholesky
sur la **matrice de corrélation** R, variable de mélange
**partagée** :
```
z ~ N(0, I_d),  w ~ χ²_ν / ν  (UN tirage par scénario, commun aux d actifs)
eps = L·z / sqrt(w),          L = chol(R)
r_i = μ_i + σ_i · sqrt((ν−2)/ν) · eps_i      (variance de r_i = σ_i²)
```
Le mélange **partagé** est ce qui fait une vraie t jointe : il crée la
**dépendance de queue** (les chocs extrêmes frappent ensemble) — des t
indépendantes par actif n'en produisent aucune. C'est le point essentiel.

**Interface (extension `var/monte_carlo.py`)** :
```python
def var_monte_carlo_student(returns, weights, alpha=0.99, notional=1.0,
                            df: float | None = None,   # None => MLE (B2.1) sur r_p/σ_p
                            n_sims=50_000, seed=42) -> float:
```
- μ_i, σ_i, R estimés sur l'échantillon (comme le MC normal) ; `df=None` ⇒
  ν estimé par MLE sur la série de portefeuille standardisée r_p/σ_p
  (simplification univariée documentée ; le MLE joint multivarié est rejeté).
- Propriétés testées : df grand (≥ 80) ⇒ converge vers `var_monte_carlo`
  normal (tolérance) ; df petit (ex. 4) ⇒ VaR 99 % strictement supérieure au
  normal ; reproductibilité seed.

### B2.3 VaR conditionnelle Student-t (bouclage du résultat phare)

Extension de `var/conditional.py` — le quantile change, σ_t ne change pas :
```
VaR_t = |q_std(1−α, ν)| · σ_t · notional
```
```python
def var_conditional(sigma, alpha=0.99, notional=1.0, df: float | None = None): ...
def var_conditional_monte_carlo(sigma_t, alpha=0.99, notional=1.0,
                                n_sims=50_000, seed=42, df: float | None = None): ...
def rolling_var_conditional(pnl_returns, vol_method, alpha=0.99, window=1000,
                            refit_every=20, lam=0.94, notional=1.0,
                            dist: str = "normal",        # "normal" | "student"
                            df: float | None = None): ...  # None => MLE par refit
```
- **Deux étapes (QMLE)** : le GARCH reste estimé en vraisemblance gaussienne
  (B1) — c'est un QMLE, **convergent pour (ω,α,β) même sous innovations non
  gaussiennes** — puis ν est estimé par MLE sur les résidus standardisés de la
  fenêtre d'estimation. Pour l'EWMA (« student ») : mêmes `window`/`refit_every`
  pour réestimer ν sur les résidus r/σ_ewma de la fenêtre (le filtre σ_t reste
  quotidien). Aucune date ne voit r_t ni dans σ²_t ni dans ν_t.
- **Étude complétée (notebook + README + test CI)** : backtest 2019–2021 à
  99 % des variantes EWMA-t et GARCH-t. **Résultat attendu : Kupiec passe
  (≈ 7–12 exceptions) ET l'indépendance tient** — la VaR conditionnelle t
  boucle l'histoire (B0 : tout échoue ; B1 : dynamique réparée ; B2 : niveau
  réparé). Les verdicts exacts seront constatés sur le snapshot à
  l'implémentation puis **verrouillés dans `tests/test_etude_2020.py`** —
  s'ils diffèrent de l'attendu, ils sont documentés tels quels (précédent B1).

### B2.4 Expected Shortfall (historique, fermé normal/t, Monte Carlo)

Convention perte positive, μ = 0. `ES_α = E[L | L > VaR_α]`.

**Formules fermées** (à vérifier contre intégration numérique scipy) :
- **Normale** : `ES_α = σ · φ(z_α) / (1−α)` avec `z_α = Φ⁻¹(α)`.
- **Student-t standardisée** (McNeil-Frey-Embrechts, éq. 2.24) — pour la t
  brute t_ν puis mise à l'échelle variance 1 :
```
ES_α(t_ν) = [ f_ν(t⁻¹_ν(α)) / (1−α) ] · [ (ν + (t⁻¹_ν(α))²) / (ν−1) ]
ES_α(std) = ES_α(t_ν) · sqrt((ν−2)/ν),   puis × σ · notional
```
**Interface (extension `es.py`)** :
```python
def expected_shortfall(pnl_returns, alpha=0.99, notional=1.0) -> float   # historique (B0, inchangé)
def es_parametric(pnl_returns, alpha=0.99, notional=1.0,
                  df: float | None = None) -> float
    """Fermé : normal si df=None, Student-t standardisée sinon. sigma = std(ddof=1), mu=0."""
def es_monte_carlo(returns, weights, alpha=0.99, notional=1.0,
                   dist="normal", df=None, n_sims=50_000, seed=42) -> float
    """Moyenne des pertes simulées au-delà du quantile alpha (mêmes moteurs que la VaR MC)."""
def es_conditional(sigma, alpha=0.99, notional=1.0, df: float | None = None)
    """ES_t = ES_alpha(loi std) · sigma_t — scalaire ou série (pour le backtest ES)."""
```
**Propriétés testées** : `ES ≥ VaR` pour toute méthode/alpha/loi (propriété
systématique) ; fermés normal et t validés contre `scipy.integrate.quad` de
l'espérance de queue (tolérance 1e-8 relative) ; t → normal quand ν → ∞ ;
ES/VaR croît quand ν décroît (queues plus épaisses).

### B2.5 Angle FRTB : ES 97,5 % vs VaR 99 %

- Tableau (README + notebook) sur le portefeuille : VaR 99 % vs ES 97,5 %
  par méthode (historique, normal, t) — sous gaussienne ES 97,5 % ≈ VaR 99 %
  (ratio théorique ≈ 1,001, à montrer numériquement), sous t l'ES décolle.
- Explication README (~10 lignes) : pourquoi Bâle a migré (la VaR ignore la
  sévérité au-delà du seuil et n'est pas sous-additive ; l'ES est cohérente
  au sens Artzner et capte la queue), et pourquoi 97,5 % (transition douce
  calibrée sur la VaR 99 % gaussienne).

### B2.6 Backtest d'ES : Acerbi-Székely Z₂ (choix justifié)

**Choix** : statistique **Z₂ d'Acerbi-Székely (2014)**, test direct et
conjoint (fréquence × sévérité) :
```
Z₂ = Σ_t [ L_t · I(L_t > VaR_t) ] / [ T·(1−α)·ES_t ]  −  1
```
E[Z₂] = 0 sous H0 (modèle correct) ; Z₂ > 0 ⇒ pertes de queue plus lourdes
que l'ES annoncé. **p-value par simulation sous H0** : B = 5 000 trajectoires
de pertes simulées depuis les prévisions du modèle (σ_t et loi
normale/Student-ν du jour), Z₂ recalculé sur chacune, p = fraction des Z₂
simulés ≥ Z₂ observé (test unilatéral — on cherche la sous-estimation).
*Alternative rejetée* : approche Emmer-Kratz-Tasche par violations de VaR
multi-niveaux — indirecte ; l'AS Z₂ teste l'ES
lui-même.

**Interface (`backtest/es_backtest.py`)** :
```python
def acerbi_szekely_z2(realized_returns: pd.Series, var_series: pd.Series,
                      es_series: pd.Series, sigma_series: pd.Series,
                      alpha: float = 0.99, df: float | None = None,
                      n_sims: int = 5_000, seed: int | None = 42) -> dict
    """dict: z_stat, p_value, reject (5 %), n_exceptions, n_obs. Aligne les 4 séries."""
```
Cas limites : 0 exception ⇒ Z₂ = −1, p-value calculée normalement (pas de
NaN) ; séries désalignées ⇒ intersection ; ES_t ≤ 0 ⇒ ValueError.

### B2.7 Tests (`tests/test_distributions.py`, `test_es.py`,
`test_es_backtest.py`, extensions `test_var_conditional.py`,
`test_var.py`/MC, `test_etude_2020.py`)

- **distributions** : ν retrouvé par MLE sur t simulée (ν=5, T=5000,
  tolérance ±1) ; données gaussiennes ⇒ ν estimé en butée haute ;
  `student_quantile_std` : variance 1 vérifiée par intégration, ν→∞ → z,
  erreurs (ν≤2, série courte/NaN).
- **MC Student-t** : df ≥ 80 ≈ MC normal (rel 3 %) ; df = 4 ⇒ VaR 99 % > 
  normal ; mélange partagé vérifié (dépendance de queue : corrélation des
  indicatrices de queue > cas normal) ; seed.
- **ES** : fermés normal/t vs `scipy.integrate.quad` (rel 1e-8) ; ES ≥ VaR
  systématique (paramétrique, MC, historique, conditionnel — paramétré sur
  α ∈ {0.95, 0.975, 0.99} et ν ∈ {4, 8, ∞}) ; ES 97,5 % normal ≈ VaR 99 %
  normal (rel < 1 %) ; monotonie en ν.
- **AS Z₂** : sur données simulées DU modèle (H0 vraie) ⇒ pas de rejet et
  Z₂ ≈ 0 ; sur ES volontairement sous-estimé (σ/2) ⇒ rejet ; 0 exception ⇒
  Z₂ = −1 sans NaN.
- **Étude** : verdicts EWMA-t / GARCH-t 99 % (Kupiec + indépendance)
  verrouillés sur le snapshot ; ES 97,5 % conditionnel backtesté par AS Z₂.
- **Mutation tests (manuels, documentés)** : (ν−1) → (ν+1) dans l'ES-t ;
  oubli du facteur sqrt((ν−2)/ν) ; mélange w par actif au lieu de partagé ;
  φ(z)/(1−α) → φ(z)·(1−α) ; I(L>VaR) → I(L≥0) dans Z₂. Chacun doit faire
  échouer un test ci-dessus.

### B2.8 Livrables & critère de fin

- `distributions.py`, extensions `var/monte_carlo.py`, `var/conditional.py`,
  `es.py`, nouveau `backtest/es_backtest.py` ; notebook étendu ou second
  notebook (queues normal vs t + bouclage 2020 + FRTB) ; README (section
  FRTB + mise à jour résultat phare) ; `ARCHITECTURE.md`.
- **Critère de fin** : tests verts (dont verdicts d'étude verrouillés), CI
  verte, couverture ≥ 85 % global / > 90 % cœur quant.

### B2.9 Décisions à valider avant implémentation

| # | Décision proposée | Alternative rejetée | Raison |
|---|---|---|---|
| 1 | t multivariée par **variable de mélange partagée** (w ~ χ²_ν/ν commun au scénario) | t indépendantes par actif | seule la t jointe crée la dépendance de queue — le point qui fait le sens du MC-t ; l'indépendante n'est pas une t multivariée |
| 2 | Cholesky sur la **corrélation** R + rescale par σ_i·√((ν−2)/ν) | Cholesky sur la covariance directement | sépare proprement corrélation (structure) et variance (échelle t) |
| 3 | ν estimé par **MLE univarié** sur la série de portefeuille standardisée ; paramètre `df` pour le fixer | MLE joint de la t multivariée | simplification défendable (la VaR porte sur r_p) ; le joint est lourd et fragile |
| 4 | GARCH-t en **deux étapes** : QMLE gaussien (B1) puis MLE de ν sur les résidus standardisés | MLE joint GARCH-t | le QMLE gaussien est convergent pour (ω,α,β) sous innovations non gaussiennes (résultat standard) ; réutilise B1 tel quel |
| 5 | Backtest ES : **Acerbi-Székely Z₂**, p-value par simulation sous H0 (B=5 000) | Emmer-Kratz-Tasche (VaR multi-niveaux) | teste l'ES directement (fréquence × sévérité), canonique depuis 2014 |
| 6 | Nouveau module `riskplatform/distributions.py` (fit ν + quantile std) | dupliquer dans var/ et es.py | un seul foyer pour la logique t standardisée, utilisée par 3 modules ; ajout mineur à l'arborescence, documenté |
| 7 | `dist="student"` réutilise le calendrier de refit B1 (window 1000, refit 20) pour ν | ν fixé une fois pour toutes | cohérence out-of-sample : ν_t comme σ_t ne voit que t-1 ; coût nul (MLE 1D) |
| 8 | Verdicts d'étude constatés puis verrouillés en CI (précédent B1) ; si l'attendu (« Kupiec passe ») n'est pas au rendez-vous, documenté tel quel | n'asserter que ce qui arrange | honnêteté du backtest — règle déjà appliquée en B1 |
| 9 | ES 97,5 % conditionnel (σ_t, t) backtesté par AS Z₂ dans l'étude | backtester seulement la VaR | c'est l'artefact « FRTB-ready » du projet |

---

## B3. Brique 3 — Stress testing + traffic light bâlois

> **Statut : VALIDÉE le 2026-07-06** — B3.10 #1–9 tels que proposés, avec
> ajout sur #5 : documenter en limite que les bêtas OLS pleine période
> sous-estiment la propagation en crise (bêtas conditionnels hors périmètre).
>
> Sources : Comité de Bâle, *Supervisory framework for the use of
> "backtesting" in conjunction with the internal models approach to market
> risk capital requirements* (janvier 1996) — le traffic light ; BCBS,
> *Stress testing principles* (2018) ; Jorion, *Value at Risk*, chap. 14
> (stress testing) ; Hull, chap. VaR.

### B3.0 Motivation (ce que la VaR ne voit pas)

La VaR/ES répond « combien je perds au seuil α, sous la distribution
estimée » — elle est aveugle à ce qui n'est pas dans l'échantillon (ou n'y
pèse presque rien) et l'étude B2 a montré ses limites structurelles sur les
sauts jour-1. Le stress testing renverse la question : **« combien je perds
SI ce scénario se réalise »**, sans probabilité attachée. C'est le
complément réglementaire obligatoire du modèle interne (Bâle exige un
programme de stress à côté de la VaR), et le volet le plus parlant pour un
poste middle office. Deuxième volet réglementaire de la brique : le
**traffic light** (Bâle 1996), le mécanisme qui relie mécaniquement le
backtesting au capital (multiplicateur 3 + plus-factor selon les exceptions
à 250 j) — il donne une conséquence concrète aux backtests B0–B2.

Au passage, le **benchmark `^STOXX50E`** chargé hors poids depuis B0
(« resservira aux stress tests B3 ») entre en service : scénario indiciel
propagé par bêtas (B3.3).

### B3.1 Conventions communes du stress

- Portefeuille **courant** : poids `w` et notional `N` de la config —
  chocs instantanés appliqués au portefeuille d'aujourd'hui, sans
  rebalancement ni réaction de gestion (hypothèse standard, documentée).
- Choc par titre = **rendement arithmétique total** `R_i` sur le scénario
  (ex. −0.30). P&L par position : `PnL_i = N · w_i · R_i` (signé, perte < 0) ;
  total `PnL = Σ_i PnL_i` ; **perte stressée `L = −PnL`** (positive si
  perte, cohérente avec la convention VaR).
- Comparaison « au capital VaR » : chaque perte est rapportée à
  deux références — `VaR 99 % 1 j` courante (défaut : historique plein
  échantillon) et un **proxy de capital IMA `3 · √10 · VaR99`**
  (multiplicateur plancher bâlois × horizon 10 j en √t, hors plus-factor —
  ordre de grandeur documenté, pas un calcul de capital réglementaire).

### B3.2 Scénarios historiques (replay de fenêtres)

Un scénario historique = une fenêtre datée `[d0, d1]` rejouée sur le
portefeuille courant. Choc par titre = rendement **arithmétique exact**
cumulé sur la fenêtre (buy-and-hold, pas de rebalancement intra-fenêtre) :
```
R_i = exp( Σ_{t ∈ [d0,d1]} r_{i,t} ) − 1
```
L'approximation log `Σ w·r` est proscrite ici : à −38 % de choc, elle
surestime la perte de plusieurs points — l'exactitude en queue est le sujet
même du stress (testé explicitement).

**Catalogue par défaut** (dates fixées dans la spec, vérifiées dans l'étude) :

| Scénario | Fenêtre | Justification |
|---|---|---|
| **COVID-19** | 2020-02-19 → 2020-03-18 | pic pré-crise → point bas de l'Euro Stoxx 50 (~−38 %) |
| **Hausse des taux 2022** | 2022-01-03 → 2022-10-12 | premier jour ouvré 2022 → creux du S&P 500 (12/10), englobe le creux Stoxx (29/09) |
| **Pire fenêtre 20 j** | extraite des données | `worst_window` (ci-dessous) trouve la pire fenêtre glissante de rendement cumulé du portefeuille — l'extraction **prouve** que c'est mars 2020 au lieu de le supposer |

2008 : **indisponible** — l'historique de référence
commence en 2014 (décision B0.7 #3) ; documenté, pas contourné.

**Cas limites** : fenêtre sans aucune date dans les rendements → ValueError ;
`d0 ≥ d1` → ValueError ; le slicing par label pandas (`returns.loc[d0:d1]`)
tolère des bornes non cotées (week-end) tant que la fenêtre est non vide.

### B3.3 Scénarios hypothétiques — chocs de prix (sortie = P&L)

- **Choc uniforme** : `R_i = −x` pour tout titre (défaut −20 %). Sanity
  check structurel : `PnL = −x·N` puisque Σw=1 — testé exactement.
- **Choc sectoriel** : dict `ticker → choc`, tickers absents choqués à 0.
  Défaut du catalogue : **« Tech US −30 % »** (AAPL, MSFT, NVDA). Les
  secteurs vivent dans le catalogue de scénarios, **pas dans le YAML** (pas
  de churn de schéma config pour un scénario).
- **Choc indiciel par bêtas** (entrée en service du benchmark) : bêta OLS
  `β_i = Cov(r_i, r_b) / Var(r_b)` estimé sur l'échantillon commun, choc
  d'indice `−x` (défaut −15 %) propagé : `R_i = β_i · (−x)`. Limite
  documentée (amendement de validation #5) : propagation
  **linéaire et en régime moyen** — des bêtas OLS estimés sur la pleine
  période **sous-estiment la propagation en crise** (les bêtas montent quand
  les corrélations montent) ; les bêtas conditionnels sont hors périmètre,
  et c'est le choc de corrélation (B3.4) qui capture cet effet par ailleurs.

Cas limites : choc dict avec ticker inconnu du portefeuille → ValueError ;
scénario indiciel sans série benchmark fournie → ValueError explicite.

### B3.4 Scénarios hypothétiques — chocs de paramètres (sortie = VaR/ES stressées)

Un choc de corrélation ou de volatilité **ne bouge aucun prix** : il n'a pas
de P&L. Il déforme la distribution → la sortie est une **VaR/ES paramétrique
stressée**, comparée à la base. Les deux familles (B3.3 P&L / B3.4 risque)
sont rendues dans **deux panneaux distincts** de la table de sortie.

Décomposition `Σ = D·R·D` avec `D = diag(σ_i)` :

- **Choc de corrélation** (« corrélations → 1 ») — mélange convexe :
```
R_s = (1−s)·R + s·J,    J = matrice de 1,    s ∈ [0,1]
```
  `R_s` reste **semi-définie positive** pour tout s (combinaison convexe de
  deux matrices PSD) — testé sur les valeurs propres. Effet limite (testé) :
  quand s→1, `σ_p → Σ w_i σ_i` (la borne comonotone : la diversification
  meurt, σ_p devient la moyenne pondérée des vols).
- **Choc de volatilité** : `σ_i → k·σ_i` (k > 0, défaut k = 2). Uniforme et
  s = 0 ⇒ `VaR* = k·VaR` (homogénéité — sanity test).
- **Combiné « crise systémique »** (défaut catalogue) : k = 2 **et** s = 1.
- Sortie : `σ_p* = √(wᵀ D_k R_s D_k w)` avec `D_k = diag(k·σ_i)`, puis
  `VaR*_α = |z_{1−α}|·σ_p*·N` et `ES*_α` fermée normale — réutilise
  `var_conditional` / `es_conditional` (B1.3/B2.4) avec σ = σ_p*.

Cas limites : `s ∉ [0,1]` → ValueError ; `k ≤ 0` → ValueError ; portefeuille
à 1 actif → choc de corrélation sans effet (`σ_p* = k·σ`), légal et testé.

### B3.5 Moteur & interfaces (`src/riskplatform/stress/`)

```python
# stress/scenarios.py — définitions (dataclasses gelées) + catalogue
@dataclass(frozen=True)
class HistoricalWindow:
    name: str
    start: str          # dates ISO
    end: str

@dataclass(frozen=True)
class PriceShock:
    name: str
    shock: float | Mapping[str, float]   # float = uniforme ; dict = par ticker (absents → 0)

@dataclass(frozen=True)
class IndexShock:
    name: str
    index_return: float                  # ex. -0.15, propagé par bêtas OLS vs benchmark

@dataclass(frozen=True)
class RiskParamShock:
    name: str
    vol_multiplier: float = 1.0          # k > 0
    corr_shift: float = 0.0              # s ∈ [0, 1] : R_s = (1-s)·R + s·J

DEFAULT_SCENARIOS: tuple = (
    HistoricalWindow("COVID-19 (19/02→18/03/2020)", "2020-02-19", "2020-03-18"),
    HistoricalWindow("Hausse des taux 2022 (03/01→12/10)", "2022-01-03", "2022-10-12"),
    PriceShock("Actions uniformes -20 %", -0.20),
    PriceShock("Tech US -30 %", {"AAPL": -0.30, "MSFT": -0.30, "NVDA": -0.30}),
    IndexShock("Euro Stoxx 50 -15 % (bêtas)", -0.15),
    RiskParamShock("Volatilités x2", vol_multiplier=2.0),
    RiskParamShock("Corrélations → 1", corr_shift=1.0),
    RiskParamShock("Crise systémique (σx2, ρ→1)", vol_multiplier=2.0, corr_shift=1.0),
)
```

```python
# stress/engine.py — application des scénarios
@dataclass(frozen=True)
class StressResult:                       # scénarios de P&L (B3.2/B3.3)
    name: str
    kind: str                             # "historical" | "price" | "index"
    pnl_by_position: pd.Series            # EUR, signé (perte < 0)
    pnl_total: float
    loss: float                           # -pnl_total (perte positive, convention VaR)

@dataclass(frozen=True)
class StressedRiskResult:                 # scénarios de paramètres (B3.4)
    name: str
    var_base: float
    var_stressed: float
    es_base: float
    es_stressed: float
    ratio: float                          # var_stressed / var_base

def worst_window(portfolio_returns: pd.Series, horizon: int = 20) -> HistoricalWindow:
    """Pire fenêtre glissante de `horizon` jours (rendement cumulé minimal).
    ValueError si série < horizon."""

def replay_window(returns, weights, scenario: HistoricalWindow, notional=1.0) -> StressResult
def apply_price_shock(weights, scenario: PriceShock, notional=1.0) -> StressResult
def apply_index_shock(returns, benchmark_returns, weights,
                      scenario: IndexShock, notional=1.0) -> StressResult
def stressed_var_parametric(returns, weights, scenario: RiskParamShock,
                            alpha=0.99, notional=1.0) -> StressedRiskResult

@dataclass(frozen=True)
class StressSuite:
    pnl_table: pd.DataFrame          # par scénario : loss_eur, pct_notional, ratio_var, ratio_capital
    pnl_by_position: pd.DataFrame    # scénarios × tickers, EUR signé (la table « par position »)
    risk_table: pd.DataFrame         # par scénario : var_base, var_stressed, es_stressed, ratio
    worst: str                       # nom du pire scénario P&L

def run_stress_suite(returns, weights, notional=1.0, benchmark_returns=None,
                     scenarios=DEFAULT_SCENARIOS, add_worst_window=True, horizon=20,
                     alpha=0.99, var_ref: float | None = None) -> StressSuite:
    """Applique le catalogue ; ajoute la pire fenêtre extraite si demandé.
    var_ref=None => VaR historique 99 % plein échantillon sur r_p ;
    ratio_capital rapporté à 3·sqrt(10)·var_ref (proxy IMA, B3.1).
    IndexShock présent sans benchmark_returns => ValueError."""
```

Intégration pipeline : `run()` (CLI) exécute la suite par défaut ;
`reporting` gagne `render_stress_report(suite, out_dir)` → `stress_tests.csv`
+ section markdown + graphe barres `stress_pnl.png` (pertes par scénario,
lignes VaR 99 % et proxy capital). Dépendances : `stress` importe
`portfolio`, `var` (historique + conditionnel), `es` — aucun cycle.

### B3.6 Traffic light bâlois (`backtest/traffic_light.py`)

Cadre Bâle (1996) : sur les **250 dernières observations**, le nombre
d'exceptions de la VaR 99 % classe le modèle en zone :

| Zone | Exceptions | Règle générale (CDF binomiale, p = 1−α) | Plus-factor |
|---|---|---|---|
| verte | 0–4 | `P(X ≤ k) < 0.95` | 0.00 |
| jaune | 5–9 | `0.95 ≤ P(X ≤ k) < 0.9999` | 0.40 / 0.50 / 0.65 / 0.75 / 0.85 (5→9) |
| rouge | ≥ 10 | `P(X ≤ k) ≥ 0.9999` | 1.00 |

Multiplicateur de capital = **3 + plus-factor**. Les bornes sont **dérivées
de la CDF binomiale** `X ~ B(window, 1−α)` (implémentation générique en α et
fenêtre) ; les bornes canoniques (4, 9) à (99 %, 250) sont **vérifiées par
test** contre la table de Bâle — pas codées en dur. La table du plus-factor,
elle, n'est définie par Bâle que pour la configuration canonique : hors
(0.99, 250), `plus_factor = None` (zone seule). Point clé : les bornes
équilibrent erreur de type I (pénaliser un bon modèle : P(X ≥ 5 | p=1 %) ≈
10,8 %) et type II (ne pas détecter un mauvais) ; la zone jaune est l'espace
d'ambiguïté où le superviseur arbitre.

```python
def basel_zone_bounds(alpha: float = 0.99, window: int = 250) -> tuple[int, int]:
    """(green_max, yellow_max) dérivés de la CDF binomiale (0.95 / 0.9999)."""

def traffic_light(exceptions: pd.Series, alpha: float = 0.99, window: int = 250) -> dict:
    """Sur les `window` DERNIERS points : n_exceptions, zone ('green'|'yellow'|'red'),
    cum_prob, plus_factor (None hors config canonique), multiplier (3 + plus).
    ValueError si len < window ou série non binaire."""

def rolling_traffic_light(exceptions: pd.Series, alpha=0.99, window=250) -> pd.DataFrame:
    """Par date : compte glissant des exceptions sur `window` jours + zone — pour le
    graphe à bandes de l'étude."""
```

La CLI ajoute au rapport la ligne traffic light (zone + multiplicateur) pour
les deux backtests déjà calculés (historique et paramétrique, par α).

### B3.7 Étude (notebook `notebooks/etude_stress_traffic_light.ipynb`)

Un notebook = une étude (précédent B2) : « que disent les scénarios, et où
en serait le multiplicateur de capital ? »

1. **Stress** : suite par défaut sur le snapshot — table des deux panneaux,
   graphe barres pertes vs VaR 99 % et proxy capital, pire scénario ;
   vérification que `worst_window(20 j)` tombe en mars 2020.
2. **Traffic light rolling 2015–2026** (fenêtre 250 j) pour trois modèles :
   paramétrique 250 j (inconditionnel), EWMA-t, GARCH-t — graphe compte
   d'exceptions avec bandes verte/jaune/rouge. **Attendu** : l'inconditionnel
   passe au rouge en 2020 (voire 2022), les conditionnels-t restent
   verts/jaunes hors pics. Verdicts **constatés à l'implémentation puis
   verrouillés par test** (`tests/test_etude_stress.py`, sur le snapshot) ;
   s'ils diffèrent de l'attendu, documentés tels quels (précédent B1/B2).
3. README : section stress (table + graphe) et section traffic light
   (graphe + lecture réglementaire : zones, plus-factor, lien au capital).

### B3.8 Tests (`tests/test_stress.py`, `tests/test_traffic_light.py`,
`tests/test_etude_stress.py`)

- **Replay** : valeurs main-calculées sur 2 titres × 3 jours (exp(Σr)−1
  exact) ; sur un gros choc, l'écart exact vs approximation log est vérifié
  (le test casse si on remplace par Σw·r) ; fenêtre vide / dates inversées →
  ValueError.
- **worst_window** : creux placé dans une série fabriquée → dates retrouvées ;
  série < horizon → ValueError.
- **PriceShock** : uniforme −x ⇒ perte = x·N exactement ; dict partiel ⇒
  tickers absents à 0 ; ticker inconnu → ValueError.
- **IndexShock** : sur données fabriquées `r_i = β_i·r_b` (bruit nul), bêtas
  et P&L exacts ; sans benchmark → ValueError.
- **RiskParamShock** : (k=1, s=0) ⇒ VaR* = VaR base ; (k=2, s=0) ⇒
  VaR* = 2·VaR ; s=1 ⇒ σ_p* = Σw_iσ_i (comonotone, tolérance num.) ; R_s PSD
  pour s ∈ {0, 0.5, 1} (valeurs propres ≥ −1e-10) ; ES* ≥ VaR* maintenu sous
  stress ; s hors [0,1] / k ≤ 0 → ValueError ; 1 actif ⇒ corrélation sans
  effet.
- **Traffic light** : bornes canoniques (4, 9) retrouvées depuis la
  binomiale (vs table Bâle) ; table plus-factor exacte ; le comptage porte
  bien sur les 250 **derniers** points (exceptions placées en tête ≠ en
  queue) ; rolling : zones datées correctes sur série fabriquée ;
  < 250 points / série non binaire → ValueError ; cohérence croisée : une
  série en zone rouge est aussi rejetée par Kupiec à 5 %.
- **Étude** : verdicts stress (perte du replay COVID > VaR 99 % 1 j ; pire
  scénario constaté) et zones traffic light des trois modèles verrouillés
  sur le snapshot.
- **Mutation tests (manuels, hors CI, documentés ici)** : `exp(Σr)−1 → Σr` ;
  `(1−s)·R + s·J → (1−s)·R` ; `β = Cov/Var → Cov/σ` ; bornes de zone
  décalées d'un (verte 0–5) ; plus-factor décalé d'une ligne ; comptage sur
  les 250 PREMIERS points ; `k·σ → k²·σ`. Chacun doit faire échouer au moins
  un test ci-dessus.

### B3.9 Livrables & critère de fin

- `stress/{scenarios.py, engine.py}` (+ ré-exports `__init__.py`),
  `backtest/traffic_light.py`, extension `reporting`/CLI (table stress +
  lignes traffic light dans le rapport), notebook d'étude exécuté, README
  (2 sections), `ARCHITECTURE.md`.
- **Critère de fin** : tests verts (dont verdicts d'étude verrouillés sur le
  snapshot), CI verte, couverture ≥ 85 % global / > 90 % sur `stress/` et
  `traffic_light.py`.

### B3.10 Décisions à valider avant implémentation

| # | Décision proposée | Alternative rejetée | Raison |
|---|---|---|---|
| 1 | Replay historique en rendements **arithmétiques exacts** `exp(Σr)−1`, buy-and-hold sur la fenêtre | approximation log `Σ w·r` (convention du reste du projet) | à −38 % l'approximation surestime la perte de plusieurs points ; l'exactitude en queue est le sujet même du stress — l'écart est testé |
| 2 | Fenêtres historiques **datées dans la spec** (COVID 19/02→18/03/2020, taux 03/01→12/10/2022) + extraction automatique `worst_window(20 j)` | extraction automatique seule ; dates configurables en YAML | des dates nommées sont plus lisibles et l'extraction **valide** le choix au lieu de le remplacer ; 2008 indisponible (historique 2014→) |
| 3 | Choc de corrélation par **mélange convexe** `R_s = (1−s)R + sJ` | forcer les hors-diagonales à 1 directement | le mélange garantit une matrice PSD pour tout s et rend la sévérité graduable ; « tout à 1 » n'est que le cas s=1 |
| 4 | Chocs de paramètres (σ, ρ) → sortie **VaR/ES stressées**, panneau séparé des chocs de prix | leur inventer un P&L | un choc de σ/ρ ne bouge aucun prix ; mélanger les deux familles rendrait la table illisible |
| 5 | Scénario indiciel par **bêtas OLS vs `^STOXX50E`** | laisser le benchmark inutilisé encore une brique | c'est l'usage prévu depuis la décision B0 « benchmark hors poids » ; limite linéaire documentée |
| 6 | Secteurs définis dans le **catalogue de scénarios** (Tech US = AAPL/MSFT/NVDA) | champ `sector` dans `portfolio.yaml` | pas de churn du schéma config pour un scénario ; le YAML reste la source de vérité du portefeuille, pas des stress |
| 7 | Zones traffic light **dérivées de la CDF binomiale** (générique α/fenêtre), bornes canoniques (4, 9) vérifiées par test ; plus-factor réservé à la config canonique | bornes 0–4/5–9/≥10 codées en dur | la dérivation expose la logique type I/type II et le test contre la table de Bâle verrouille le canonique |
| 8 | Étude : traffic light rolling sur 3 modèles (paramétrique 250 j, EWMA-t, GARCH-t), verdicts constatés puis verrouillés | n'asserter que l'attendu a priori | honnêteté du backtest (précédent B1/B2) |
| 9 | Suite de stress intégrée au rapport CLI par défaut (CSV + markdown + graphe) | attendre le dashboard B4 | la table stress est un livrable de B3 ; B4 réutilisera la même `StressSuite` telle quelle |

---

## B4. Brique 4 — Dashboard Streamlit + rapport quotidien

> **Statut : VALIDÉE le 2026-07-06** — B4.9 #1–10 tels que proposés, avec
> trois amendements : (a) date du snapshot affichée dans l'app (en tête de
> chaque page), (b) commentaire explicatif dans `requirements.txt` (artefact
> de déploiement, source de vérité = `pyproject.toml`), (c) page
> méthodologie structurée comme le **récit B0→B3** (rien ne tient → timing
> réparé → niveau à moitié réparé → risque de saut → stress testing), les
> formules en second niveau.
>
> Sources : docs Streamlit (multipage apps,
> `st.cache_data`, `streamlit.testing.v1.AppTest`) ; pour le contenu du
> rapport quotidien : pratique middle office standard (daily risk report —
> VaR/ES, exceptions, top risques), Jorion chap. 21 (VaR reporting).

### B4.0 Motivation (la restitution, pas un nouveau quant)

Les briques 0–3 ont construit tout le contenu : 3 VaR + conditionnelles,
Student-t, ES, backtests Kupiec/Christoffersen, traffic light, stress. La B4
n'ajoute **aucune formule nouvelle** : elle rend ce contenu consultable par
un non-développeur — le dashboard en ligne et le **rapport de
risque quotidien** une page, l'artefact le plus parlant pour un poste middle
office / reporting réglementaire. Le risque principal de la brique est
architectural : dupliquer le pipeline de calcul dans l'app. La parade est le
refactoring B4.1 : une **source de calcul unique** consommée par la CLI, le
dashboard et le rapport.

### B4.1 Refactoring préalable : pipeline réutilisable (`pipeline.py`)

Aujourd'hui `cli.run()` mélange calcul, `print` et rendu. On extrait le
calcul dans `src/riskplatform/pipeline.py`, **sans changer le comportement
de la CLI** (mêmes sorties console et fichiers — les tests existants restent
verts) :

```python
@dataclass(frozen=True)
class RiskAnalysis:
    config: RunConfig
    returns: pd.DataFrame                 # log-returns EUR par titre
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series | None
    var_results: list[dict]               # schéma actuel (method, alpha, horizon_days, var, es)
    backtest_results: dict[str, dict]     # schéma actuel (+ tl_* si ≥ 250 pts)
    stress: StressSuite | None            # None si < 20 pts et catalogue vide
    skipped_scenarios: tuple[str, ...]    # noms écartés (fenêtre hors échantillon, pas de benchmark)
    as_of: pd.Timestamp                   # dernière date de données

def run_analysis(config: RunConfig, cache_dir: str | None = "data/cache") -> RiskAnalysis:
    """Data -> portefeuille -> VaR/ES -> backtests (+ traffic light) -> stress.
    Silencieux (aucun print) ; la CLI imprime, le dashboard affiche."""
```

- `run_analysis` reprend exactement les calculs de `cli.run()` +
  `cli._run_stress()` (mêmes méthodes : historique/paramétrique/MC par α,
  backtests historique + paramétrique 250 j, traffic light, suite de stress
  avec filtrage des scénarios non applicables).
- `cli.run()` devient : `analysis = run_analysis(...)` → prints actuels →
  `report.render_report(...)` + `render_stress_report(...)` +
  `render_daily_report(...)` (B4.3).
- Cas limites : mêmes erreurs qu'aujourd'hui (`RuntimeError` données,
  `ValueError` config) — elles remontent de `run_analysis`.

### B4.2 Dashboard Streamlit (`app/`)

Structure multipage native (fichiers sans accents ni emojis — Windows/CI) :

```
app/
├── streamlit_app.py        # page 1 — Portefeuille & données (nom attendu par Streamlit Cloud)
├── _shared.py              # load_analysis() : run_analysis sur le snapshot, sous @st.cache_data
└── pages/
    ├── 2_VaR_ES.py
    ├── 3_Backtesting.py
    ├── 4_Stress_tests.py
    └── 5_Methodologie.py
```

**Données** : l'app lit le **snapshot committé** (`data/cache/prices.csv`,
via `load_returns(cache_dir="data/cache")`) — offline, reproductible,
déployable sans dépendre de yfinance au runtime. La date as-of est affichée
en tête de chaque page ; le rafraîchissement des données se fait par la CLI
en local (`riskplatform --no-cache`), pas depuis l'app.

**Pages** — les pages sont de la **colle UI** : tout calcul
passe par `riskplatform.*`, tout graphe par un helper `reporting` :

1. **Portefeuille & données** : table positions (ticker, devise, poids,
   notional), prix EUR normalisés base 100 (+ benchmark), stats des
   rendements (vol annualisée, skewness, kurtosis — la kurtosis > 3 introduit
   la page VaR/ES), heatmap de corrélation.
2. **VaR/ES par méthode** : contrôles α ∈ {0.95, 0.99} et horizon (√t) ;
   table VaR/ES pour historique, paramétrique, MC normal, MC Student-t,
   conditionnelles EWMA/GARCH (σ du jour) ; histogramme des rendements avec
   lignes VaR/ES ; zoom queue gauche normal vs Student-t (le message B2).
3. **Backtesting interactif** : sélecteur de modèle — historique 250 j,
   paramétrique 250 j, EWMA-t, GARCH-t (les 4 de l'étude B3) — et d'α ;
   graphe pertes vs VaR avec exceptions en rouge (helper existant
   `plot_var_backtest`) ; p-values Kupiec/Christoffersen ; traffic light :
   zone courante, multiplicateur, et graphe rolling à bandes verte/jaune/
   rouge. Les rollings conditionnels (~2 s) sont calculés à la demande sous
   `st.cache_data`.
4. **Stress tests** : les deux panneaux de la `StressSuite` (P&L et VaR/ES
   stressées), graphe barres (helper existant `plot_stress_pnl`), table P&L
   par position du scénario sélectionné, références VaR/proxy capital.
5. **Méthodologie** (amendement de validation (c)) : structurée comme le
   **récit B0→B3**, pas comme un formulaire — 5 actes : (1) *rien ne tient*
   (2014–2026 : paramétrique ET historique rejetées, clustering 2020/2022) ;
   (2) *le timing est réparé* (EWMA/GARCH : l'indépendance passe de p=0.005
   à p=0.61, mais 21 exceptions vs 7.5) ; (3) *le niveau est à moitié
   réparé* (Student-t : écart de couverture ÷2, rejet 99 % en crise
   subsiste) ; (4) *le risque résiduel est un risque de saut* (jour-1 depuis
   régime calme, irréductible pour un filtre à retard 1) ; (5) *d'où le
   stress testing* (la question « et si » sans probabilité, replay COVID =
   1,19× le proxy capital). Les formules (log-returns, 3 VaR, ES, EWMA/
   GARCH, Kupiec/Christoffersen, traffic light) en **second niveau**
   (`st.expander` par acte), avec sources.

**Nouveaux helpers `reporting/report.py`** (réutilisés par l'app ET le
rapport quotidien, testés sans Streamlit) :

```python
def plot_return_distribution(returns: pd.Series, markers: Mapping[str, float],
                             out_path: str | None = None) -> Figure:
    """Histogramme des rendements + lignes verticales (VaR/ES par méthode)."""

def plot_traffic_light(rolling: pd.DataFrame, alpha: float = 0.99,
                       out_path: str | None = None) -> Figure:
    """Compte d'exceptions rolling avec bandes verte/jaune/rouge
    (sortie de backtest.rolling_traffic_light)."""
```

Graphes **matplotlib** partout (`st.pyplot`) : zéro nouvelle dépendance de
visualisation ; l'interactivité vient des widgets Streamlit, pas du graphe.

### B4.3 Rapport quotidien HTML (`reporting/daily_report.py`)

```python
def render_daily_report(analysis: RiskAnalysis,
                        out_path: str | None = "outputs/daily_report.html") -> str:
    """Rapport de risque quotidien, une page HTML autonome. Retourne le HTML."""
```

- **HTML autonome** : CSS inline, figures matplotlib embarquées en
  **base64 PNG** — un seul fichier, zéro référence externe, envoyable par
  mail, imprimable en PDF par le navigateur (pas de dépendance PDF native).
- Templating **stdlib** (`string.Template`) — pas de jinja2 pour une page.
- Contenu — VaR, ES, exceptions récentes, top risques :
  1. En-tête : portefeuille, date as-of, notional, devise.
  2. Table VaR/ES 1 j à 95 %/99 % par méthode (+ ES 97.5 % historique, la
     référence FRTB de B2).
  3. Backtesting : zone traffic light, exceptions/250 j, multiplicateur,
     p-values Kupiec/Christoffersen, **5 dernières exceptions datées**.
  4. Top risques : pire scénario stress (perte, ratio VaR, ratio capital) +
     top 3 positions contributrices dans ce scénario.
  5. Graphe : pertes vs VaR 99 % sur les 250 derniers jours, exceptions
     marquées.
  6. Pied : une ligne de méthodologie + renvoi au dashboard.
- Généré **par défaut** par la CLI (ajout à la phase de rendu de `run()`).
- Cas limites : backtest < 250 pts → section traffic light remplacée par une
  mention explicite ; `stress=None` → section top risques omise avec mention.

### B4.4 Dépendances & déploiement

- `pyproject.toml` : extra **`app = ["streamlit>=1.36"]`** ; `dev` gagne
  `streamlit>=1.36` (pour `AppTest` en CI). Le cœur quant reste installable
  sans Streamlit ; `import streamlit` n'apparaît que sous `app/`.
- **`requirements.txt` réintroduit pour le seul déploiement** Streamlit
  Community Cloud (une ligne : `.[app]`) — documenté comme artefact de
  déploiement, la source de vérité des dépendances reste `pyproject.toml`.
- Déploiement : share.streamlit.io → repo public, main file
  `app/streamlit_app.py`. Le pas-à-pas est documenté dans le README ; le lien
  est ajouté au README après déploiement.

### B4.5 Tests (`tests/test_pipeline.py`, `test_daily_report.py`, `test_app.py`)

Toujours **sans réseau** (snapshot/données fabriquées). Pas de mutation
tests : brique de restitution, aucun nouveau code quantitatif.

- **Pipeline** : `run_analysis` sur le snapshot → champs cohérents
  (`var > 0`, `es ≥ var` à méthode/α égaux, clés backtest attendues,
  `as_of` = dernière date, stress non vide, scénarios écartés listés) ;
  équivalence CLI : les sorties `outputs/` d'un run complet sont identiques
  avant/après refactoring (les tests CLI existants restent verts, complétés
  d'un test sur `skipped_scenarios`).
- **Rapport quotidien** : le HTML contient la date as-of, la VaR 99 %
  formatée, la zone traffic light, les dates des dernières exceptions ;
  ≥ 1 image `data:image/png;base64` ; aucune URL `http` externe ; cas
  limites (< 250 pts, stress absent) → mentions explicites, pas de crash.
- **Dashboard** (`streamlit.testing.v1.AppTest`, headless, en CI) : chaque
  page s'exécute sans exception sur le snapshot ; éléments clés présents
  (titre, tables non vides) ; interaction : changer α sur la page VaR/ES
  change la VaR affichée ; changer de modèle sur la page Backtesting change
  la série tracée.
- **Helpers de plot** : figures retournées, fichiers écrits, entrées vides →
  ValueError (mêmes conventions que les helpers existants).
- **Couverture** : `app/` reste **hors du périmètre** (`source =
  riskplatform` inchangé) — le gate 85 % protège le quant ; la logique riche
  (pipeline, daily_report, helpers) vit dans `src/` et compte dedans.

### B4.6 Étude & README (pas de notebook)

Pas de notebook B4 : aucun résultat quantitatif nouveau à raconter
(décision B4.9 #8). À la place :

- README : section **Dashboard** (capture d'écran de la page Backtesting +
  lien Streamlit Cloud + `streamlit run app/streamlit_app.py`) et section
  **Rapport quotidien** (capture + commande).

### B4.7 Ordre d'implémentation (après validation)

1. `pipeline.py` + refactoring `cli.py` (tests existants verts) — commit.
2. Helpers `reporting` (distribution, traffic light) + tests — commit.
3. `reporting/daily_report.py` + intégration CLI + tests — commit.
4. `app/` (5 pages + `_shared.py`) + extra `[app]` + tests `AppTest` — commit.
5. `requirements.txt` déploiement + README (sections, captures) +
   `ARCHITECTURE.md` — commit.
6. CI verte → déploiement Streamlit Cloud → merge après validation.

### B4.8 Livrables & critère de fin

- `src/riskplatform/pipeline.py`, `reporting/daily_report.py`, 2 helpers de
  plot, `app/` (5 pages), extra `[app]`, `requirements.txt` (déploiement),
  `outputs/daily_report.html` régénéré, README (2 sections + captures),
  `ARCHITECTURE.md`.
- **Critère de fin** : tests verts (pipeline, rapport, AppTest), CI verte,
  couverture ≥ 85 % sur `src/`, dashboard fonctionnel en local, déployé sur
  Streamlit Community Cloud (lien dans le README).

### B4.9 Décisions à valider avant implémentation

| # | Décision proposée | Alternative rejetée | Raison |
|---|---|---|---|
| 1 | **`pipeline.run_analysis()` → `RiskAnalysis`**, source de calcul unique consommée par CLI, dashboard et rapport ; CLI au comportement inchangé | dupliquer les calculs dans `app/` ; ou faire de l'app un simple lecteur des CSV `outputs/` | un seul chemin de calcul testable sans UI ; les CSV perdraient les objets riches (séries, exceptions datées) |
| 2 | Le dashboard lit le **snapshot committé** (offline) ; rafraîchissement des données via la CLI seulement | bouton « refresh yfinance » dans l'app | reproductibilité + rate limits yfinance sur Streamlit Cloud ; l'app démontre l'analyse, pas l'ingestion |
| 3 | `streamlit` en **extra `[app]`** (+ dans `dev` pour la CI) | dépendance principale | le cœur quant reste installable léger ; Streamlit n'est importé que sous `app/` |
| 4 | Graphes **matplotlib** (`st.pyplot`) partout | plotly/altair | zéro dépendance nouvelle, réutilise les helpers existants ; l'interactivité vient des widgets |
| 5 | Rapport quotidien = **HTML autonome** (figures base64, CSS inline), PDF via impression navigateur ; généré par défaut par la CLI ; templating **stdlib** | weasyprint/reportlab (PDF natif) ; jinja2 | dépendances lourdes (Cairo sous Windows) et un moteur de templates pour UNE page ; l'HTML autonome s'envoie par mail tel quel |
| 6 | Tests dashboard via **`streamlit.testing.v1.AppTest`** en CI (smoke + 2 interactions) | E2E selenium/playwright ; ne pas tester l'app | AppTest est headless, sans navigateur ni réseau ; « pas de résultat affiché sans test » vaut aussi pour l'app |
| 7 | `app/` **hors du gate de couverture** (source = `riskplatform` inchangé), logique riche dans `src/` | inclure `app/` dans le gate 85 % | le gate protège le cœur quant ; les pages sont de la colle UI, testées par AppTest mais pas comptées |
| 8 | **Pas de notebook B4** ; captures d'écran README | notebook d'étude artificiel | aucun résultat quant nouveau ; le précédent « un notebook = une étude » (B2) impose de ne pas en créer un vide |
| 9 | Backtesting interactif limité aux **4 modèles de l'étude B3** (historique 250 j, paramétrique 250 j, EWMA-t, GARCH-t), calcul à la demande sous `st.cache_data` | exposer toutes les combinaisons (normal/t × EWMA/GARCH × fenêtres) | les 4 modèles portent le récit du projet ; le rolling GARCH ~2 s reste fluide au clic ; l'explosion combinatoire diluerait le message |
| 10 | `requirements.txt` réintroduit **pour le seul déploiement** (`.[app]`) | déploiement sans requirements (pari sur le support pyproject de Streamlit Cloud) | chemin d'installation documenté et standard ; `pyproject.toml` reste la source de vérité (décision B0 non contredite : plus de doublon de liste de dépendances) |
