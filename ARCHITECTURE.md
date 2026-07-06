# ARCHITECTURE.md — `risk-platform`

> Document d'architecture : modules, fonctions publiques, structures de
> données échangées, graphe de dépendances. Référence méthodologique :
> `SPEC.md`.
> **Convention de signe globale : une VaR est une PERTE POSITIVE.**
>
> Réécrit en fin de brique 0 (2026-07-06) sur l'arborescence
> `src/riskplatform/` ; la version décrivant l'ancienne structure plate de
> var-engine vit dans l'historique git (avant le commit de migration).

---

## 0. Vue d'ensemble

```
config/portfolio.yaml ──▶ config.py ──▶ pipeline.py ──▶ RiskAnalysis (gelé)
                                          │                   │
      ┌────────────┬────────────┬─────┴──────┬──────────┐    ├─▶ cli.py (prints + rendu outputs/)
      ▼            ▼            ▼            ▼          ▼    ├─▶ app/ (dashboard Streamlit, 5 vues)
 ┌─────────┐ ┌───────────┐ ┌────────┐  ┌────────┐ ┌─────────┐└─▶ reporting/daily_report.py (HTML)
 │  data/  │▶│portfolio. │▶│  var/  │  │ es.py  │ │ stress/ │
 │ loader  │ │   py      │ └───┬────┘  └───┬────┘ └────┬────┘   ┌────────────┐
 └─────────┘ └───────────┘     │▲          │           │        │ backtest/  │
  (+ cache CSV,                │└──────┐   │           │        │ (+traffic  │
   benchmark ^STOXX50E)        │ ┌─────┴───┴─┐         │        │   light)   │
                               │ │volatility/│(sigma_t)│        └─────┬──────┘
                               │ └───────────┘         │              │
                               └───────────┬───────────┴──────────────┘
                                           ▼
                                   ┌──────────────┐
                                   │  reporting/  │
                                   └──────────────┘
```

Flux : `config` charge le portefeuille de référence depuis le YAML →
**`pipeline.run_analysis` (B4) exécute tout le calcul, sans effet de bord** :
`data` télécharge prix + FX (ou relit le snapshot `data/cache/`) et fabrique
les log-returns EUR → `portfolio` agrège → `volatility` produit σ_t
(EWMA/GARCH) → `var` et `es` calculent les mesures → `backtest` confronte VaR
prévue et pertes réalisées (Kupiec/Christoffersen + traffic light bâlois) →
`stress` applique les scénarios — le tout renvoyé dans un **`RiskAnalysis`
gelé**, consommé par trois restitutions : `cli.py` (prints + rendu
`outputs/`, dont `daily_report.html`), le dashboard `app/` (5 vues Streamlit
sur le snapshot) et `reporting/daily_report.py` (rapport une page autonome).

**Convention de datation (SPEC.md B1.0), transverse :** toute grandeur σ²_t ou
VaR_t est une **prévision pour t construite avec l'information ≤ t-1** — le
backtest est out-of-sample sans look-ahead.

### Structures de données partagées (contrats inter-modules)

| Nom logique        | Type Python                     | Forme / index |
|--------------------|---------------------------------|----------------|
| `PricesEUR`        | `pd.DataFrame`                  | index = dates, colonnes = tickers, prix **en EUR** |
| `Returns`          | `pd.DataFrame`                  | index = dates, colonnes = tickers, **log-returns** journaliers |
| `Weights`          | `pd.Series`                     | index = tickers, somme = 1.0 |
| `PortfolioReturns` | `pd.Series`                     | index = dates, log-return agrégé du portefeuille |
| `VaRResult`        | `dict[str, float]`              | clés : `method`, `alpha`, `horizon_days`, `var`, `es` (perte positive) |
| `VaRSeries`        | `pd.Series`                     | index = dates, VaR glissante (pour backtest) |
| `BacktestResult`   | `dict[str, float \| bool]`      | clés : `n_obs`, `n_exceptions`, `expected`, `lr_stat`, `p_value`, `reject` |
| `RunConfig`        | dataclass gelée (`config.py`)   | portefeuille + benchmark + période + alphas + horizon |
| `StressSuite`      | dataclass gelée (`stress/`)     | `pnl_table`, `pnl_by_position`, `risk_table`, `worst`, `var_ref`, `capital_ref` (pertes positives, EUR) |
| `RiskAnalysis`     | dataclass gelée (`pipeline.py`) | run complet : `config`, `returns`, `portfolio_returns`, `benchmark_returns`, `var_results`, `backtest_results`, `stress`, `skipped_scenarios` (paires nom/raison), `as_of` |

Conventions transverses : valeurs monétaires en **EUR** ; VaR/ES en **perte
positive** (fraction du portefeuille ou montant selon `notional`) ; horizon de
base **1 jour** (échelle √h documentée `SPEC.md §4`).

---

## 1. `config.py`

Charge et valide `config/portfolio.yaml` (source de vérité du portefeuille de
référence). Le reste du code ne voit jamais le YAML.

| Fonction / classe | Contrat |
|---|---|
| `RunConfig` | dataclass gelée : `name`, `portfolio: Portfolio`, `start`, `end` (None = aujourd'hui), `alphas`, `horizon_days`, `benchmark_ticker/currency` (None si absent) |
| `load_config(path) -> RunConfig` | validation : positions non vides et uniques, devises ∈ {EUR, USD}, poids tous présents ou tous absents (défaut équipondéré) et Σ=1 à 1e-9, dates ISO avec `end > start`, alphas ∈ ]0,1[, horizon ≥ 1, notional > 0 |

Le **benchmark** (`^STOXX50E`) est une série de marché HORS poids : non
investi, ignoré par le calcul de VaR, réservé aux stress tests (B3).

## 2. `data/` (`loader.py`, ré-exporté par `data/__init__.py`)

Acquisition des clôtures ajustées (yfinance), du taux EURUSD, conversion EUR,
log-returns. Aucun calcul de risque. Règles FX (forward-fill, suppression de
ligne) : `SPEC.md §2`.

| Fonction | Contrat |
|---|---|
| `download_prices(tickers, start, end) -> DataFrame` | clôtures ajustées en devise locale, jointure interne sur dates communes |
| `download_fx(pair, start, end) -> Series` | taux EURUSD journalier (USD pour 1 EUR) |
| `convert_to_eur(prices, currencies, eurusd) -> DataFrame` | `price_eur = price_usd / eurusd`, FX forward-fillé, lignes sans taux supprimées |
| `to_log_returns(prices_eur) -> DataFrame` | `r_t = ln(P_t/P_{t-1})`, première ligne supprimée |
| `load_returns(tickers, currencies, start, end, cache_dir=None)` | façade ; si `cache_dir` : cache CSV write-through (lit `prices.csv`/`eurusd.csv` s'ils existent, sinon télécharge et écrit). Snapshot daté committé dans `data/cache/` |

## 3. `portfolio.py`

| Fonction / classe | Contrat |
|---|---|
| `Portfolio` | dataclass gelée : `weights` (Σ=1), `currencies`, `notional_eur` |
| `make_equal_weight(tickers, currencies, notional_eur)` | portefeuille 1/N |
| `portfolio_returns(returns, weights) -> Series` | `r_p ≈ Σ w·r` (approximation documentée `SPEC.md §1.2`) |
| `covariance_matrix(returns, weights=None) -> DataFrame` | Σ journalière, sous-sélection sur `weights.index` |

## 3 bis. `distributions.py` — Student-t standardisée (SPEC.md B2.1)

Un seul foyer pour la logique t standardisée (variance 1), consommé par
`var/monte_carlo`, `var/conditional` et `es`.

| Fonction | Contrat |
|---|---|
| `student_quantile_std(p, df)` | `t⁻¹_df(p)·√((df−2)/df)`, df > 2 exigé ; → z quand df → ∞ |
| `fit_student_df(standardized, bounds=(2.05, 100))` | MLE 1D (bounded) du degré de liberté sur série de variance ~1 ; butée haute = données ≈ gaussiennes |

## 4. `volatility/` — volatilité conditionnelle (SPEC.md B1)

Validateurs partagés dans `riskplatform/_validation.py`. La lib `arch` n'est
qu'un oracle de tests (dépendance dev, jamais importée par `src/`).

| Module | Fonction / classe | Contrat |
|---|---|---|
| `ewma.py` | `ewma_variance(returns, lam=0.94, init_window=30)` | récursion RiskMetrics, init = variance des 30 premiers points, sortie indexée `returns.index[30:]` |
| `ewma.py` | `ewma_volatility(..., annualize=False)` | `√σ²_t`, ×√252 si annualize |
| `garch.py` | `GarchParams` | dataclass gelée : ω, α, β, loglik, n_obs + `persistence`, `long_run_variance` |
| `garch.py` | `fit_garch(returns, min_obs=250)` | MLE gaussien SLSQP (bornes + α+β ≤ 1-1e-6), départ variance targeting, standardisation interne des rendements ; RuntimeError si non convergé |
| `garch.py` | `garch_variance(returns, params)` | filtre σ²_t (info ≤ t-1), init variance d'échantillon |
| `garch.py` | `forecast_variance(params, sigma2_next, horizon)` | mean-reversion `σ²_LT + (α+β)^(h-1)·(σ²_{t+1} − σ²_LT)` |

## 5. `var/` — cœur méthodologique (SPEC.md §3-4, B1.3)

Convention de signe rappelée dans `var/__init__.py`, qui ré-exporte tout
(`from riskplatform.var import var_historical, ...`).

| Module | Fonction | Contrat |
|---|---|---|
| `historical.py` | `var_historical(pnl_returns, alpha, notional)` | quantile empirique (1-α), méthode `linear` |
| `parametric.py` | `var_parametric(pnl_returns, alpha, notional, mean_zero=True)` | `-(μ + z·σ)`, μ=0 par défaut |
| `parametric.py` | `var_parametric_portfolio(returns, weights, alpha, notional)` | `σ_p = √(wᵀΣw)` |
| `monte_carlo.py` | `var_monte_carlo(returns, weights, alpha, notional, n_sims=50000, seed=42)` | multinormale via Cholesky (jitter 1e-12 si non-SDP) |
| `monte_carlo.py` | `var_monte_carlo_student(returns, weights, alpha, notional, df=None, n_sims, seed)` | t multivariée : Cholesky sur la CORRÉLATION, mélange w~χ²_df/df PARTAGÉ (dépendance de queue) ; df=None ⇒ MLE sur r_p standardisé |
| `conditional.py` | `var_conditional(sigma, alpha, notional, df=None)` | `\|q_{1-α}\|·σ_t·notional` — quantile normal (df=None) ou t standardisée ; scalaire ou série |
| `conditional.py` | `var_conditional_monte_carlo(sigma_t, alpha, notional, n_sims, seed, df=None)` | r = σ_t·ε, ε normal ou t standardisée |
| `conditional.py` | `rolling_var_conditional(pnl_returns, vol_method, alpha, window=1000, refit_every=20, lam=0.94, notional, dist="normal", df=None)` | `vol_method ∈ {ewma, garch}` ; dist="student" : ν réestimé par MLE sur les résidus de la fenêtre à chaque refit (QMLE 2 étapes) |
| `rolling.py` | `scale_var(var_1d, horizon_days)` | `VaR_h = VaR_1j·√h` |
| `rolling.py` | `rolling_var(pnl_returns, method, alpha, window=250, notional)` | VaR out-of-sample sur fenêtre glissante, `method ∈ {historical, parametric}` |

## 6. `es.py`

| Fonction | Contrat |
|---|---|
| `expected_shortfall(pnl_returns, alpha, notional)` | ES historique : moyenne des pertes > VaR_α |
| `es_parametric(pnl_returns, alpha, notional, df=None)` | fermé : `σ·φ(z_α)/(1−α)` (normal) ou formule t standardisée MFE 2.24 ; validé vs `scipy.integrate.quad` |
| `es_conditional(sigma, alpha, notional, df=None)` | `ES_α(loi std)·σ_t` — scalaire ou série (alimente le backtest d'ES) |
| `es_monte_carlo(returns, weights, alpha, notional, dist, df, n_sims, seed)` | moyenne des pertes simulées au-delà du quantile (mêmes moteurs que la VaR MC) |

## 6 bis. `stress/` — stress testing (SPEC.md B3)

Deux familles de scénarios (dataclasses gelées, validation à la construction
dans `scenarios.py` ; application dans `engine.py`) : chocs de **prix**
(`HistoricalWindow` rejouée en arithmétique exact `exp(Σr)−1`, `PriceShock`
uniforme ou par ticker, `IndexShock` propagé par bêtas OLS vs benchmark) →
P&L stressé ; chocs de **paramètres** (`RiskParamShock` : σ×k, corrélations
→ 1 par mélange convexe `(1−s)R+sJ`, PSD) → VaR/ES paramétriques stressées.

| Fonction | Contrat |
|---|---|
| `worst_window(portfolio_pnl, horizon=20)` | pire fenêtre glissante de rendement cumulé → `HistoricalWindow` |
| `replay_window(returns, weights, scenario, notional)` | `StressResult` (P&L par position signé, perte positive) |
| `apply_price_shock(weights, scenario, notional)` | choc uniforme (float) ou dict ticker → choc (absents = 0) |
| `estimate_betas(returns, benchmark_returns)` | `β_i = Cov(r_i, r_b)/Var(r_b)` sur dates communes (pleine période, limite documentée) |
| `apply_index_shock(returns, benchmark_returns, weights, scenario, notional)` | `R_i = β_i · choc` ; ValueError sans benchmark |
| `stressed_var_parametric(returns, weights, scenario, alpha, notional)` | `StressedRiskResult` : σ_p* = √(wᵀD_k R_s D_k w), VaR/ES fermées normales |
| `run_stress_suite(returns, weights, notional, benchmark_returns, scenarios, add_worst_window, horizon, alpha, var_ref)` | `StressSuite` ; `var_ref=None` ⇒ VaR historique plein échantillon ; `capital_ref = 3·√10·var_ref` |
| `DEFAULT_SCENARIOS` | catalogue spec B3.5 : COVID, taux 2022, uniforme −20 %, Tech US −30 %, indice −15 %, σ×2, ρ→1, combiné |

## 7. `backtest/` (SPEC.md §6, B2.6, B3.6)

Validateurs et `log_term` (convention 0·ln 0 = 0) dans `backtest/_common.py` ;
ré-exports dans `backtest/__init__.py`.

| Module | Fonction | Contrat |
|---|---|---|
| `exceptions.py` | `count_exceptions(realized_returns, var_series, notional)` | indicatrice 0/1 sur l'intersection des dates |
| `kupiec.py` | `kupiec_pof(exceptions, alpha)` | H0 : taux observé = 1-α ; LR ~ χ²(1), seuil 3.841 |
| `christoffersen.py` | `christoffersen_independence(exceptions)` | H0 : π01 = π11 (Markov ordre 1) ; LR ~ χ²(1) |
| `christoffersen.py` | `christoffersen_cc(exceptions, alpha)` | LR_cc = LR_POF + LR_ind ~ χ²(2), seuil 5.991 |
| `es_backtest.py` | `acerbi_szekely_z2(realized, var_series, es_series, sigma_series, alpha, df, n_sims, seed)` | Z₂ d'Acerbi-Székely (fréquence × sévérité), E[Z₂]=0 sous H0, p-value par simulation (normal ou t) |
| `traffic_light.py` | `basel_zone_bounds(alpha, window)` | bornes (green_max, yellow_max) dérivées de la CDF binomiale aux seuils 0.95/0.9999 — (4, 9) en canonique, vérifié vs table de Bâle |
| `traffic_light.py` | `traffic_light(exceptions, alpha=0.99, window=250)` | zone sur les 250 DERNIERS points + plus-factor et multiplicateur 3+plus (config canonique seulement, None sinon) |
| `traffic_light.py` | `rolling_traffic_light(exceptions, alpha, window)` | compte glissant + zone par date (graphe à bandes de l'étude) |

## 8. `reporting/` (`report.py`, `daily_report.py`)

| Fonction | Contrat |
|---|---|
| `summary_table(var_results) -> DataFrame` | table triée method/alpha, colonne `es_method` explicite |
| `plot_var_backtest(realized, var_series, exceptions, notional, out_path)` | pertes vs VaR, exceptions en rouge ; backend Agg |
| `render_report(var_results, backtest_results, out_dir="outputs")` | CSV + markdown + PNG par backtest (lignes traffic light incluses si présentes) |
| `plot_return_distribution(returns, markers, bins, out_path)` (B4) | histogramme des rendements + lignes verticales VaR/ES (pertes positives tracées à −valeur) |
| `plot_traffic_light(rolling, alpha, window, out_path)` (B4) | compte d'exceptions rolling sur bandes verte/jaune/rouge (bornes re-dérivées de la binomiale) |
| `plot_stress_pnl(pnl_table, var_ref, capital_ref, out_path)` | barres des pertes stressées vs lignes VaR 99 % et proxy capital |
| `render_stress_report(suite, out_dir="outputs")` | `stress_tests.csv/md`, `stress_risk.csv`, `stress_by_position.csv`, `stress_pnl.png` |
| `render_daily_report(analysis, out_path) -> str` (B4) | rapport quotidien **une page HTML autonome** (CSS inline, figure base64, zéro référence externe) : VaR/ES EUR + ES 97,5 % FRTB, traffic light + 5 dernières exceptions datées, top risques stress, graphe 250 j ; sections absentes mentionnées explicitement |

## 9. `pipeline.py` — source de calcul unique (SPEC.md B4.1)

| Fonction / classe | Contrat |
|---|---|
| `RiskAnalysis` | dataclass gelée (cf. contrats §0) — le résultat complet d'un run |
| `run_analysis(config, cache_dir="data/cache") -> RiskAnalysis` | data → portfolio → VaR ×3 + ES (par alpha, remis à l'échelle `horizon_days`) → backtests historique/paramétrique 250 j (+ `tl_*` si ≥ 250 points prévus) → stress (benchmark chargé si configuré ; scénarios non applicables écartés et listés dans `skipped_scenarios`, le moteur restant strict). **Silencieux** : aucun print, aucune écriture fichier |

## 9 bis. `cli.py` / `__main__.py` et `app/`

`main()` : argparse → `load_config(--config, défaut config/portfolio.yaml)` →
overrides CLI (`--start`, `--end`, `--alphas`, `--horizon-days`) via
`dataclasses.replace` → `run(config)`. `run` (mince depuis B4) :
`pipeline.run_analysis` → résumé console (VaR/ES, backtests, traffic light,
stress, scénarios écartés) → rendu `outputs/` (`render_report`,
`render_stress_report`, `render_daily_report`).

`app/` (hors package, extra `[app]`) : `streamlit_app.py` = routeur
`st.navigation` ; `views/{portefeuille, var_es, backtesting, stress_tests,
methodologie}.py` = les 5 vues, chacune un script autonome testable par
`AppTest` ; `_shared.py` = `load_analysis()` (le `run_analysis` du snapshot
sous `st.cache_data`), `backtest_model(model, alpha)` (les 4 modèles du récit
B3 à la demande), `mc_student_var(alpha)` et le bandeau commun affichant la
date du snapshot. Tout le calcul passe par `riskplatform.*` — l'app est de la
colle UI.

---

## 10. Graphe de dépendances (résumé)

| Module        | Importe (interne)                           |
|---------------|---------------------------------------------|
| `config`      | `portfolio`                                 |
| `data/`       | —                                           |
| `portfolio`   | —                                           |
| `volatility/` | `_validation`                               |
| `var/`        | `_validation`, `portfolio`, `volatility/`, `distributions` |
| `distributions`| `_validation`                              |
| `es`          | `_validation`, `distributions`, `portfolio`, `var/monte_carlo` |
| `stress/`     | `_validation`, `portfolio`, `var/`, `es`    |
| `backtest/`   | `backtest/_common` (interne au package)     |
| `reporting/`  | `backtest` (bornes traffic light), `pipeline` (daily_report), `stress` (types, TYPE_CHECKING) |
| `pipeline`    | `config`, `data`, `portfolio`, `var`, `es`, `stress`, `backtest` |
| `cli`         | `config`, `pipeline`, `reporting`           |
| `app/` (hors package) | `config`, `pipeline`, `var`, `es`, `backtest`, `reporting` |

Aucun cycle. Tests : un fichier par domaine (`tests/test_{data,portfolio,var,
var_conditional,ewma,garch,distributions,monte_carlo_student,es,es_backtest,
backtest,traffic_light,stress,report,config,cli,pipeline,daily_report,app,
etude_2020,etude_stress}.py`), 260 tests, aucun accès réseau (yfinance mocké
+ snapshot `data/cache/` — le dashboard est testé headless par
`streamlit.testing.v1.AppTest`).
