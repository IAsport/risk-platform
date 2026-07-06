# Snapshot de données de marché (figé)

- **Date du snapshot : 2026-07-06** (source yfinance, clôtures ajustées).
- Couverture : 2014-01-02 → 2026-07-02, 8 tickers du portefeuille de référence
  (`config/portfolio.yaml`) en devise locale + taux EURUSD, **plus la colonne
  `^STOXX50E`** (benchmark hors poids, ajoutée en brique 3 pour le scénario
  indiciel — fusion bornée aux dates existantes, lignes des 8 titres
  inchangées ; l'indice a des NaN sur ~57 dates où il ne cote pas, supprimés
  à la lecture quand on ne demande que lui).
- Rôle : rendre l'étude 2019–2021 (`notebooks/`), le test du résultat phare
  (`tests/test_etude_2020.py`) et la CI **rejouables offline et à l'identique**
  (SPEC.md B1.4). Les chiffres publiés dans le README sont calculés sur CE
  snapshot.
- Rafraîchir : supprimer `prices.csv` / `eurusd.csv` puis lancer
  `riskplatform` (cache write-through) — et mettre à jour cette date.
