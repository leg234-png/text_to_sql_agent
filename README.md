# Agent Text-to-SQL avec ranking appris

Agent LLM qui traduit des questions en langage naturel (français / anglais) en requêtes **SQL**
sur un entrepôt e-commerce **DuckDB de 81 000 lignes**, puis **classe les requêtes candidates avec un
ranker appris** (LightGBM LambdaRank) avant d'exécuter la meilleure.

| Métrique (jeu de test, 3 splits) | TF-IDF (baseline) | Ranker appris |
|---|---|---|
| **NDCG@5** | 0.4237 ± 0.0246 | **0.8962 ± 0.0007** |
| Réponse parfaite en #1 | — | 131/140 groupes |

Le ranker appris multiplie le NDCG@5 par **×2.1** par rapport à la similarité lexicale seule.
Modèle entraîné inclus (`models/ranker.txt`) : l'agent fonctionne immédiatement, sans clé API.

---

## Fonctionnement

```
question en langage naturel
        │
        ▼
┌───────────────────────────┐   LLM (OpenAI-compatible, optionnel)
│  Génération de candidats  │   + générateur à templates hors-ligne
│  (pool de requêtes SQL)   │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐   28 features : accord lexical, structure SQL,
│  Exécution sûre (DuckDB)  │   formes des résultats, montants, confiance…
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐   LightGBM LambdaRank (label_gain 0/1/3/7)
│  Ranking appris (NDCG@5)  │   → meilleure requête sélectionnée
└───────────────────────────┘
        │
        ▼
   réponse (tableau + SQL)
```

1. **Candidats** : le générateur à templates (hors-ligne, FR/EN, 15 intentions métier) produit le
   pool de base ; si `OPENAI_API_KEY` est défini, un LLM ajoute ses propres candidats.
2. **Features** : chaque candidat est exécuté sur la base ; on calcule 28 features
   (chevauchement lexical, structure, agrégations, résultat, exécution, confiance).
3. **Ranking** : un modèle LightGBM LambdaRank, entraîné sur 700 questions synthétiques
   (8 773 candidats, pertinence graduée 0–3), réordonne le pool. Le top-1 valide est exécuté.

## Démarrage rapide

```bash
git clone https://github.com/leg234-png/text_to_sql_agent.git
cd text_to_sql_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. Construire la base (déterministe, seed=42, 81 000 lignes dans order_items)
python scripts/build_database.py

# 2. Poser une question (le modèle de ranking est déjà inclus)
text2sql "Quels sont les 10 produits les plus vendus en 2024 ?"
text2sql "Quel est le chiffre d'affaires par catégorie en mars 2024 ?" --explain
text2sql "What is the total revenue in 2024 ?"
text2sql "Combien de commandes ont été passées en mars 2024 ?" --json
```

Sortie typique :

```
Question : Quels sont les 10 produits les plus vendus en 2024 ?
Ranker   : learned (LightGBM LambdaRank)

SQL exécuté :
SELECT p.name AS product, SUM(oi.quantity) AS units_sold, ROUND(SUM(...), 2) AS revenue
FROM order_items oi JOIN orders o ON ... JOIN products p ON ...
WHERE o.order_date >= DATE '2024-01-01' AND o.order_date < DATE '2025-01-01'
GROUP BY 1 ORDER BY revenue DESC LIMIT 10

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ product      ┃ units_sold ┃ revenue   ┃
...
```

`--explain` affiche le classement complet des candidats (score, origine, validité) ;
`--json` produit une sortie JSON.

### Utiliser un vrai LLM (optionnel)

Sans clé API, le générateur à templates assure seul le pool de candidats — l'agent reste
entièrement fonctionnel. Pour brancher un LLM :

```bash
export OPENAI_API_KEY=sk-...          # n'importe quel endpoint compatible OpenAI
export T2S_LLM_MODEL=gpt-4o-mini      # optionnel
export T2S_LLM_BASE_URL=...           # optionnel (Ollama, vLLM, Azure…)
```

## Rejouer l'entraînement du ranker

```bash
python scripts/generate_training_data.py   # 700 questions, pools gradués 0-3
python scripts/train_ranker.py             # 3 splits 80/20, NDCG@5, sauvegarde du modèle
```

- **Données d'entraînement** : pour chaque question, un pool de ~12 requêtes candidates :
  grade 3 = réponse correcte (+ variante cosmétique), grade 2 = filtre/LIMIT/ORDER BY manquant,
  grade 1 = mauvaise agrégation / direction / période / statut, grade 0 = autre intention
  (sélectionnée parmi les plus similaires lexicalement, donc trompeuse).
- **Baseline** : similarité cosinus TF-IDF (1–2 grammes) question ↔ SQL, entraînée sur le split train.
- **Modèle appris** : `LGBMRanker(objective=lambdarank, label_gain=[0,1,3,7])`, 500 arbres.
- Métriques détaillées dans [`models/metrics.json`](models/metrics.json).

## La base de données

Base DuckDB synthétique mais réaliste, régénérable à l'identique (seed 42) :

| Table | Lignes | Description |
|---|---|---|
| `categories` | 8 | catégories produits |
| `products` | 2 500 | produits, prix, coût |
| `customers` | 8 100 | clients, pays, segment |
| `orders` | 21 600 | commandes, statut, canal, montant |
| `order_items` | **81 000** | lignes de commande (table de faits) |

Revenu d'une ligne = `quantity × unit_price × (1 − discount)`.

## Structure du projet

```
├── src/text_to_sql_agent/
│   ├── agent.py        # orchestration : candidats → exécution → ranking → réponse
│   ├── templates.py    # générateur SQL hors-ligne FR/EN (15 intentions)
│   ├── llm.py          # client LLM OpenAI-compatible (optionnel)
│   ├── executor.py     # exécution sûre DuckDB
│   ├── features.py     # 28 features de ranking
│   ├── ranking.py      # NDCG, baseline TF-IDF, LambdaRank (LightGBM)
│   ├── database.py     # données synthétiques + moteur DuckDB
│   └── cli.py          # CLI `text2sql`
├── scripts/
│   ├── build_database.py
│   ├── generate_training_data.py
│   └── train_ranker.py
├── models/             # ranker.txt (modèle appris), metrics.json, feature_names.json
├── tests/              # 35 tests (pytest)
├── config.yaml
└── Makefile            # make data / train / test / ask Q="..."
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

La suite couvre la cohérence de la base (81 000 lignes, montants, déterminisme), la génération
SQL FR/EN, les métriques NDCG, l'entraînement du ranker et l'agent de bout en bout.

## Licence

MIT — voir [LICENSE](LICENSE).
