PY := .venv/bin/python
Q ?= "Quels sont les 10 produits les plus vendus en 2024 ?"

.PHONY: help data train test ask clean

help:
	@echo "make data    — (re)construire la base DuckDB (81 000 lignes)"
	@echo "make train   — générer les données + entraîner le ranker (NDCG@5)"
	@echo "make test    — lancer la suite pytest"
	@echo "make ask Q=\"...\" — poser une question à l'agent"
	@echo "make clean   — supprimer les artefacts régénérables"

data:
	$(PY) scripts/build_database.py

train:
	$(PY) scripts/generate_training_data.py
	$(PY) scripts/train_ranker.py

test:
	$(PY) -m pytest

ask:
	$(PY) -m text_to_sql_agent.cli $(Q)

clean:
	rm -f data/*.duckdb data/*.parquet
