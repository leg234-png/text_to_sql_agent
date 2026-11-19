"""The offline template generator must map FR/EN questions to valid SQL."""
from __future__ import annotations

import pytest

from text_to_sql_agent.executor import execute
from text_to_sql_agent.templates import TemplateSQLGenerator

GEN = TemplateSQLGenerator()

CASES = [
    # (question, intent attendu, fragment SQL attendu, prédicat sur le résultat)
    ("Quel est le chiffre d'affaires total ?", "total_revenue", "SUM", lambda df: len(df) == 1 and df.iloc[0, 0] > 0),
    ("Quel est le chiffre d'affaires total en 2024 ?", "total_revenue", "2024-01-01", lambda df: len(df) == 1),
    ("Top 5 des produits par quantité vendue en 2024", "top_products", "LIMIT 5", lambda df: len(df) == 5),
    ("Quels sont les 10 produits les plus vendus dans la catégorie Electronics ?", "top_products", "Electronics", lambda df: len(df) == 10),
    ("Combien de commandes par statut ?", "orders_by_status", "GROUP BY", lambda df: len(df) == 5),
    ("Quel est le chiffre d'affaires par catégorie en mars 2024 ?", "revenue_by_category", "2024-03-01", lambda df: 0 < len(df) <= 8),
    ("Quel est le chiffre d'affaires mensuel ?", "revenue_by_month", "strftime", lambda df: len(df) > 10),
    ("Quelles sont les ventes par pays en 2024 ?", "revenue_by_country", "cu.country", lambda df: 0 < len(df) <= 8),
    ("Combien de clients ont passé plus d'une commande ?", "repeat_customers", "HAVING COUNT(*) > 1", lambda df: len(df) == 1),
    ("Quel est le panier moyen ?", "avg_order_value", "AVG", lambda df: len(df) == 1),
    ("Quels sont les produits jamais commandés ?", "products_never_ordered", "LEFT JOIN", lambda df: len(df) == 10),
    ("Quel est le montant total des remises accordées en 2025 ?", "discount_lost", "discount", lambda df: len(df) == 1),
    ("Combien de commandes ont été passées en mars 2024 ?", "orders_in_period", "COUNT", lambda df: len(df) == 1),
    ("What is the total revenue in 2024 ?", "total_revenue", "2024-01-01", lambda df: len(df) == 1),
    ("How many orders by status ?", "orders_by_status", "GROUP BY", lambda df: len(df) == 5),
]


@pytest.mark.parametrize("question,intent,fragment,check", CASES)
def test_template_generation(con, question, intent, fragment, check):
    gens = GEN.generate(question, with_variants=False)
    assert gens, f"aucun SQL généré pour : {question}"
    g = gens[0]
    assert g.intent == intent, f"{question} → {g.intent} (attendu {intent})"
    assert fragment in g.sql
    res = execute(con, g.sql)
    assert res.error is None, f"SQL invalide: {res.error}\n{g.sql}"
    assert check(res.df)


def test_top_products_are_sorted_desc(con):
    gens = GEN.generate("Quels sont les 5 produits les plus vendus ?", with_variants=False)
    res = execute(con, gens[0].sql)
    units = res.df["units_sold"].tolist()
    assert units == sorted(units, reverse=True)


def test_unknown_question_returns_nothing():
    assert GEN.generate("Peux-tu me raconter une blague ?") == []
