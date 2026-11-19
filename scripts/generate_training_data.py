"""Generate graded training data for the learned ranker.

For each synthetic natural-language question we build a *pool* of candidate
SQL queries with graded relevance labels:

  grade 3 — answers the question correctly (gold + cosmetic variants)
  grade 2 — relevant but flawed (missing LIMIT/ORDER BY/filter)
  grade 1 — wrong aggregation / direction / status / period
  grade 0 — unrelated query (wrong intent)

The pool is what the ranker learns to re-order, evaluated at NDCG@5.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_to_sql_agent.config import Config  # noqa: E402
from text_to_sql_agent.templates import (  # noqa: E402
    TemplateSQLGenerator,
    extract_slots,
    normalize,
    tokenize,
    q_avg_order_value,
    q_discount_lost,
    q_orders_by_country,
    q_orders_by_month,
    q_orders_by_status,
    q_orders_in_period,
    q_products_never_ordered,
    q_repeat_customers,
    q_revenue_by_category,
    q_revenue_by_country,
    q_revenue_by_month,
    q_top_categories,
    q_top_customers,
    q_top_products,
    q_total_revenue,
)

_DATE_FRAG = re.compile(r"o\.order_date\s*>=\s*DATE\s*'[^']+'\s*AND\s*o\.order_date\s*<\s*DATE\s*'[^']+'")
_DATE_LITERAL = re.compile(r"DATE\s*'(\d{4})-(\d{2})-(\d{2})'")
_LIMIT_FRAG = re.compile(r"\s+LIMIT\s+\d+", re.IGNORECASE)
_ORDER_FRAG = re.compile(r"\s+ORDER BY\s+[^\n]+", re.IGNORECASE)
_ROUND_SUM = re.compile(r"ROUND\(\s*(SUM\([^)]*\))\s*,\s*\d+\s*\)", re.IGNORECASE)
_STATUS_FRAG = re.compile(r"o\.status\s*=\s*'[^']+'")
_CAT_COND = re.compile(r"c\.name\s*=\s*'[^']+'")

ALL_STATUSES = ("delivered", "shipped", "returned", "cancelled", "pending")

INTENT_HANDLERS = {
    "total_revenue": q_total_revenue,
    "revenue_by_category": q_revenue_by_category,
    "revenue_by_month": q_revenue_by_month,
    "revenue_by_country": q_revenue_by_country,
    "top_products": q_top_products,
    "top_categories": q_top_categories,
    "top_customers": q_top_customers,
    "avg_order_value": q_avg_order_value,
    "orders_by_status": q_orders_by_status,
    "orders_by_month": q_orders_by_month,
    "orders_by_country": q_orders_by_country,
    "repeat_customers": q_repeat_customers,
    "products_never_ordered": q_products_never_ordered,
    "discount_lost": q_discount_lost,
    "orders_in_period": q_orders_in_period,
}

PERIODS = (
    "", "en 2023", "en 2024", "en 2025", "en mars 2024", "en novembre 2023", "en janvier 2025",
    "en juillet 2024", "au cours de l'année 2024", "pendant l'année 2023", "au mois de mars 2024",
    "sur l'année 2025", "durant l'année 2024",
)
STATUS_RAW = ("", "livrées", "retournées", "annulées", "expédiées")
STATUS_SG = ("", "livrée", "retournée", "annulée", "expédiée")
CHANNEL = ("", "web", "mobile", "store")
CATS = ("", "Electronics", "Clothing", "Books", "Sports", "Beauty")
COUNTRIES = ("", "France", "Germany", "United States", "Spain")
TOPS = (3, 5, 10)

# intents: target question counts (sum = n_questions)
TARGETS = {
    "total_revenue": 55, "top_products": 55, "top_customers": 55,
    "avg_order_value": 55, "orders_in_period": 55,
    "revenue_by_category": 47, "revenue_by_month": 47, "revenue_by_country": 47,
    "top_categories": 46, "orders_by_status": 46, "orders_by_month": 46,
    "orders_by_country": 46, "repeat_customers": 46, "discount_lost": 46,
    "products_never_ordered": 8,
}

PHRASES: dict[str, list[str]] = {
    "total_revenue": [
        "Quel est le chiffre d'affaires total {period} ?",
        "Quel est le chiffre d'affaires total {status} {period} ?",
        "J'aimerais savoir combien de chiffre d'affaires total nous avons réalisé {period} {channel}.",
        "What is the total revenue {period} ?",
        "Quel montant total de revenus avons-nous généré {status} {period}, s'il vous plaît ?",
        "Peux-tu me dire combien de chiffre d'affaires total on a fait {period} ?",
    ],
    "revenue_by_category": [
        "Quel est le chiffre d'affaires par catégorie {period} ?",
        "J'aimerais la ventilation du revenu par catégorie {period}, s'il vous plaît.",
        "Quelles sont les ventes par catégorie {period} ?",
        "Sales by category {period}",
        "Pouvez-vous me dire le chiffre d'affaires par catégorie {status} {period} ?",
        "Comment se répartit le revenu par catégorie {channel} {period} ?",
    ],
    "revenue_by_month": [
        "Quel est le chiffre d'affaires mensuel {period} ?",
        "J'aimerais voir l'évolution du chiffre d'affaires par mois {period}.",
        "Quel est le revenu par mois {period} ?",
        "Revenue by month {period}",
        "Pouvez-vous me donner le chiffre d'affaires par mois {channel} {period} ?",
        "Dis-moi quel est le revenu par mois {period}, merci.",
    ],
    "revenue_by_country": [
        "Quel est le chiffre d'affaires par pays {period} ?",
        "Quelles sont les ventes par pays {period} ?",
        "Comment se répartissent les ventes par pays {period} ?",
        "Sales by country {period}",
        "Quel chiffre d'affaires par pays avons-nous réalisé {channel} {period} ?",
        "J'aimerais connaître les ventes par pays {period}, s'il vous plaît.",
    ],
    "top_products": [
        "Quels sont les {top_n} produits les plus vendus {period} {category} ?",
        "J'aimerais connaître le top {top_n} des produits par quantité vendue {period} {category}.",
        "Quels sont les meilleurs produits {period} {category} ?",
        "What are the top {top_n} products {period} {category} ?",
        "Quels sont les produits les plus achetés {period} {category}, merci ?",
        "Peux-tu me lister les {top_n} produits les plus vendus {category} {period} ?",
    ],
    "top_categories": [
        "Quelles sont les {top_n} catégories avec le plus de ventes {period} ?",
        "J'aimerais le top {top_n} des catégories par unités vendues {period}.",
        "Quelles sont les meilleures catégories {period} ?",
        "Quelles sont les {top_n} catégories les plus vendues {channel} {period} ?",
        "Top {top_n} des catégories les plus vendues {status} {period}",
        "Dis-moi quelles sont les catégories les plus vendues {period}, s'il te plaît.",
    ],
    "top_customers": [
        "Qui sont les {top_n} meilleurs clients {period} {country} ?",
        "J'aimerais le top {top_n} des clients par montant dépensé {period}.",
        "Quels sont les clients les plus dépensiers {period} ?",
        "Top {top_n} customers by spend {period} {country}",
        "Quels clients ont le plus dépensé {period} {country}, s'il vous plaît ?",
    ],
    "avg_order_value": [
        "Quel est le panier moyen {period} ?",
        "J'aimerais connaître la valeur moyenne d'une commande {status_sg} {period}.",
        "Quel est le montant moyen des commandes {period} ?",
        "What is the average order value {period} ?",
        "Peux-tu me dire quel est le panier moyen {period}, merci ?",
    ],
    "orders_by_status": [
        "Combien de commandes par statut {period} ?",
        "J'aimerais la répartition des commandes par statut {period}.",
        "How many orders by status {period} ?",
        "Combien de commandes par statut {channel} {period} ?",
        "Peux-tu me donner la répartition des commandes par statut {period}, s'il te plaît ?",
    ],
    "orders_by_month": [
        "Combien de commandes par mois {period} ?",
        "Peux-tu me montrer l'évolution du nombre de commandes par mois {period} ?",
        "How many orders by month {period} ?",
        "Combien de commandes par mois {channel} {period} ?",
        "J'aimerais voir le nombre de commandes par mois {period}, merci.",
    ],
    "orders_by_country": [
        "Combien de commandes par pays {period} ?",
        "Comment se répartissent les commandes par pays {period} ?",
        "How many orders by country {period} ?",
        "Combien de commandes par pays {channel} {period} ?",
        "Peux-tu me dire combien de commandes par pays il y a eu {period} ?",
    ],
    "repeat_customers": [
        "Combien de clients ont passé plus d'une commande {period} ?",
        "J'aimerais savoir combien de clients sont des clients répétés {period}.",
        "How many repeat customers are there {period} ?",
        "Combien de clients ont passé plus d'une commande {channel} {period} ?",
        "Dis-moi combien de clients ont commandé plus d'une fois {period}, s'il te plaît.",
    ],
    "products_never_ordered": [
        "Quels sont les produits jamais commandés ?",
        "Quels produits n'ont jamais été vendus ?",
        "Which products were never ordered ?",
        "Quels produits du catalogue n'ont jamais été commandés ?",
        "Liste des produits sans aucune vente",
        "Which catalog products have never been sold ?",
        "Quels produits n'ont jamais été achetés ?",
        "Which products have never been purchased ?",
    ],
    "discount_lost": [
        "Quel est le montant total des remises accordées {period} ?",
        "J'aimerais savoir combien de remises ont été accordées {period}.",
        "How much discount was given {period} ?",
        "Quel est le montant des remises sur les commandes {status_raw} {period} ?",
        "Quel est le montant total des remises {channel} {period} ?",
        "Peux-tu me dire combien de remises nous avons accordées {period}, merci ?",
    ],
    "orders_in_period": [
        "Combien de commandes ont été passées {period} ?",
        "J'aimerais savoir combien de commandes {status_raw} ont été passées {period}.",
        "How many orders were placed {period} ?",
        "Combien de commandes ont été passées {channel} {period} ?",
        "Peux-tu me dire combien de commandes il y a eu {period}, s'il te plaît ?",
    ],
}


def build_questions(targets: dict[str, int], seed: int = 42) -> list[tuple[str, str]]:
    """Return distinct (question, intent) pairs, all parseable by the templates."""
    rng = np.random.default_rng(seed)
    questions: list[tuple[str, str]] = []
    seen: set[str] = set()

    for intent, target in targets.items():
        phrases = PHRASES[intent]
        attempts = 0
        while len([q for q, i in questions if i == intent]) < target and attempts < 3000:
            attempts += 1
            phrase = phrases[attempts % len(phrases)]
            slots = {
                "period": PERIODS[rng.integers(len(PERIODS))],
                "status": "", "status_raw": "", "status_sg": "", "channel": "",
                "category": "", "country": "", "top_n": None,
            }
            if intent in ("total_revenue", "revenue_by_category", "top_categories", "discount_lost"):
                raw = STATUS_RAW[rng.integers(len(STATUS_RAW))]
                slots["status"] = f"des commandes {raw}" if raw else ""
                slots["status_raw"] = raw
            if intent in ("total_revenue", "revenue_by_category", "revenue_by_month", "revenue_by_country",
                          "top_categories", "orders_by_status", "orders_by_month", "orders_by_country",
                          "repeat_customers", "discount_lost", "orders_in_period"):
                ch = CHANNEL[rng.integers(len(CHANNEL))]
                slots["channel"] = f"via le canal {ch}" if ch else ""
            if intent == "avg_order_value":
                sg = STATUS_SG[rng.integers(len(STATUS_SG))]
                slots["status_sg"] = sg
            if intent == "top_products":
                cat = CATS[rng.integers(len(CATS))]
                slots["category"] = f"dans la catégorie {cat}" if cat else ""
                slots["top_n"] = TOPS[rng.integers(len(TOPS))]
            if intent == "top_customers":
                country = COUNTRIES[rng.integers(len(COUNTRIES))]
                slots["country"] = f"en {country}" if country else ""
                slots["top_n"] = TOPS[rng.integers(len(TOPS))]
            if intent == "top_categories":
                slots["top_n"] = TOPS[rng.integers(len(TOPS))]

            question = phrase.format(**slots)
            question = re.sub(r"\s{2,}", " ", question).replace(" ?", "?").strip()
            question = re.sub(r"\s+\?", "?", question)
            if question.count("?") > 1:
                question = question.replace("?", "", question.count("?") - 1)
            key = normalize(question)
            if key in seen:
                continue
            gens = TemplateSQLGenerator().generate(question, with_variants=False)
            if not gens or gens[0].intent != intent:
                continue
            seen.add(key)
            questions.append((question, intent))

    print(f"Généré {len(questions)} questions sur {sum(targets.values())} cibles")
    for intent, target in targets.items():
        n = len([q for q, i in questions if i == intent])
        if n < target:
            print(f"  ! {intent}: {n}/{target}")
    return questions


# ------------------------------------------------------------ mutations --

def _mutate(sql: str, rng: np.random.Generator, gold: str) -> list[tuple[str, int]]:
    """Apply applicable mutations to a gold SQL; returns (sql, grade)."""
    out: list[tuple[str, int]] = []

    if _LIMIT_FRAG.search(sql):
        out.append((_LIMIT_FRAG.sub("", sql), 2))
    if _ORDER_FRAG.search(sql):
        out.append((_ORDER_FRAG.sub("", sql), 2))
    if _DATE_FRAG.search(sql):
        dropped = _DATE_FRAG.sub("", sql)
        dropped = re.sub(r"\nWHERE\s*$", "", dropped)  # period was the only condition
        out.append((dropped, 2))
    if _CAT_COND.search(sql):
        out.append((_CAT_COND.sub("1=1", sql), 2))

    if "SUM(" in sql.upper():
        out.append((re.sub(r"SUM\(", "COUNT(", sql, count=1, flags=re.IGNORECASE), 1))
    if " ORDER BY " in sql.upper():
        if " DESC" in sql.upper():
            out.append((re.sub(r"DESC\b", "ASC", sql, count=1, flags=re.IGNORECASE), 1))
        else:
            out.append((re.sub(r"ORDER BY\s+([^\n]+)", r"ORDER BY \1 DESC", sql, count=1, flags=re.IGNORECASE), 1))
    if _DATE_LITERAL.search(sql):
        out.append((_DATE_LITERAL.sub(lambda m: f"DATE '{int(m.group(1)) - 1}-{m.group(2)}-{m.group(3)}'", sql), 1))
    if _STATUS_FRAG.search(sql):
        current = _STATUS_FRAG.search(sql).group(0)
        other = next(s for s in ALL_STATUSES if f"'{s}'" not in current)
        out.append((_STATUS_FRAG.sub(f"o.status = '{other}'", sql), 1))
    if "JOIN orders o" in sql and "WHERE" in sql.upper():
        cond = rng.choice(["returned", "cancelled", "pending"])
        out.append((re.sub(r"WHERE ", f"WHERE o.status = '{cond}' AND ", sql, count=1, flags=re.IGNORECASE), 1))

    # cosmetic grade-3 variants
    if _ROUND_SUM.search(sql):
        out.append((_ROUND_SUM.sub(r"\1", sql), 3))
    else:
        out.append((re.sub(r"\bAS\s+([a-z_]+)", r"\1", sql, count=1, flags=re.IGNORECASE), 3))

    gold_norm = normalize(gold)
    uniq: list[tuple[str, int]] = []
    for s, g in out:
        if normalize(s) != gold_norm and not any(normalize(s) == normalize(u) for u, _ in uniq):
            uniq.append((s, g))
    return uniq


def build_pool(question: str, intent: str, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Return candidate pool [(sql, grade)] with gold first."""
    handler = INTENT_HANDLERS[intent]
    slot = extract_slots(question)
    gold = handler(slot)[0].sql
    pool: list[tuple[str, int]] = [(gold, 3)]

    for sql, grade in _mutate(gold, rng, gold):
        if grade == 3 and len([p for p in pool if p[1] == 3]) < 2:
            pool.append((sql, grade))
        elif grade == 2 and len([p for p in pool if p[1] == 2]) < 2:
            pool.append((sql, grade))
        elif grade == 1 and len([p for p in pool if p[1] == 1]) < 3:
            pool.append((sql, grade))

    # grade-0 distractors: SQL answering a different question, chosen among the
    # most lexically similar ones (same tables/period) so that only the learned
    # features (execution, structure) can tell them apart.
    others = [h for key, h in INTENT_HANDLERS.items() if key != intent]
    seen = {normalize(s) for s, _ in pool}
    candidates0 = []
    for handler in others:
        sql = handler(slot)[0].sql
        if normalize(sql) not in seen:
            cov = len(tokenize(question) & tokenize(sql)) / max(1, len(tokenize(question)))
            tables = {"orders", "order_items", "products", "categories", "customers"}
            tables_gold = tables & set(tokenize(gold))
            tables_sql = tables & set(tokenize(sql))
            tsim = len(tables_gold & tables_sql) / max(1, len(tables_gold | tables_sql))
            candidates0.append((cov + tsim, sql))
    candidates0.sort(reverse=True)
    top_similar = candidates0[:10]
    rng.shuffle(top_similar)
    for _, sql in top_similar:
        if len([p for p in pool if p[1] == 0]) >= 7:
            break
        if normalize(sql) not in seen:
            seen.add(normalize(sql))
            pool.append((sql, 0))
    return pool


def build_training_frame(n_questions: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    questions = build_questions(TARGETS, seed)
    rng.shuffle(questions)
    questions = questions[:n_questions]

    rows: list[dict] = []
    for qid, (question, intent) in enumerate(questions):
        for sql, grade in build_pool(question, intent, rng):
            rows.append({"question_id": qid, "question": question, "sql": sql, "grade": grade, "intent": intent})

    frame = pd.DataFrame(rows)
    print(f"Questions: {frame['question_id'].nunique()} | candidates: {len(frame)} | "
          f"pool moyen: {len(frame) / frame['question_id'].nunique():.1f}")
    print(frame.groupby("grade")["sql"].count().rename("n").to_string())
    return frame


def main(cfg: Config | None = None) -> None:
    from text_to_sql_agent.config import load_config
    cfg = cfg or load_config()
    out = cfg.resolve("training_parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build_training_frame(cfg.n_questions, cfg.database_seed)
    frame.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
