"""Offline, template-based NL→SQL generator (works without any LLM key).

Parses French and English natural-language questions against a fixed set of
business intents, extracts entities (period, category, country, status, top-N)
and emits canonical SQL plus a few valid variants. The variants feed the
candidate pool that the learned ranker re-orders.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from .schema import CATEGORIES, CHANNELS, COUNTRIES, REVENUE_EXPR, STATUSES


@dataclass
class GeneratedSQL:
    sql: str
    intent: str
    confidence: float
    origin: str = "template"


@dataclass
class Slot:
    """Entities extracted from a question."""

    period_sql: str | None = None        # WHERE fragment on o.order_date
    period_label: str | None = None
    category: str | None = None
    country: str | None = None
    status: str | None = None
    channel: str | None = None
    top_n: int | None = None

    def n_entities(self) -> int:
        return sum(
            v is not None
            for v in (self.period_sql, self.category, self.country, self.status, self.channel, self.top_n)
        )


MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

STATUS_MAP = {
    "livré": "delivered", "livrée": "delivered", "livrées": "delivered", "livree": "delivered",
    "delivered": "delivered", "livrés": "delivered", "livres": "delivered",
    "expédié": "shipped", "expédiée": "shipped", "expédiées": "shipped", "expedie": "shipped", "shipped": "shipped",
    "retourné": "returned", "retournée": "returned", "retournées": "returned", "retourne": "returned", "returned": "returned",
    "annulé": "cancelled", "annulée": "cancelled", "annulées": "cancelled", "annule": "cancelled", "cancelled": "cancelled",
    "en attente": "pending", "pending": "pending",
}

CHANNEL_MAP = {"web": "web", "mobile": "mobile", "magasin": "store", "store": "store", "boutique": "store"}


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.lower().strip())


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize(text)))


def extract_slots(question: str) -> Slot:
    q = normalize(question)
    slot = Slot()

    # top N
    m = re.search(r"\b(top|meilleurs?|meilleures?|premiers?|premieres?)\s+(\d{1,2})\b", q)
    if m:
        slot.top_n = int(m.group(2))
    else:
        m = re.search(
            r"\b(\d{1,2})\s+(meilleurs?|meilleures?|premiers?|premieres?|plus\s+vendus?|produits|clients|categories)\b", q
        )
        if m:
            slot.top_n = int(m.group(1))

    # period
    for name, num in MONTHS.items():
        if name in q:
            m = re.search(rf"{name}\s+(\d{{4}})", q)
            if m:
                year = int(m.group(1))
                slot.period_sql = (
                    f"o.order_date >= DATE '{year}-{num:02d}-01' "
                    f"AND o.order_date < DATE '{year + 1 if num == 12 else year}-{(num % 12) + 1:02d}-01'"
                )
                slot.period_label = f"{name} {year}"
            break
    if slot.period_sql is None:
        m = re.search(r"\b(en|in|de l'annee|annee)\s+(\d{4})\b", q) or re.search(r"\b(\d{4})\b", q)
        if m:
            year = int(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
            if 2020 <= year <= 2026:
                slot.period_sql = (
                    f"o.order_date >= DATE '{year}-01-01' AND o.order_date < DATE '{year + 1}-01-01'"
                )
                slot.period_label = str(year)

    # category / country / status / channel
    for cat in CATEGORIES:
        if normalize(cat).replace("&", "and") in q or normalize(cat) in q:
            slot.category = cat
            break
    for country in COUNTRIES:
        if normalize(country) in q:
            slot.country = country
            break
    for key, value in STATUS_MAP.items():
        if key in q:
            slot.status = value
            break
    for key, value in CHANNEL_MAP.items():
        if key in q:
            slot.channel = value
            break
    return slot


def _order_where(slot: Slot, alias: str = "o") -> str:
    conds = []
    if slot.period_sql:
        conds.append(slot.period_sql)
    if slot.status:
        conds.append(f"{alias}.status = '{slot.status}'")
    if slot.channel:
        conds.append(f"{alias}.channel = '{slot.channel}'")
    if slot.country:
        conds.append(f"{alias}.customer_id IN (SELECT customer_id FROM customers WHERE country = '{slot.country}')")
    return " AND ".join(conds)


def _where(slot: Slot, alias: str = "o", include_category: bool = True) -> str:
    conds = []
    order_where = _order_where(slot, alias)
    if order_where:
        conds.append(order_where)
    if include_category and slot.category:
        conds.append("c.name = '%s'" % slot.category)
    return " AND ".join(conds)


def _limit(slot: Slot) -> str:
    return f" LIMIT {slot.top_n}" if slot.top_n else ""


def _filtered_order_items(slot: Slot) -> str:
    """FROM order_items oi JOIN orders o (+ optional products/categories/customers)."""
    joins = ["FROM order_items oi", "JOIN orders o ON o.order_id = oi.order_id"]
    if slot.category:
        joins += ["JOIN products p ON p.product_id = oi.product_id", "JOIN categories c ON c.category_id = p.category_id"]
    where = _where(slot, "o")
    return "\n".join(joins) + (f"\nWHERE {where}" if where else "")


# ---------------------------------------------------------------- intents --

def q_total_revenue(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    base = (
        f"SELECT ROUND(SUM({REVENUE_EXPR}), 2) AS revenue\n"
        "FROM order_items oi\nJOIN orders o ON o.order_id = oi.order_id"
    )
    sql = base + (f"\nWHERE {where}" if where else "")
    return [GeneratedSQL(sql, "total_revenue", 0.62 + 0.06 * slot.n_entities())]


def q_revenue_by_category(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT c.name AS category, ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue\n"
        "FROM order_items oi\n"
        "JOIN orders o ON o.order_id = oi.order_id\n"
        "JOIN products p ON p.product_id = oi.product_id\n"
        "JOIN categories c ON c.category_id = p.category_id"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY revenue DESC"
    )
    return [GeneratedSQL(sql, "revenue_by_category", 0.62 + 0.06 * slot.n_entities())]


def q_revenue_by_month(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT strftime(o.order_date, '%Y-%m') AS month, ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue\n"
        "FROM order_items oi\nJOIN orders o ON o.order_id = oi.order_id"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY 1"
    )
    return [GeneratedSQL(sql, "revenue_by_month", 0.62 + 0.06 * slot.n_entities())]


def q_revenue_by_country(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT cu.country, ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue\n"
        "FROM order_items oi\n"
        "JOIN orders o ON o.order_id = oi.order_id\n"
        "JOIN customers cu ON cu.customer_id = o.customer_id"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY revenue DESC"
    )
    return [GeneratedSQL(sql, "revenue_by_country", 0.62 + 0.06 * slot.n_entities())]


def q_top_products(slot: Slot) -> list[GeneratedSQL]:
    base = (
        "SELECT p.name AS product, SUM(oi.quantity) AS units_sold, "
        f"ROUND(SUM({REVENUE_EXPR}), 2) AS revenue\n"
        "FROM order_items oi\n"
        "JOIN orders o ON o.order_id = oi.order_id\n"
        "JOIN products p ON p.product_id = oi.product_id"
        + ("" if not slot.category else "\nJOIN categories c ON c.category_id = p.category_id")
    )
    where = _where(slot, "o", include_category=bool(slot.category))
    sql = base + (f"\nWHERE {where}" if where else "") + "\nGROUP BY 1\nORDER BY units_sold DESC" + _limit(slot)
    return [GeneratedSQL(sql, "top_products", 0.60 + 0.06 * slot.n_entities())]


def q_top_categories(slot: Slot) -> list[GeneratedSQL]:
    base = (
        "SELECT c.name AS category, SUM(oi.quantity) AS units_sold, "
        f"ROUND(SUM({REVENUE_EXPR}), 2) AS revenue\n"
        "FROM order_items oi\n"
        "JOIN orders o ON o.order_id = oi.order_id\n"
        "JOIN products p ON p.product_id = oi.product_id\n"
        "JOIN categories c ON c.category_id = p.category_id"
    )
    where = _where(slot, "o", include_category=False)
    sql = base + (f"\nWHERE {where}" if where else "") + "\nGROUP BY 1\nORDER BY units_sold DESC" + _limit(slot)
    return [GeneratedSQL(sql, "top_categories", 0.60 + 0.06 * slot.n_entities())]


def q_top_customers(slot: Slot) -> list[GeneratedSQL]:
    base = (
        "SELECT cu.name AS customer, COUNT(o.order_id) AS nb_orders, ROUND(SUM(o.total_amount), 2) AS total_spent\n"
        "FROM orders o\nJOIN customers cu ON cu.customer_id = o.customer_id"
    )
    where = _order_where(slot)
    sql = base + (f"\nWHERE {where}" if where else "") + "\nGROUP BY 1\nORDER BY total_spent DESC" + _limit(slot)
    return [GeneratedSQL(sql, "top_customers", 0.60 + 0.06 * slot.n_entities())]


def q_avg_order_value(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT ROUND(AVG(o.total_amount), 2) AS average_order_value\nFROM orders o"
        + (f"\nWHERE {where}" if where else "")
    )
    return [GeneratedSQL(sql, "avg_order_value", 0.62 + 0.06 * slot.n_entities())]


def q_orders_by_status(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT o.status, COUNT(*) AS nb_orders\nFROM orders o"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY nb_orders DESC"
    )
    return [GeneratedSQL(sql, "orders_by_status", 0.62 + 0.06 * slot.n_entities())]


def q_orders_by_month(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT strftime(o.order_date, '%Y-%m') AS month, COUNT(*) AS nb_orders\nFROM orders o"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY 1"
    )
    return [GeneratedSQL(sql, "orders_by_month", 0.62 + 0.06 * slot.n_entities())]


def q_orders_by_country(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT cu.country, COUNT(*) AS nb_orders\n"
        "FROM orders o\nJOIN customers cu ON cu.customer_id = o.customer_id"
        + (f"\nWHERE {where}" if where else "")
        + "\nGROUP BY 1\nORDER BY nb_orders DESC"
    )
    return [GeneratedSQL(sql, "orders_by_country", 0.62 + 0.06 * slot.n_entities())]


def q_repeat_customers(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT COUNT(*) AS repeat_customers\nFROM (\n"
        "  SELECT o.customer_id\n  FROM orders o"
        + (f"\n  WHERE {where}" if where else "")
        + "\n  GROUP BY o.customer_id\n  HAVING COUNT(*) > 1\n) t"
    )
    return [GeneratedSQL(sql, "repeat_customers", 0.62 + 0.06 * slot.n_entities())]


def q_products_never_ordered(slot: Slot) -> list[GeneratedSQL]:
    sql = (
        "SELECT p.product_id, p.name\nFROM products p\n"
        "LEFT JOIN order_items oi ON oi.product_id = p.product_id\n"
        "WHERE oi.item_id IS NULL\nORDER BY p.product_id"
    )
    return [GeneratedSQL(sql, "products_never_ordered", 0.66)]


def q_discount_lost(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = (
        "SELECT ROUND(SUM(oi.quantity * oi.unit_price * oi.discount), 2) AS discount_given\n"
        "FROM order_items oi\nJOIN orders o ON o.order_id = oi.order_id"
        + (f"\nWHERE {where}" if where else "")
    )
    return [GeneratedSQL(sql, "discount_lost", 0.62 + 0.06 * slot.n_entities())]


def q_orders_in_period(slot: Slot) -> list[GeneratedSQL]:
    where = _order_where(slot)
    sql = "SELECT COUNT(*) AS nb_orders\nFROM orders o" + (f"\nWHERE {where}" if where else "")
    return [GeneratedSQL(sql, "orders_in_period", 0.62 + 0.06 * slot.n_entities())]


INTENTS: list[tuple[tuple[str, ...], Callable[[Slot], list[GeneratedSQL]]]] = [
    (("chiffre d'affaires total", "revenue total", "revenu global", "total sales", "total revenue", "ventes totales", "montant total de revenus"), q_total_revenue),
    (("revenu par categorie", "chiffre d'affaires par categorie", "sales by category", "revenue by category", "ventes par categorie"), q_revenue_by_category),
    (("revenu mensuel", "chiffre d'affaires mensuel", "revenu par mois", "chiffre d'affaires par mois", "sales by month", "revenue by month", "ca mensuel", "ca par mois"), q_revenue_by_month),
    (("ventes par pays", "revenu par pays", "chiffre d'affaires par pays", "sales by country", "revenue by country", "ca par pays"), q_revenue_by_country),
    (("produits les plus vendus", "meilleurs produits", "top produits", "best-selling products", "top products", "best selling products", "produits par quantite vendue", "produits les plus achetes", "quantite vendue"), q_top_products),
    (("categories les plus vendues", "meilleures categories", "top categories", "plus de ventes", "categories par", "unites vendues"), q_top_categories),
    (("meilleurs clients", "top clients", "clients les plus depensiers", "top customers", "best customers", "clients par montant depense", "montant depense", "le plus depense"), q_top_customers),
    (("panier moyen", "valeur moyenne des commandes", "average order value", "commande moyenne", "montant moyen des commandes", "valeur moyenne d'une commande", "moyenne des commandes"), q_avg_order_value),
    (("commandes par statut", "statut des commandes", "orders by status"), q_orders_by_status),
    (("commandes par mois", "nombre de commandes par mois", "orders by month", "nb de commandes par mois"), q_orders_by_month),
    (("commandes par pays", "orders by country", "nombre de commandes par pays"), q_orders_by_country),
    (("clients repetes", "clients qui ont commande plus d'une fois", "repeat customers", "clients fideles", "plus d'une commande", "commande plus d'une fois"), q_repeat_customers),
    (("produits jamais commandes", "produits jamais vendus", "products never ordered", "produits sans vente", "jamais commandes", "jamais vendus", "jamais vendu", "jamais achetes", "never ordered", "never been sold", "never been purchased", "sans aucune vente", "jamais ete"), q_products_never_ordered),
    (("remises accordees", "montant des remises", "discount given", "revenu perdu a cause des remises", "discounts"), q_discount_lost),
    (("combien de commandes", "nombre de commandes", "how many orders", "nb de commandes"), q_orders_in_period),
]


class TemplateSQLGenerator:
    """Rule-based SQL generator used offline and for candidate expansion."""

    def generate(self, question: str, with_variants: bool = True) -> list[GeneratedSQL]:
        q = normalize(question)
        slot = extract_slots(question)
        out: list[GeneratedSQL] = []
        for keywords, handler in INTENTS:
            if any(kw in q for kw in keywords):
                out.extend(handler(slot))
                break
        if not out:
            return []
        if with_variants:
            out = self._variants(out, slot)
        return out

    # -- variants: same intent, slightly different (valid) SQL ---------------
    def _variants(self, gens: list[GeneratedSQL], slot: Slot) -> list[GeneratedSQL]:
        extra: list[GeneratedSQL] = []
        for g in gens:
            sql = g.sql
            if g.intent == "top_products":
                extra.append(GeneratedSQL(
                    sql.replace("ORDER BY units_sold DESC", "ORDER BY revenue DESC"),
                    g.intent, max(0.3, g.confidence - 0.18), "template_variant"))
            elif g.intent == "total_revenue":
                extra.append(GeneratedSQL(
                    "SELECT ROUND(SUM(o.total_amount), 2) AS revenue FROM orders o"
                    + (f"\nWHERE {_order_where(slot)}" if _order_where(slot) else ""),
                    g.intent, max(0.3, g.confidence - 0.15), "template_variant"))
            elif g.intent == "revenue_by_month":
                extra.append(GeneratedSQL(
                    sql.replace("strftime(o.order_date, '%Y-%m')", "strftime(o.order_date, '%m-%Y')"),
                    g.intent, max(0.3, g.confidence - 0.2), "template_variant"))
            elif g.intent == "revenue_by_country":
                extra.append(GeneratedSQL(
                    sql.replace("ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue",
                                "ROUND(SUM(o.total_amount), 2) AS revenue")
                    .replace("FROM order_items oi\nJOIN orders o ON o.order_id = oi.order_id\n", "FROM orders o\n"),
                    g.intent, max(0.3, g.confidence - 0.15), "template_variant"))
        return gens + extra
