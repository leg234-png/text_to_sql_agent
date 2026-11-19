"""E-commerce warehouse schema (5 tables, 81,000+ rows in the fact table)."""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id BIGINT PRIMARY KEY,
    name        VARCHAR
);

CREATE TABLE IF NOT EXISTS products (
    product_id  BIGINT PRIMARY KEY,
    name        VARCHAR,
    category_id BIGINT,
    unit_price  DOUBLE,
    cost        DOUBLE
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id    BIGINT PRIMARY KEY,
    name           VARCHAR,
    email          VARCHAR,
    signup_date    DATE,
    country        VARCHAR,
    segment        VARCHAR,
    lifetime_value DOUBLE
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     BIGINT PRIMARY KEY,
    customer_id  BIGINT,
    order_date   DATE,
    status       VARCHAR,
    channel      VARCHAR,
    total_amount DOUBLE
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id    BIGINT PRIMARY KEY,
    order_id   BIGINT,
    product_id BIGINT,
    quantity   BIGINT,
    unit_price DOUBLE,
    discount   DOUBLE
);
"""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]


SCHEMA = [
    Table("categories", (Column("category_id", "BIGINT"), Column("name", "VARCHAR"))),
    Table(
        "products",
        (
            Column("product_id", "BIGINT"),
            Column("name", "VARCHAR"),
            Column("category_id", "BIGINT"),
            Column("unit_price", "DOUBLE"),
            Column("cost", "DOUBLE"),
        ),
    ),
    Table(
        "customers",
        (
            Column("customer_id", "BIGINT"),
            Column("name", "VARCHAR"),
            Column("email", "VARCHAR"),
            Column("signup_date", "DATE"),
            Column("country", "VARCHAR"),
            Column("segment", "VARCHAR"),
            Column("lifetime_value", "DOUBLE"),
        ),
    ),
    Table(
        "orders",
        (
            Column("order_id", "BIGINT"),
            Column("customer_id", "BIGINT"),
            Column("order_date", "DATE"),
            Column("status", "VARCHAR"),
            Column("channel", "VARCHAR"),
            Column("total_amount", "DOUBLE"),
        ),
    ),
    Table(
        "order_items",
        (
            Column("item_id", "BIGINT"),
            Column("order_id", "BIGINT"),
            Column("product_id", "BIGINT"),
            Column("quantity", "BIGINT"),
            Column("unit_price", "DOUBLE"),
            Column("discount", "DOUBLE"),
        ),
    ),
]

# Revenu d'une ligne de commande : quantité * prix unitaire * (1 - remise)
REVENUE_EXPR = "oi.quantity * oi.unit_price * (1 - oi.discount)"

CATEGORIES = (
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Sports",
    "Books",
    "Toys",
    "Beauty",
    "Grocery",
)

COUNTRIES = ("France", "United States", "Germany", "United Kingdom", "Spain", "Italy", "Belgium", "Canada")

STATUSES = ("delivered", "shipped", "returned", "cancelled", "pending")

CHANNELS = ("web", "mobile", "store")


def schema_prompt() -> str:
    """Human-readable schema description used in LLM prompts."""
    lines = ["Base DuckDB `ecommerce` — schéma :"]
    for table in SCHEMA:
        cols = ", ".join(f"{c.name} {c.dtype}" for c in table.columns)
        lines.append(f"  - {table.name}({cols})")
    lines.append(
        "Règles métier : le revenu d'une ligne = order_items.quantity * order_items.unit_price * (1 - order_items.discount) ; "
        "order_items.order_id -> orders.order_id ; orders.customer_id -> customers.customer_id ; "
        "order_items.product_id -> products.product_id ; products.category_id -> categories.category_id ; "
        "orders.status IN (delivered, shipped, returned, cancelled, pending) ; "
        "orders.channel IN (web, mobile, store)."
    )
    return "\n".join(lines)
