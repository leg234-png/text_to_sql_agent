"""Deterministic synthetic e-commerce data + DuckDB engine.

Builds 81,000 rows in `order_items` (the fact table) plus dimension tables:
categories (8), products (2,500), customers (8,100), orders (21,600).
Everything is seeded so the warehouse is reproducible on any machine.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .schema import SCHEMA_SQL

SEED = 42
N_CATEGORIES = 8
N_PRODUCTS = 2_500
N_NEVER_ORDERED_PRODUCTS = 10  # products N_PRODUCTS-N_NEVER_ORDERED_PRODUCTS+1 .. N_PRODUCTS have no sales
N_CUSTOMERS = 8_100
N_ORDERS = 21_600
N_ORDER_ITEMS = 81_000

CATEGORY_NAMES = ("Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys", "Beauty", "Grocery")
COUNTRIES = ("France", "United States", "Germany", "United Kingdom", "Spain", "Italy", "Belgium", "Canada")
COUNTRY_W = (0.30, 0.20, 0.12, 0.10, 0.08, 0.07, 0.05, 0.08)
SEGMENTS = ("Retail", "Wholesale", "VIP")
SEGMENT_W = (0.70, 0.20, 0.10)
STATUSES = ("delivered", "shipped", "returned", "cancelled", "pending")
STATUS_W = (0.72, 0.08, 0.06, 0.05, 0.09)
CHANNELS = ("web", "mobile", "store")
CHANNEL_W = (0.55, 0.25, 0.20)
DISCOUNTS = (0.0, 0.05, 0.10, 0.20)
DISCOUNT_W = (0.60, 0.20, 0.12, 0.08)

FIRST_NAMES = ("Emma", "Liam", "Olivia", "Noah", "Ava", "Lucas", "Léa", "Hugo", "Mia", "Gabriel",
               "Chloé", "Adam", "Inès", "Nathan", "Zoé", "Tom", "Louise", "Ethan", "Jade", "Louis",
               "Camille", "Arthur", "Sarah", "Victor", "Anna", "Maxime", "Julie", "Antoine", "Nora", "Paul")
LAST_NAMES = ("Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy",
              "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux",
              "Vincent", "Fournier", "Morel", "Girard", "Andre", "Lefevre", "Mercier", "Dupont", "Lambert", "Bonnet")
BRANDS = ("Nova", "Aurora", "Vertex", "Pulse", "Zephyr", "Orbit", "Lumen", "Cedar", "Ember", "Atlas")
NOUNS = ("Pro", "Lite", "Max", "Plus", "Mini", "Air", "Wave", "Core", "Prime", "Neo",
         "Edge", "Flex", "Zoom", "Swift", "Zen", "Peak", "Flow", "Glow", "Spark", "Echo")

PRODUCT_BY_CATEGORY_PRICE = {
    0: (40.0, 600.0),   # Electronics
    1: (15.0, 120.0),   # Clothing
    2: (10.0, 250.0),   # Home & Garden
    3: (20.0, 180.0),   # Sports
    4: (5.0, 40.0),     # Books
    5: (8.0, 90.0),     # Toys
    6: (6.0, 80.0),     # Beauty
    7: (2.0, 30.0),     # Grocery
}


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_dataframes(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate all tables as pandas DataFrames (deterministic)."""
    rng = _rng(seed)

    # ---- categories -------------------------------------------------------
    categories = pd.DataFrame(
        {"category_id": np.arange(1, N_CATEGORIES + 1), "name": list(CATEGORY_NAMES)}
    )

    # ---- products ---------------------------------------------------------
    cat = rng.integers(1, N_CATEGORIES + 1, size=N_PRODUCTS)  # 1-based
    prices = np.empty(N_PRODUCTS)
    for cid in range(N_CATEGORIES):
        lo, hi = PRODUCT_BY_CATEGORY_PRICE[cid]
        mask = cat == cid + 1
        prices[mask] = np.round(rng.uniform(lo, hi, size=mask.sum()), 2)
    products = pd.DataFrame(
        {
            "product_id": np.arange(1, N_PRODUCTS + 1),
            "name": [
                f"{rng.choice(BRANDS)} {rng.choice(NOUNS)}"
                for _ in range(N_PRODUCTS)
            ],
            "category_id": cat,
            "unit_price": prices,
            "cost": np.round(prices * rng.uniform(0.45, 0.70, size=N_PRODUCTS), 2),
        }
    )

    # ---- customers --------------------------------------------------------
    n = N_CUSTOMERS
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "name": [
                f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)
            ],
            "email": [
                f"customer{i}@example.com" for i in range(1, n + 1)
            ],
            "signup_date": pd.to_datetime(
                rng.integers(0, int((pd.Timestamp("2025-06-30") - pd.Timestamp("2020-01-01")).days) + 1, size=n),
                unit="D",
                origin=pd.Timestamp("2020-01-01"),
            ),
            "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_W),
            "segment": rng.choice(SEGMENTS, size=n, p=SEGMENT_W),
            "lifetime_value": np.round(
                rng.lognormal(mean=6.0, sigma=0.8, size=n), 2
            ),
        }
    )

    # ---- orders -----------------------------------------------------------
    # Power-law customer pick so some customers order repeatedly.
    pick = rng.zipf(1.35, size=N_ORDERS)
    pick = ((pick - 1) % n) + 1  # fold into [1, n]
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, N_ORDERS + 1),
            "customer_id": pick,
            "order_date": pd.to_datetime(
                rng.integers(0, int((pd.Timestamp("2025-10-31") - pd.Timestamp("2023-01-01")).days) + 1, size=N_ORDERS),
                unit="D",
                origin=pd.Timestamp("2023-01-01"),
            ),
            "status": rng.choice(STATUSES, size=N_ORDERS, p=STATUS_W),
            "channel": rng.choice(CHANNELS, size=N_ORDERS, p=CHANNEL_W),
            "total_amount": 0.0,
        }
    )

    # ---- order_items (the 81,000-row fact table) ---------------------------
    items_per_order = rng.integers(1, 7, size=N_ORDERS)  # 1..6 items
    order_ids = np.repeat(np.arange(1, N_ORDERS + 1), items_per_order)
    if len(order_ids) < N_ORDER_ITEMS:
        extra = rng.integers(1, N_ORDERS + 1, size=N_ORDER_ITEMS - len(order_ids))
        order_ids = np.concatenate([order_ids, extra])
    order_ids = order_ids[:N_ORDER_ITEMS]
    rng.shuffle(order_ids)

    pid = rng.integers(1, N_PRODUCTS - N_NEVER_ORDERED_PRODUCTS + 1, size=N_ORDER_ITEMS)
    base_price = products["unit_price"].values[pid - 1]
    order_items = pd.DataFrame(
        {
            "item_id": np.arange(1, N_ORDER_ITEMS + 1),
            "order_id": order_ids,
            "product_id": pid,
            "quantity": rng.integers(1, 6, size=N_ORDER_ITEMS),
            "unit_price": np.round(base_price * rng.uniform(0.9, 1.1, size=N_ORDER_ITEMS), 2),
            "discount": rng.choice(DISCOUNTS, size=N_ORDER_ITEMS, p=DISCOUNT_W),
        }
    )

    # ---- derive orders.total_amount from the items -------------------------
    line_total = (
        order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount"])
    )
    totals = pd.Series(line_total).groupby(order_items["order_id"]).sum()
    orders["total_amount"] = orders["order_id"].map(totals).round(2)

    return {
        "categories": categories,
        "products": products,
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
    }


def build_database(path: str, seed: int = SEED) -> dict[str, int]:
    """Create the DuckDB file and load all tables. Returns row counts."""
    frames = generate_dataframes(seed)
    con = duckdb.connect(path)
    con.execute(SCHEMA_SQL)
    counts: dict[str, int] = {}
    for name, df in frames.items():
        con.execute(f"DELETE FROM {name}")
        con.register(f"df_{name}", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM df_{name}")
        con.unregister(f"df_{name}")
        counts[name] = int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
    con.close()
    return counts


def connect(path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path, read_only=read_only)
