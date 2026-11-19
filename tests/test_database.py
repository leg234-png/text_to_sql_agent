"""The warehouse must contain 81,000 fact rows and be internally consistent."""
from __future__ import annotations

import hashlib

from text_to_sql_agent.database import build_database, connect


def test_row_counts(con):
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("categories", "products", "customers", "orders", "order_items")}
    assert counts == {"categories": 8, "products": 2500, "customers": 8100, "orders": 21600, "order_items": 81000}
    assert counts["order_items"] >= 81_000


def test_every_order_has_items(con):
    n_orders_with_items = con.execute(
        "SELECT COUNT(DISTINCT order_id) FROM order_items"
    ).fetchone()[0]
    n_orders = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert n_orders_with_items == n_orders == 21600


def test_total_amount_is_consistent(con):
    mismatch = con.execute(
        """
        SELECT COUNT(*)
        FROM orders o
        JOIN (
            SELECT order_id, SUM(quantity * unit_price * (1 - discount)) AS computed
            FROM order_items GROUP BY order_id
        ) t ON t.order_id = o.order_id
        WHERE ABS(o.total_amount - t.computed) > 1.0
        """
    ).fetchone()[0]
    assert mismatch == 0


def test_rebuild_is_deterministic(tmp_path):
    p1, p2 = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
    build_database(str(p1), seed=42)
    build_database(str(p2), seed=42)

    def digest(path):
        con = connect(str(path), read_only=True)
        h = hashlib.sha256()
        for t in ("categories", "products", "customers", "orders", "order_items"):
            for row in con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall():
                h.update(str(row).encode())
        con.close()
        return h.hexdigest()

    assert digest(p1) == digest(p2)
