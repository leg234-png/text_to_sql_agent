"""CLI: build the 81,000-row DuckDB warehouse."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_to_sql_agent.config import load_config  # noqa: E402
from text_to_sql_agent.database import build_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic e-commerce DuckDB database")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--path", default=None, help="output .duckdb path (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = args.path or str(cfg.resolve("database_path"))
    seed = args.seed if args.seed is not None else cfg.database_seed

    counts = build_database(path, seed)
    total = sum(counts.values())
    for name, n in counts.items():
        print(f"  {name:<12} {n:>9,} rows")
    print(f"  {'TOTAL':<12} {total:>9,} rows")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
