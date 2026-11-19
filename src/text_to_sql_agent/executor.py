"""Safe SQL execution on DuckDB, returning DataFrames or errors."""
from __future__ import annotations

import time
from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass
class ExecResult:
    df: pd.DataFrame | None
    error: str | None
    elapsed: float
    truncated: bool = False


def execute(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = 200) -> ExecResult:
    """Execute a candidate SQL and capture the result (or the error)."""
    t0 = time.perf_counter()
    try:
        df = con.execute(sql).fetchdf()
        elapsed = time.perf_counter() - t0
        truncated = len(df) > max_rows
        if truncated:
            df = df.head(max_rows)
        return ExecResult(df=df, error=None, elapsed=elapsed, truncated=truncated)
    except Exception as exc:  # noqa: BLE001 - we keep the message for ranking features
        return ExecResult(df=None, error=str(exc), elapsed=time.perf_counter() - t0, truncated=False)
