"""Shared fixtures for the test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_agent.config import Config  # noqa: E402
from text_to_sql_agent.database import build_database, connect  # noqa: E402


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("db") / "ecommerce.duckdb"
    build_database(str(p), seed=42)
    return str(p)


@pytest.fixture(scope="session")
def con(db_path):
    c = connect(db_path, read_only=True)
    yield c
    c.close()


@pytest.fixture()
def cfg(db_path, tmp_path):
    c = Config()
    c.database_path = db_path
    c.ranker_model_path = str(tmp_path / "ranker.txt")
    c.feature_names_path = str(tmp_path / "feature_names.json")
    c.metrics_path = str(tmp_path / "metrics.json")
    return c
