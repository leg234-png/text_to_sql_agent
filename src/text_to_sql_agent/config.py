"""Configuration loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    database_path: str = "data/ecommerce.duckdb"
    database_seed: int = 42
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_temperature: float = 0.2
    llm_candidates: int = 3
    llm_api_key_env: str = "OPENAI_API_KEY"
    ranker_model_path: str = "models/ranker.txt"
    feature_names_path: str = "models/feature_names.json"
    metrics_path: str = "models/metrics.json"
    rank_k: int = 5
    label_gain: list[int] = field(default_factory=lambda: [0, 1, 3, 7])
    training_parquet: str = "data/ranker_training.parquet"
    n_questions: int = 500

    def resolve(self, key: str) -> Path:
        """Resolve a path-like setting against the project root."""
        p = Path(getattr(self, key))
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, overridable by environment variables."""
    cfg = Config()
    src = Path(path) if path else DEFAULT_CONFIG
    if src.exists():
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}

        db = raw.get("database", {})
        cfg.database_path = db.get("path", cfg.database_path)
        cfg.database_seed = int(db.get("seed", cfg.database_seed))

        llm = raw.get("llm", {})
        cfg.llm_provider = llm.get("provider", cfg.llm_provider)
        cfg.llm_model = llm.get("model", cfg.llm_model)
        cfg.llm_base_url = llm.get("base_url") or None
        cfg.llm_temperature = float(llm.get("temperature", cfg.llm_temperature))
        cfg.llm_candidates = int(llm.get("candidates", cfg.llm_candidates))
        cfg.llm_api_key_env = llm.get("api_key_env", cfg.llm_api_key_env)

        rk = raw.get("ranking", {})
        cfg.ranker_model_path = rk.get("model_path", cfg.ranker_model_path)
        cfg.feature_names_path = rk.get("feature_names_path", cfg.feature_names_path)
        cfg.metrics_path = rk.get("metrics_path", cfg.metrics_path)
        cfg.rank_k = int(rk.get("k", cfg.rank_k))
        cfg.label_gain = [int(x) for x in rk.get("label_gain", cfg.label_gain)]

        dt = raw.get("data", {})
        cfg.training_parquet = dt.get("training_parquet", cfg.training_parquet)
        cfg.n_questions = int(dt.get("n_questions", cfg.n_questions))

    # Environment overrides
    cfg.llm_model = os.environ.get("T2S_LLM_MODEL", cfg.llm_model)
    cfg.llm_base_url = os.environ.get("T2S_LLM_BASE_URL", cfg.llm_base_url)
    return cfg
