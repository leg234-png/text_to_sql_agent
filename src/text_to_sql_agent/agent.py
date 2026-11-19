"""The Text-to-SQL agent: generate → execute → rank → answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .database import build_database, connect
from .executor import ExecResult, execute
from .features import FEATURE_NAMES, extract_features, feature_matrix
from .llm import LLMClient
from .ranking import LearnedRanker, TFIDFBaseline
from .schema import schema_prompt
from .templates import GeneratedSQL, TemplateSQLGenerator

_WS = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    return _WS.sub(" ", sql.strip().rstrip(";").lower())


@dataclass
class Candidate:
    sql: str
    intent: str
    origin: str
    confidence: float
    valid: bool
    error: str | None
    n_rows: int
    score: float = float("-inf")


@dataclass
class Answer:
    question: str
    sql: str
    df: pd.DataFrame
    ranker_name: str
    candidates: list[Candidate] = field(default_factory=list)


class TextToSQLAgent:
    """Natural language → SQL, with a learned ranker choosing the best candidate."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        db_path = self.cfg.resolve("database_path")
        if not db_path.exists():
            counts = build_database(str(db_path), self.cfg.database_seed)
            total = sum(counts.values())
            print(f"[agent] base absente → construite ({len(counts)} tables, {total:,} lignes, seed={self.cfg.database_seed})")
        self.con = connect(str(db_path), read_only=True)
        self.templates = TemplateSQLGenerator()
        self.llm = LLMClient(self.cfg)
        self.ranker, self.ranker_name = self._load_ranker()

    # ------------------------------------------------------------- ranker --
    def _load_ranker(self):
        model_path = self.cfg.resolve("ranker_model_path")
        if model_path.exists():
            try:
                return LearnedRanker.load(model_path), "learned (LightGBM LambdaRank)"
            except Exception as exc:  # noqa: BLE001
                print(f"[agent] rangement appris illisible ({exc}) → repli TF-IDF")
        else:
            print("[agent] pas de modèle appris trouvé → repli TF-IDF (lancez scripts/train_ranker.py)")
        return TFIDFBaseline(), "TF-IDF baseline"

    # --------------------------------------------------------- candidates --
    def _generate_candidates(self, question: str) -> list[GeneratedSQL]:
        gens = self.templates.generate(question, with_variants=True)
        if self.llm.available:
            for sql, conf in self.llm.generate_sql(question, schema_prompt(), n=self.cfg.llm_candidates):
                gens.append(GeneratedSQL(sql, "llm", conf, origin="llm"))
        return gens

    # ---------------------------------------------------------------- ask --
    def ask(self, question: str, max_candidates: int = 12, explain: bool = False) -> Answer:
        gens = self._generate_candidates(question)[:max_candidates]

        # dedupe + execute
        seen: set[str] = set()
        rows: list[tuple[GeneratedSQL, ExecResult]] = []
        for g in gens:
            key = normalize_sql(g.sql)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append((g, execute(self.con, g.sql)))

        if not rows:
            raise ValueError("Aucun candidat SQL généré pour cette question.")

        # rank
        features = [extract_features(question, g.sql, res, g.confidence) for g, res in rows]
        X = feature_matrix(features)
        if isinstance(self.ranker, TFIDFBaseline):
            self.ranker.fit([question], [g.sql for g, _ in rows])
            scores = self.ranker.scores(question, [g.sql for g, _ in rows])
        else:
            scores = self.ranker.predict(X)

        candidates = [
            Candidate(
                sql=g.sql,
                intent=g.intent,
                origin=g.origin,
                confidence=g.confidence,
                valid=res.error is None,
                error=res.error,
                n_rows=len(res.df) if res.df is not None else 0,
                score=float(scores[i]),
            )
            for i, (g, res) in enumerate(rows)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        best = next((c for c in candidates if c.valid), None)
        if best is None:
            raise ValueError(
                "Aucun candidat exécutable. "
                f"Première erreur : {candidates[0].error if candidates else 'inconnue'}"
            )
        best_res = next(res for (g, res) in rows if normalize_sql(g.sql) == normalize_sql(best.sql))
        assert best_res.df is not None
        return Answer(question=question, sql=best.sql, df=best_res.df,
                      ranker_name=self.ranker_name, candidates=candidates if explain else [])

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------ helpers --
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES
