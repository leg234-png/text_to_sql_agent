"""End-to-end tests of the agent (offline path, real DB, learned ranker)."""
from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sql_agent.agent import TextToSQLAgent

ROOT = Path(__file__).resolve().parent.parent
PROD_MODEL = ROOT / "models" / "ranker.txt"


@pytest.fixture()
def agent(cfg):
    if PROD_MODEL.exists():
        cfg.ranker_model_path = str(PROD_MODEL)
    a = TextToSQLAgent(cfg)
    yield a
    a.close()


def test_offline_end_to_end_french(agent):
    answer = agent.ask("Quels sont les 10 produits les plus vendus en 2024 ?", explain=True)
    assert "LIMIT 10" in answer.sql
    assert len(answer.df) == 10
    assert answer.candidates, "explain=True doit retourner le classement"
    assert answer.candidates[0].valid


def test_offline_end_to_end_english(agent):
    answer = agent.ask("What is the total revenue in 2024 ?")
    assert len(answer.df) == 1
    assert float(answer.df.iloc[0, 0]) > 0


def test_ranker_orders_candidates(agent):
    answer = agent.ask("Combien de commandes par statut en 2024 ?", explain=True)
    scores = [c.score for c in answer.candidates]
    assert scores == sorted(scores, reverse=True)
    assert answer.candidates[0].score == max(scores)


def test_unknown_question_raises(agent):
    with pytest.raises(ValueError):
        agent.ask("Peux-tu me raconter une blague ?")


def test_llm_client_absent_key(cfg):
    from text_to_sql_agent.llm import LLMClient
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    client = LLMClient(cfg)
    assert not client.available
    assert client.generate_sql("combien de commandes ?", "schema", n=1) == []
