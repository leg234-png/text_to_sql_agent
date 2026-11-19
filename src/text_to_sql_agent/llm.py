"""Optional LLM client (OpenAI-compatible). Falls back silently to templates."""
from __future__ import annotations

import os
import re

from .config import Config

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMClient:
    """Thin wrapper around any OpenAI-compatible chat API.

    The agent stays fully functional without an API key: the template
    generator produces the candidate pool instead.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None
        self._init_failed = False

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.cfg.llm_api_key_env, "").strip())

    def _ensure_client(self):
        if self._client is not None or self._init_failed:
            return self._client
        try:
            from openai import OpenAI  # type: ignore

            kwargs: dict = {"api_key": os.environ.get(self.cfg.llm_api_key_env, "")}
            if self.cfg.llm_base_url:
                kwargs["base_url"] = self.cfg.llm_base_url
            self._client = OpenAI(**kwargs)
        except Exception:
            self._init_failed = True
            self._client = None
        return self._client

    def generate_sql(self, question: str, schema_text: str, n: int = 1) -> list[tuple[str, float]]:
        """Return up to `n` (sql, confidence) pairs, or [] when unavailable."""
        client = self._ensure_client()
        if client is None or not self.available:
            return []

        system = (
            "You are an expert DuckDB SQL developer. Translate the user's business "
            "question into a single valid SQL SELECT query. Return ONLY the SQL, no "
            "explanation. Prefer explicit JOINs over comma joins."
        )
        user = f"Schema:\n{schema_text}\n\nQuestion: {question}\n\nSQL:"
        try:
            resp = client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.cfg.llm_temperature,
                n=n,
            )
        except Exception:
            return []

        out: list[tuple[str, float]] = []
        for choice in resp.choices:
            text = choice.message.content or ""
            m = _SQL_FENCE.search(text)
            sql = m.group(1).strip() if m else text.strip()
            if sql.lower().startswith(("select", "with")):
                out.append((sql, 0.7))
        return out
