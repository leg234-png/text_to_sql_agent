"""Ranking features computed from (question, SQL, execution result).

The feature vector is the bridge between candidate generation and the learned
ranker: it captures lexical agreement between the question and the SQL, the
SQL structure, what the execution returned, and the generator's confidence.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .executor import ExecResult

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_AGG_RE = re.compile(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"DATE\s*'\d{4}-\d{2}-\d{2}'")
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)
_TOP_RE = re.compile(
    r"\b(?:top|meilleurs?|meilleures?|premiers?|premieres?)\s*(\d{1,2})\b"
    r"|\b(\d{1,2})\s*(?:meilleurs?|meilleures?|premiers?|premieres?|plus\s+vendus?|produits|clients|categories)\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    # lexical agreement question <-> SQL
    "tok_jaccard", "tok_overlap_q",
    # numeric entities (top N)
    "limit_match",
    # thematic agreement
    "q_cat_sql_cat", "q_country_sql_country", "q_month_sql_month", "q_status_sql_status",
    "q_rank_sql_desc", "q_period_sql_date", "q_chan_sql_chan",
    # structure
    "n_tables", "n_joins", "has_group_by", "has_order_by", "has_limit",
    "n_aggs", "n_where_conds", "has_date_filter", "n_subqueries", "sql_len_log", "uses_strftime",
    # execution
    "is_valid", "n_rows_log", "n_cols", "elapsed_log", "money_magnitude_log", "truncated",
    # generator
    "gen_confidence",
]

_CAT_WORDS = ("categorie", "category", "categories")
_COUNTRY_WORDS = ("pays", "country", "countries")
_MONTH_WORDS = ("mois", "month", "monthly", "mensuel", "mensuelle")
_STATUS_WORDS = ("statut", "status")
_RANK_WORDS = ("top", "meilleur", "premier", "plus vendu", "plus de ventes", "depensier", "classement")
_PERIOD_WORDS = ("annee", "année", "year", "mois de", "month of", "mars", "avril", "mai", "juin",
                 "juillet", "aout", "septembre", "octobre", "novembre", "decembre", "janvier", "fevrier",
                 "march", "april", "june", "july", "august", "september", "october", "november",
                 "december", "january", "february", "2023", "2024", "2025", "2026")
_CHAN_WORDS = ("canal", "channel", "web", "mobile", "magasin", "store", "boutique")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text.lower() for w in words)


def extract_features(
    question: str,
    sql: str,
    result: ExecResult | None,
    generator_confidence: float,
) -> dict[str, float]:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sql)
    q_low, s_low = question.lower(), sql.lower()

    feats: dict[str, float] = {}
    feats["tok_jaccard"] = _jaccard(q_tokens, s_tokens)
    feats["tok_overlap_q"] = len(q_tokens & s_tokens) / len(q_tokens) if q_tokens else 0.0

    # top-N agreement: does the SQL LIMIT match the number in the question?
    m_q = _TOP_RE.search(question)
    m_s = _LIMIT_RE.search(sql)
    if m_q and m_s:
        n_q = int(m_q.group(1) or m_q.group(2))
        feats["limit_match"] = 1.0 if n_q == int(m_s.group(1)) else 0.5
    elif not m_q and not m_s:
        feats["limit_match"] = 1.0
    else:
        feats["limit_match"] = 0.0

    # thematic agreement
    feats["q_cat_sql_cat"] = float(_has_word(question, _CAT_WORDS) and ("categories" in s_low))
    feats["q_country_sql_country"] = float(_has_word(question, _COUNTRY_WORDS) and ("cu.country" in s_low or "country" in s_low))
    feats["q_month_sql_month"] = float(_has_word(question, _MONTH_WORDS) and "strftime" in s_low)
    feats["q_status_sql_status"] = float(_has_word(question, _STATUS_WORDS) and "o.status" in s_low)

    q_rank = _has_word(question, _RANK_WORDS)
    sql_desc = " desc" in f" {s_low} "
    if q_rank:
        feats["q_rank_sql_desc"] = float(sql_desc)
    else:
        feats["q_rank_sql_desc"] = 0.5
    q_period = _has_word(question, _PERIOD_WORDS)
    sql_date = bool(_DATE_RE.search(sql))
    if q_period:
        feats["q_period_sql_date"] = float(sql_date)
    else:
        feats["q_period_sql_date"] = 0.5
    q_chan = _has_word(question, _CHAN_WORDS)
    sql_chan = bool(re.search(r"\bo\.channel\s*=", s_low))
    if q_chan:
        feats["q_chan_sql_chan"] = float(sql_chan)
    else:
        feats["q_chan_sql_chan"] = 0.5

    # structure
    feats["n_tables"] = float(len(re.findall(r"\bFROM\b", s_low)) + len(re.findall(r"\bJOIN\b", s_low)))
    feats["n_joins"] = float(len(re.findall(r"\bJOIN\b", s_low)))
    feats["has_group_by"] = float("group by" in s_low)
    feats["has_order_by"] = float("order by" in s_low)
    feats["has_limit"] = float(bool(m_s))
    feats["n_aggs"] = float(len(_AGG_RE.findall(sql)))
    feats["n_where_conds"] = float(1 + len(re.findall(r"\bAND\b|\bOR\b", s_low))) if " where " in f" {s_low} " else 0.0
    feats["has_date_filter"] = float(bool(_DATE_RE.search(sql)))
    feats["n_subqueries"] = float(max(0, len(_SELECT_RE.findall(sql)) - 1))
    feats["sql_len_log"] = math.log1p(len(sql))
    feats["uses_strftime"] = float("strftime" in s_low)

    # execution
    if result is not None:
        feats["is_valid"] = float(result.error is None)
        feats["n_rows_log"] = math.log1p(len(result.df)) if result.df is not None else 0.0
        feats["n_cols"] = float(len(result.df.columns)) if result.df is not None else 0.0
        feats["elapsed_log"] = math.log1p(result.elapsed * 1000.0)
        feats["money_magnitude_log"] = _money_magnitude(result.df)
        feats["truncated"] = float(result.truncated)
    else:
        for name in ("is_valid", "n_rows_log", "n_cols", "elapsed_log", "money_magnitude_log", "truncated"):
            feats[name] = 0.0

    feats["gen_confidence"] = float(generator_confidence)
    return feats


def _money_magnitude(df: pd.DataFrame | None) -> float:
    """log1p of the absolute value of the first numeric column's sum.

    Separates aggregate money questions (large values) from count queries.
    """
    if df is None or df.empty:
        return 0.0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].abs().sum()
            return math.log1p(min(float(s), 1e12))
    return 0.0


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.asarray([feats.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


def feature_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.vstack([to_vector(r) for r in rows])
