"""Learning-to-rank: NDCG metrics, TF-IDF baseline, LightGBM LambdaRank model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # lightgbm is optional at runtime; numpy-only fallback keeps the agent usable
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None
    HAS_LIGHTGBM = False


# ----------------------------------------------------------------- metrics --

def dcg_at_k(grades: list[int] | np.ndarray, k: int = 5) -> float:
    grades = np.asarray(grades, dtype=np.float64)[:k]
    gains = np.power(2.0, grades) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(grades) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(grades: list[int] | np.ndarray, k: int = 5) -> float:
    """NDCG@k of a *ranking* (grades already ordered by the system's scores)."""
    grades = np.asarray(grades, dtype=np.float64)
    ideal = np.sort(grades)[::-1]
    idcg = dcg_at_k(ideal, k)
    return float(dcg_at_k(grades, k) / idcg) if idcg > 0 else 0.0


def average_ndcg(groups: list[dict], scores: np.ndarray, k: int = 5) -> float:
    """Average NDCG@k over question groups.

    groups: list of dicts with keys 'grades' (list of ints) and 'sizes' or
            implied consecutive layout; scores: flat array aligned with the
            concatenated candidates, in group order.
    """
    values, offset = [], 0
    for g in groups:
        n = len(g["grades"])
        block = scores[offset : offset + n]
        order = np.argsort(-block)
        ranked_grades = np.asarray(g["grades"])[order]
        values.append(ndcg_at_k(ranked_grades, k))
        offset += n
    return float(np.mean(values)) if values else 0.0


# -------------------------------------------------------------- baselines --

class TFIDFBaseline:
    """Cosine similarity between TF-IDF vectors of the question and each SQL."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=ngram_range, lowercase=True)
        self._question_vecs: np.ndarray | None = None
        self._sql_vecs: np.ndarray | None = None
        self._fitted = False

    def fit(self, questions: list[str], sqls: list[str]) -> "TFIDFBaseline":
        self.vectorizer.fit(questions + sqls)
        self._question_vecs = self.vectorizer.transform(questions)
        self._sql_vecs = self.vectorizer.transform(sqls)
        self._fitted = True
        return self

    def scores(self, question: str, sqls: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFBaseline.fit() must be called first")
        q = self.vectorizer.transform([question])
        s = self.vectorizer.transform(sqls)
        return np.asarray((q @ s.T).toarray()).ravel()


# ----------------------------------------------------------- learned ranker --

class LearnedRanker:
    """LightGBM LambdaRank model trained on graded relevance labels."""

    def __init__(
        self,
        label_gain: list[int] | None = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        seed: int = 42,
    ):
        self.label_gain = list(label_gain) if label_gain else [0, 1, 3, 7]
        self.params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[5],
            label_gain=self.label_gain,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=seed,
            verbosity=-1,
        )
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int], eval_set: tuple | None = None):
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is required to train the ranker")
        self.model = lgb.LGBMRanker(**self.params)
        if eval_set is not None:
            X_val, y_val, g_val = eval_set
            self.model.fit(X, y, group=groups, eval_set=[(X_val, y_val)], eval_group=[g_val])
        else:
            self.model.fit(X, y, group=groups)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker has not been trained or loaded")
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        self.model.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRanker":
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is required to load the ranker")
        ranker = cls()
        ranker.model = lgb.Booster(model_file=str(path))
        return ranker


class ConstantRanker:
    """Deterministic numpy fallback when lightgbm is unavailable (offline)."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.weights

    @classmethod
    def load(cls, path: str | Path) -> "ConstantRanker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(data["weights"], dtype=np.float64))
