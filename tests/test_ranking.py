"""Feature extraction and NDCG metrics."""
from __future__ import annotations

import numpy as np

from text_to_sql_agent.executor import ExecResult
from text_to_sql_agent.features import FEATURE_NAMES, extract_features, feature_matrix, to_vector
from text_to_sql_agent.ranking import LearnedRanker, TFIDFBaseline, average_ndcg, dcg_at_k, ndcg_at_k


def test_feature_vector_is_complete_and_ordered():
    res = ExecResult(df=None, error="boom", elapsed=0.001)
    feats = extract_features("Top 5 des produits en 2024", "SELECT * FROM orders LIMIT 5", res, 0.7)
    assert sorted(feats.keys()) == sorted(FEATURE_NAMES)
    v = to_vector(feats)
    assert v.shape == (len(FEATURE_NAMES),)
    assert v[FEATURE_NAMES.index("limit_match")] == 1.0
    assert v[FEATURE_NAMES.index("is_valid")] == 0.0


def test_limit_match_partial():
    res = ExecResult(df=None, error=None, elapsed=0.0)
    feats = extract_features("Top 5 des produits", "SELECT * FROM orders LIMIT 3", res, 0.7)
    assert feats["limit_match"] == 0.5


def test_feature_matrix_stack():
    m = feature_matrix([{name: float(i % 3) for i, name in enumerate(FEATURE_NAMES)} for _ in range(4)])
    assert m.shape == (4, len(FEATURE_NAMES))


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k([3, 2, 1, 0], 5) == 1.0
    assert ndcg_at_k([0, 3, 2, 1], 5) < 1.0
    assert ndcg_at_k([0, 3, 0, 0, 0], 5) < ndcg_at_k([3, 0, 0, 0, 0], 5)
    assert ndcg_at_k([0, 0, 0, 0], 5) == 0.0
    assert dcg_at_k([3, 3, 2, 2, 1], 5) > dcg_at_k([3, 2, 2, 1, 1], 5)


def test_average_ndcg_groups():
    groups = [{"grades": [3, 2, 1, 0]}, {"grades": [3, 0, 2, 1]}]
    scores = np.array([9, 6, 3, 0, 9, 0, 6, 3])
    v = average_ndcg(groups, scores, k=5)
    assert abs(v - 1.0) < 1e-9


def test_tfidf_baseline_fit_scores():
    base = TFIDFBaseline().fit(["revenue par catégorie"], ["SELECT * FROM orders", "SELECT * FROM products"])
    scores = base.scores("revenue par catégorie", ["SELECT * FROM orders", "SELECT * FROM products"])
    assert scores.shape == (2,)
    assert scores[0] >= scores[1]  # 'orders' shares the token 'orders' with the question


def test_learned_ranker_train_predict_save_load(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 240, len(FEATURE_NAMES)
    X = rng.normal(size=(n, d))
    # informative feature 0
    grades = np.tile([3, 2, 1, 0, 0, 0], 40)
    X[:, 0] = grades * 3.0 + rng.normal(scale=0.2, size=n)
    groups = [6] * 40

    ranker = LearnedRanker(n_estimators=60, label_gain=[0, 1, 3, 7])
    ranker.fit(X, grades, groups)
    scores = ranker.predict(X[:6])
    assert scores.shape == (6,)

    path = tmp_path / "ranker.txt"
    ranker.save(path)
    loaded = LearnedRanker.load(path)
    assert np.allclose(loaded.predict(X[:6]), scores)

    # perfect ordering on the first group → NDCG@5 == 1
    assert ndcg_at_k(grades[np.argsort(-ranker.predict(X[:6]))], 5) == 1.0
