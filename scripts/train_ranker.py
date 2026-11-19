"""Train and evaluate the learned ranker (NDCG@5) against the TF-IDF baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_to_sql_agent.config import Config, load_config  # noqa: E402
from text_to_sql_agent.database import connect  # noqa: E402
from text_to_sql_agent.executor import execute  # noqa: E402
from text_to_sql_agent.features import FEATURE_NAMES, extract_features, feature_matrix  # noqa: E402
from text_to_sql_agent.ranking import LearnedRanker, TFIDFBaseline, average_ndcg  # noqa: E402


def featurize(frame: pd.DataFrame, db_path: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[list[str]]]:
    """Execute every candidate and compute features; return (X, y, groups, texts)."""
    con = connect(str(db_path), read_only=True)
    cache: dict[str, object] = {}
    rows, y, groups, texts = [], [], [], []
    for qid, group in frame.groupby("question_id", sort=True):
        feats_block, texts_block = [], []
        for _, r in group.iterrows():
            sql = r["sql"]
            if sql not in cache:
                cache[sql] = execute(con, sql)
            res = cache[sql]
            feats = extract_features(r["question"], sql, res, 0.7)
            feats_block.append(feats)
            texts_block.append(sql)
            y.append(int(r["grade"]))
        rows.extend(feats_block)
        groups.append(len(feats_block))
        texts.append(texts_block)
    con.close()
    return feature_matrix(rows), np.asarray(y, dtype=np.int64), groups, texts


def main(cfg: Config | None = None) -> int:
    cfg = cfg or load_config()
    train_path = cfg.resolve("training_parquet")
    if not train_path.exists():
        print(f"{train_path} manquant → lancez d'abord scripts/generate_training_data.py")
        return 1

    frame = pd.read_parquet(train_path)
    X, y, groups, texts = featurize(frame, cfg.resolve("database_path"))

    q_texts = [
        frame[frame["question_id"] == qid]["question"].iloc[0]
        for qid in sorted(frame["question_id"].unique())
    ]
    bounds = np.cumsum([0] + groups)

    def slices(idx):
        Xs, ys, gs = [], [], []
        for i in idx:
            Xs.append(X[bounds[i]:bounds[i + 1]])
            ys.append(y[bounds[i]:bounds[i + 1]])
            gs.append(groups[i])
        return np.vstack(Xs), np.concatenate(ys), gs

    # ---- multi-split evaluation (moyenne ± écart) --------------------------
    split_seeds = (42, 43, 44)
    results = {s: {} for s in split_seeds}
    n_train = int(0.8 * len(groups))

    for s in split_seeds:
        rng = np.random.default_rng(s)
        order = rng.permutation(len(groups))
        train_idx, test_idx = order[:n_train], order[n_train:]

        X_train, y_train, g_train = slices(train_idx)
        X_test, y_test, g_test = slices(test_idx)

        test_bounds = np.cumsum([0] + [groups[i] for i in test_idx])
        test_groups = [
            {"grades": y_test[test_bounds[j]:test_bounds[j + 1]].tolist()}
            for j in range(len(test_idx))
        ]

        # baseline : TF-IDF cosine (fit on train only)
        train_questions = [q_texts[i] for i in train_idx]
        train_sqls = []
        for i in train_idx:
            train_sqls.extend(texts[i])
        baseline = TFIDFBaseline().fit(train_questions, train_sqls)
        base_scores = [baseline.scores(q_texts[i], texts[i]) for i in test_idx]
        base_flat = np.concatenate(base_scores)
        ndcg_base = average_ndcg(test_groups, base_flat, k=cfg.rank_k)

        # learned ranker : LightGBM LambdaRank
        learned = LearnedRanker(label_gain=cfg.label_gain, seed=s)
        learned.fit(X_train, y_train, g_train, eval_set=(X_test, y_test, g_test))
        scores = learned.predict(X_test)
        ndcg_learned = average_ndcg(test_groups, scores, k=cfg.rank_k)

        results[s]["ndcg_base"] = ndcg_base
        results[s]["ndcg_learned"] = ndcg_learned
        results[s]["test_idx"] = test_idx
        results[s]["learned"] = learned

    ndcg_bases = [results[s]["ndcg_base"] for s in split_seeds]
    ndcg_learneds = [results[s]["ndcg_learned"] for s in split_seeds]
    mean_base, std_base = float(np.mean(ndcg_bases)), float(np.std(ndcg_bases))
    mean_learned, std_learned = float(np.mean(ndcg_learneds)), float(np.std(ndcg_learneds))

    print(f"NDCG@5 baseline TF-IDF   : {mean_base:.4f} ± {std_base:.4f}   (splits: "
          + ", ".join(f"{v:.4f}" for v in ndcg_bases) + ")")
    print(f"NDCG@5 learned LambdaRank: {mean_learned:.4f} ± {std_learned:.4f}   (splits: "
          + ", ".join(f"{v:.4f}" for v in ndcg_learneds) + ")")

    # diagnostic sur le split principal : où atterrit une réponse parfaite ?
    s0 = split_seeds[0]
    learned0 = results[s0]["learned"]
    test_idx = results[s0]["test_idx"]
    X_test, y_test, _ = slices(test_idx)
    scores = learned0.predict(X_test)
    test_bounds = np.cumsum([0] + [groups[i] for i in test_idx])
    gold_top1, gold_top5 = 0, 0
    for j in range(len(test_idx)):
        block = scores[test_bounds[j]:test_bounds[j + 1]]
        order = np.argsort(-block)
        grades = y_test[test_bounds[j]:test_bounds[j + 1]]
        gold_positions = np.where(grades == 3)[0]
        gold_top1 += int(order[0] in gold_positions)
        gold_top5 += int(any(p in order[:5] for p in gold_positions))
    print(f"diagnostic: une réponse parfaite en #1 dans {gold_top1}/{len(test_idx)} groupes, "
          f"dans le top 5 dans {gold_top5}/{len(test_idx)}")

    # ---- persist (modèle du split principal) --------------------------------
    model_path = cfg.resolve("ranker_model_path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learned0.save(model_path)
    (cfg.resolve("feature_names_path")).write_text(
        json.dumps(FEATURE_NAMES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "metric": f"NDCG@{cfg.rank_k}",
        "baseline_tfidf": round(mean_base, 4),
        "baseline_tfidf_std": round(std_base, 4),
        "learned_lambdarank": round(mean_learned, 4),
        "learned_lambdarank_std": round(std_learned, 4),
        "per_split_baseline": [round(v, 4) for v in ndcg_bases],
        "per_split_learned": [round(v, 4) for v in ndcg_learneds],
        "n_train_questions": int(n_train),
        "n_test_questions": int(len(groups) - n_train),
        "n_candidates": int(len(y)),
        "model": "lightgbm.LGBMRanker",
        "label_gain": cfg.label_gain,
        "seed": cfg.database_seed,
    }
    cfg.resolve("metrics_path").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"NDCG@{cfg.rank_k}  baseline TF-IDF      : {ndcg_base:.4f}")
    print(f"NDCG@{cfg.rank_k}  learned LambdaRank   : {ndcg_learned:.4f}")
    print(f"train {len(train_idx)} questions / test {len(test_idx)} questions / {len(y)} candidats")
    print(f"model saved to {model_path}")
    print(f"metrics saved to {cfg.resolve('metrics_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
