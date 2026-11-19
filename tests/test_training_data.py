"""The training-data generator must produce graded, valid pools."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.generate_training_data import build_training_frame  # noqa: E402


def test_build_training_frame_small():
    frame = build_training_frame(60, seed=42)
    assert isinstance(frame, pd.DataFrame)
    assert frame["question_id"].nunique() >= 55
    assert set(frame["grade"].unique()) <= {0, 1, 2, 3}
    sizes = frame.groupby("question_id").size()
    assert (sizes >= 6).all(), "chaque pool doit contenir au moins 6 candidats"
    # le gold (grade 3) est présent dans chaque pool
    assert (frame.groupby("question_id")["grade"].max() == 3).all()


def test_no_duplicate_sql_within_pool():
    frame = build_training_frame(40, seed=7)
    dup = frame.groupby("question_id").apply(
        lambda g: g["sql"].str.lower().str.replace(r"\s+", " ", regex=True).duplicated().any()
    )
    assert not dup.any()
