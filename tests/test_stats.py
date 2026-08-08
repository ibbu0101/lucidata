import math
import time

import numpy as np
import polars as pl
import pytest

from lucidata.engine.stats import (
    categorical_profile,
    correlations,
    drivers,
)


def test_correlations_perfect_positive(df_correlated) -> None:
    result = correlations(df_correlated)
    assert len(result) == 3
    # x-y should be strongest positive
    xy = next(
        r
        for r in result
        if (r.feature_a == "x" and r.feature_b == "y")
        or (r.feature_a == "y" and r.feature_b == "x")
    )
    assert xy.pearson_coef > 0.90
    assert xy.strength in ("Strong", "Very Strong")


def test_correlations_perfect_negative(df_correlated) -> None:
    result = correlations(df_correlated)
    # x-z should be negative
    xz = next(
        r
        for r in result
        if (r.feature_a == "x" and r.feature_b == "z")
        or (r.feature_a == "z" and r.feature_b == "x")
    )
    assert xz.pearson_coef < -0.70
    assert xz.strength in ("Strong", "Very Strong")


def test_correlations_threshold_filters_weak(df_correlated) -> None:
    # default min_abs=0.35 should include all 3 pairs
    result_default = correlations(df_correlated, min_abs=0.35)
    assert len(result_default) == 3

    # raise threshold to exclude weaker ones
    result_high = correlations(df_correlated, min_abs=0.95)
    assert len(result_high) <= 2  # only x-y and maybe x-z


def test_correlations_empty_on_no_numeric(df_categorical) -> None:
    result = correlations(df_categorical)
    assert result == []


def test_correlations_empty_dataframe(df_empty) -> None:
    result = correlations(df_empty)
    assert result == []


def test_correlations_skips_constant_column(df_constant_col) -> None:
    result = correlations(df_constant_col)
    # only "normal" is non-constant numeric, no pairs possible
    assert result == []


@pytest.mark.parametrize("r,expected_strength", [
    (0.10, "Weak"),
    (0.35, "Moderate"),
    (0.60, "Strong"),
    (0.85, "Very Strong"),
])
def test_correlations_strength_buckets(r, expected_strength) -> None:
    n = 200
    np.random.seed(42)
    x = np.random.normal(0, 1, n)
    # Create y with controlled correlation
    noise_scale = (1 - r**2)**0.5
    y = r * x + np.random.normal(0, noise_scale, n)
    df = pl.DataFrame({"x": x, "y": y})
    result = correlations(df, min_abs=0.0)
    assert len(result) == 1
    assert result[0].strength == expected_strength


def test_drivers_regression_target(df_with_target) -> None:
    result = drivers(df_with_target, "target_num")
    assert len(result) == 3
    assert all(d.rank == i + 1 for i, d in enumerate(result))
    assert all(0 < d.importance_score <= 1.0 for d in result)
    assert abs(sum(d.importance_score for d in result) - 1.0) < 1e-6


def test_drivers_classification_target(df_with_target) -> None:
    # target_cat has 4 numeric features (f1, f2, f3, target_num)
    result = drivers(df_with_target, "target_cat")
    assert len(result) == 4
    assert all(d.rank == i + 1 for i, d in enumerate(result))
    assert all(0 < d.importance_score <= 1.0 for d in result)
    assert abs(sum(d.importance_score for d in result) - 1.0) < 1e-6


def test_drivers_scores_sum_to_one(df_with_target) -> None:
    for target in ("target_num", "target_cat"):
        result = drivers(df_with_target, target)
        total = sum(d.importance_score for d in result)
        assert abs(total - 1.0) < 1e-6


def test_drivers_rank_descending(df_with_target) -> None:
    result = drivers(df_with_target, "target_num")
    scores = [d.importance_score for d in result]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_drivers_raises_on_missing_target(df_with_target) -> None:
    with pytest.raises(ValueError, match="not found"):
        drivers(df_with_target, "nonexistent")


def test_drivers_raises_on_text_target() -> None:
    df = pl.DataFrame({
        "feat": np.random.normal(0, 1, 100),
        "target": [f"text_{i}" for i in range(100)],  # high-cardinality text
    })
    with pytest.raises(ValueError, match="numeric or categorical"):
        drivers(df, "target")


def test_drivers_raises_on_boolean_target() -> None:
    df = pl.DataFrame({
        "feat": np.random.normal(0, 1, 100),
        "target": [True, False] * 50,
    })
    with pytest.raises(ValueError, match="numeric or categorical"):
        drivers(df, "target")


def test_categorical_profile_top_k_and_entropy(df_categorical) -> None:
    profiles = categorical_profile(df_categorical, top_k=2)
    assert "cat1" in profiles
    assert "cat2" in profiles
    p1 = profiles["cat1"]
    assert p1.total_count == 500
    assert p1.unique_count == 4
    assert len(p1.top_values) == 2
    assert p1.entropy >= 0
    # entropy should be <= log2(k) where k=unique_count=4
    assert p1.entropy <= math.log2(4) + 1e-6


@pytest.mark.slow
def test_correlations_speed_100k(df_large_100k) -> None:
    start = time.perf_counter()
    result = correlations(df_large_100k, min_abs=0.35)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"correlations took {elapsed:.2f}s, expected < 10s"
    # 10 numeric columns -> 45 pairs
    assert len(result) <= 45


@pytest.mark.slow
def test_drivers_speed_100k(df_large_100k) -> None:
    start = time.perf_counter()
    result = drivers(df_large_100k, "target", n_estimators=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"drivers took {elapsed:.2f}s, expected < 60s"
    assert len(result) == 10  # 10 features (feat_0 through feat_9, excluding target)
    assert abs(sum(d.importance_score for d in result) - 1.0) < 1e-6
