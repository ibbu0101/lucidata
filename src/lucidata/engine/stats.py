import math

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from lucidata.core.datatypes import DataType
from lucidata.core.schema import (
    CategoricalProfile,
    CorrelationPair,
    FeatureDriver,
)
from lucidata.engine.auditor import _infer_type

# Constants for correlation strength thresholds
CORR_WEAK_MAX = 0.35
CORR_MODERATE_MAX = 0.60
CORR_STRONG_MAX = 0.85
MIN_PAIR_OBSERVATIONS = 2
MIN_NUMERIC_COLS_FOR_CORR = 2
MIN_ROWS_FOR_DRIVERS = 2


def _get_numeric_columns(df: pl.DataFrame) -> list[str]:
    """Return names of numeric columns in the dataframe."""
    numeric_cols = []
    for col_name in df.columns:
        col = df[col_name]
        dtype = _infer_type(col)
        if dtype == DataType.NUMERIC:
            numeric_cols.append(col_name)
    return numeric_cols


def _pairwise_complete_obs(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays with only indices where both have finite values."""
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _correlation_strength(r: float) -> str:
    """Map absolute Pearson r to strength label."""
    abs_r = abs(r)
    if abs_r < CORR_WEAK_MAX:
        return "Weak"
    if abs_r < CORR_MODERATE_MAX:
        return "Moderate"
    if abs_r < CORR_STRONG_MAX:
        return "Strong"
    return "Very Strong"


def correlations(
    df: pl.DataFrame, min_abs: float = CORR_WEAK_MAX
) -> list[CorrelationPair]:
    """Compute Pearson and Spearman correlations for all numeric column pairs.

    Args:
        df: Input Polars DataFrame.
        min_abs: Minimum absolute Pearson |r| to include in results. Default 0.35 per FR-2.1.

    Returns:
        List of CorrelationPair objects sorted by descending |pearson_coef|.
    """
    numeric_cols = _get_numeric_columns(df)
    if len(numeric_cols) < MIN_NUMERIC_COLS_FOR_CORR:
        return []

    results: list[CorrelationPair] = []

    for i, col_a in enumerate(numeric_cols):
        a = df[col_a].to_numpy()
        for col_b in numeric_cols[i + 1 :]:
            b = df[col_b].to_numpy()

            a_clean, b_clean = _pairwise_complete_obs(a, b)
            if len(a_clean) < MIN_PAIR_OBSERVATIONS:
                continue

            try:
                pearson_coef, _ = pearsonr(a_clean, b_clean)
                spearman_coef, _ = spearmanr(a_clean, b_clean)
            except Exception:
                continue

            if not np.isfinite(pearson_coef) or not np.isfinite(spearman_coef):
                continue

            if abs(pearson_coef) >= min_abs:
                results.append(
                    CorrelationPair(
                        feature_a=col_a,
                        feature_b=col_b,
                        pearson_coef=float(pearson_coef),
                        spearman_coef=float(spearman_coef),
                        strength=_correlation_strength(pearson_coef),
                    )
                )

    results.sort(key=lambda x: abs(x.pearson_coef), reverse=True)
    return results


def _prepare_driver_data(
    df: pl.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray, list[str], DataType]:
    """Prepare features and target arrays for driver analysis.

    Returns:
        (X, y, feature_names, target_type)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    target_col = df[target]
    target_type = _infer_type(target_col)

    if target_type not in (DataType.NUMERIC, DataType.CATEGORICAL):
        raise ValueError(
            f"Target must be numeric or categorical, got {target_type.value}"
        )

    # Drop target column, keep only numeric features for RF/MI
    feature_cols = [c for c in df.columns if c != target]
    numeric_features = []
    for col_name in feature_cols:
        col = df[col_name]
        if _infer_type(col) == DataType.NUMERIC:
            numeric_features.append(col_name)

    if not numeric_features:
        raise ValueError("No numeric feature columns available for driver analysis")

    # Listwise drop rows with nulls in target or any feature
    subset_cols = numeric_features + [target]
    clean_df = df.select(subset_cols).drop_nulls()

    if clean_df.height < MIN_ROWS_FOR_DRIVERS:
        raise ValueError(
            "Insufficient rows after dropping nulls (< 2 rows remaining)"
        )

    X = clean_df.select(numeric_features).to_numpy()
    y = clean_df[target].to_numpy()

    return X, y, numeric_features, target_type


def _normalize_to_unit(scores: np.ndarray) -> np.ndarray:
    """Normalize array to sum to 1.0, handling all-zero case."""
    total = scores.sum()
    if total == 0:
        return np.ones_like(scores) / len(scores)
    return scores / total


def drivers(
    df: pl.DataFrame,
    target: str,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> list[FeatureDriver]:
    """Compute feature driver importance for a target column.

    Uses 50/50 blend of Random Forest importance and Mutual Information,
    re-normalized to sum to 1.0.

    Args:
        df: Input Polars DataFrame.
        target: Target column name.
        n_estimators: Number of trees in random forest.
        random_state: Random seed for reproducibility.

    Returns:
        List of FeatureDriver sorted by descending importance_score.
    """
    X, y, feature_names, target_type = _prepare_driver_data(df, target)

    if target_type == DataType.NUMERIC:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        rf_importance = rf.feature_importances_

        mi_scores = mutual_info_regression(X, y, random_state=random_state)
    else:  # CATEGORICAL
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        rf_importance = rf.feature_importances_

        mi_scores = mutual_info_classif(X, y, random_state=random_state)

    # Normalize each method independently, then blend 50/50 and re-normalize
    rf_norm = _normalize_to_unit(rf_importance)
    mi_norm = _normalize_to_unit(mi_scores)
    blended = 0.5 * rf_norm + 0.5 * mi_norm  # judgment call, not in Blueprint spec
    final_scores = _normalize_to_unit(blended)

    ranked_idx = np.argsort(final_scores)[::-1]

    drivers_list: list[FeatureDriver] = []
    for rank, idx in enumerate(ranked_idx, 1):
        feat_name = feature_names[idx]
        score = float(final_scores[idx])
        if score <= 0:
            continue

        if target_type == DataType.NUMERIC:
            summary = (
                f"Combined importance={score:.3f} (RF={rf_norm[idx]:.3f}, "
                f"MI={mi_norm[idx]:.3f})"
            )
        else:
            summary = (
                f"Combined importance={score:.3f} (RF={rf_norm[idx]:.3f}, "
                f"MI={mi_norm[idx]:.3f})"
            )

        drivers_list.append(
            FeatureDriver(
                feature_name=feat_name,
                importance_score=score,
                rank=rank,
                relationship_summary=summary,
            )
        )

    return drivers_list


def _compute_entropy(values: list[int]) -> float:
    """Compute Shannon entropy from frequency counts."""
    total = sum(values)
    if total == 0:
        return 0.0
    entropy = 0.0
    for v in values:
        p = v / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def categorical_profile(
    df: pl.DataFrame, top_k: int = 5
) -> dict[str, CategoricalProfile]:
    """Profile categorical columns: top-k values and entropy.

    Args:
        df: Input Polars DataFrame.
        top_k: Number of top values to return per column.

    Returns:
        Dict mapping column name to CategoricalProfile.
    """
    profiles: dict[str, CategoricalProfile] = {}

    for col_name in df.columns:
        col = df[col_name]
        dtype = _infer_type(col)

        if dtype not in (DataType.CATEGORICAL, DataType.TEXT):
            continue

        total = col.len()
        non_null = col.drop_nulls()
        unique_count = non_null.n_unique()

        # Value counts
        vc = non_null.value_counts(sort=True)
        top_rows = vc.head(top_k)
        top_values = list(
            zip(
                top_rows[col_name].to_list(),
                top_rows["count"].to_list(),
                strict=False,
            )
        )

        # Entropy on full non-null distribution
        all_counts = vc["count"].to_list()
        entropy = _compute_entropy(all_counts)

        profiles[col_name] = CategoricalProfile(
            column=col_name,
            total_count=total,
            unique_count=unique_count,
            top_values=top_values,
            entropy=entropy,
        )

    return profiles
