import datetime
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

# Import schema types used in AI fixtures
from lucidata.ai.abstraction import build_payload
from lucidata.core.schema import (
    CategoricalProfile,
    ColumnHealth,
    CorrelationPair,
    DataQualityIndex,
    DataType,
    FeatureDriver,
)

np.random.seed(42)


@pytest.fixture
def df_continuous() -> pl.DataFrame:
    n = 1000
    return pl.DataFrame(
        {
            "a": np.random.normal(0, 1, n),
            "b": np.random.exponential(2, n),
            "c": np.random.uniform(-5, 5, n),
        }
    )


@pytest.fixture
def df_categorical() -> pl.DataFrame:
    n = 500
    return pl.DataFrame(
        {
            "cat1": np.random.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "cat2": np.random.choice(["X", "Y"], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def df_mixed() -> pl.DataFrame:
    n = 200
    start = datetime.datetime(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "numeric": np.random.normal(10, 3, n),
            "category": np.random.choice(["low", "med", "high"], n),
            "datetime": dates,
        }
    )


@pytest.fixture
def df_null_heavy() -> pl.DataFrame:
    n = 100
    df = pl.DataFrame(
        {
            "mostly_null": [1.0 if i % 10 == 0 else None for i in range(n)],
            "half_null": [float(i) if i % 2 == 0 else None for i in range(n)],
            "no_null": np.random.normal(0, 1, n),
        }
    )
    return df


@pytest.fixture
def df_single_column() -> pl.DataFrame:
    return pl.DataFrame({"only": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture
def df_empty() -> pl.DataFrame:
    return pl.DataFrame(schema={"a": pl.Float64, "b": pl.Int64, "c": pl.String})


@pytest.fixture
def df_all_null() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [None] * 100,
            "col2": [None] * 100,
        }
    )


@pytest.fixture
def df_constant_col() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "constant": [42] * 100,
            "normal": np.random.normal(0, 1, 100),
        }
    )


@pytest.fixture
def df_with_duplicates() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
        }
    )
    # Add 3 duplicate rows
    dups = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
        }
    )
    return pl.concat([base, dups])


@pytest.fixture
def df_with_outliers() -> pl.DataFrame:
    n = 1000
    normal = np.random.normal(0, 1, n - 5)
    # 5 extreme outliers
    outliers = np.array([100, -100, 50, -50, 200])
    return pl.DataFrame({"values": np.concatenate([normal, outliers])})


# Stats-specific fixtures
@pytest.fixture
def df_correlated() -> pl.DataFrame:
    """DataFrame with known strong positive correlation (r ≈ 0.95)."""
    n = 500
    x = np.random.normal(0, 1, n)
    y = x * 0.9 + np.random.normal(0, 0.3, n)  # strong positive
    z = -x * 0.8 + np.random.normal(0, 0.4, n)  # strong negative
    return pl.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def df_with_target() -> pl.DataFrame:
    """DataFrame with numeric target for regression and categorical target for classification."""
    n = 300
    f1 = np.random.normal(0, 1, n)
    f2 = np.random.normal(5, 2, n)
    f3 = np.random.exponential(1, n)
    # Regression target: linear combo + noise
    target_num = 2.5 * f1 - 1.2 * f2 + 0.8 * f3 + np.random.normal(0, 0.5, n)
    # Classification target: 3 classes based on quartiles
    bins = np.percentile(target_num, [33, 66])
    target_cat = np.array(
        ["low" if v <= bins[0] else "high" if v >= bins[1] else "med" for v in target_num]
    )
    return pl.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "target_num": target_num,
            "target_cat": target_cat,
        }
    )


@pytest.fixture
def df_large_100k() -> pl.DataFrame:
    """100k rows × 10 numeric columns for benchmarking."""
    n = 100_000
    np.random.seed(123)
    data = {f"feat_{i}": np.random.normal(0, 1, n) for i in range(10)}
    # Add one target column
    target_vals = sum(data[f"feat_{i}"] * (i + 1) * 0.1 for i in range(5)) + np.random.normal(
        0, 0.5, n
    )
    data["target"] = target_vals
    return pl.DataFrame(data)


# File-based fixtures
@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "test.csv"
    df.write_csv(p)
    return p


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
    p = tmp_path / "test.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def tmp_xlsx(tmp_path: Path) -> Path:
    if pd is None:
        pytest.skip("pandas not installed")
    df = pl.DataFrame({"a": [1, 2], "b": ["foo", "bar"]})
    p = tmp_path / "test.xlsx"
    df.to_pandas().to_excel(p, index=False)
    return p


@pytest.fixture
def tmp_sqlite(tmp_path: Path) -> Path:
    if pd is None:
        pytest.skip("pandas not installed")

    p = tmp_path / "test.db"
    conn = sqlite3.connect(p)
    pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]}).to_pandas().to_sql(
        "mytable", conn, index=False
    )
    conn.close()
    return p


# AI-specific fixtures
@pytest.fixture
def sample_dqi() -> DataQualityIndex:
    """Minimal DataQualityIndex for AI testing."""
    return DataQualityIndex(
        overall_score=85.0,
        total_rows=1000,
        total_columns=5,
        duplicate_rows_count=5,
        duplicate_rows_percentage=0.5,
        total_null_cells=50,
        total_null_percentage=1.0,
        health_grade="B",
        column_health={
            "feat_1": ColumnHealth(
                name="feat_1",
                data_type=DataType.NUMERIC,
                total_count=1000,
                null_count=10,
                null_percentage=1.0,
                unique_count=950,
                is_constant=False,
                iqr_outliers_count=3,
            ),
            "feat_2": ColumnHealth(
                name="feat_2",
                data_type=DataType.NUMERIC,
                total_count=1000,
                null_count=5,
                null_percentage=0.5,
                unique_count=980,
                is_constant=False,
                iqr_outliers_count=1,
            ),
            "cat_1": ColumnHealth(
                name="cat_1",
                data_type=DataType.CATEGORICAL,
                total_count=1000,
                null_count=35,
                null_percentage=3.5,
                unique_count=4,
                is_constant=False,
                iqr_outliers_count=0,
            ),
            "const_col": ColumnHealth(
                name="const_col",
                data_type=DataType.NUMERIC,
                total_count=1000,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                is_constant=True,
                iqr_outliers_count=0,
            ),
            "target": ColumnHealth(
                name="target",
                data_type=DataType.NUMERIC,
                total_count=1000,
                null_count=0,
                null_percentage=0.0,
                unique_count=990,
                is_constant=False,
                iqr_outliers_count=2,
            ),
        },
    )


@pytest.fixture
def sample_correlations() -> list[CorrelationPair]:
    """Sample CorrelationPair list for AI testing."""
    return [
        CorrelationPair(
            feature_a="feat_1",
            feature_b="target",
            pearson_coef=0.92,
            spearman_coef=0.90,
            strength="Very Strong",
        ),
        CorrelationPair(
            feature_a="feat_2",
            feature_b="target",
            pearson_coef=-0.65,
            spearman_coef=-0.62,
            strength="Strong",
        ),
        CorrelationPair(
            feature_a="feat_1",
            feature_b="feat_2",
            pearson_coef=0.15,
            spearman_coef=0.12,
            strength="Weak",
        ),
    ]


@pytest.fixture
def sample_drivers() -> list[FeatureDriver]:
    """Sample FeatureDriver list for AI testing."""
    return [
        FeatureDriver(
            feature_name="feat_1",
            importance_score=0.65,
            rank=1,
            relationship_summary="Combined importance=0.650 (RF=0.620, MI=0.680)",
        ),
        FeatureDriver(
            feature_name="feat_2",
            importance_score=0.35,
            rank=2,
            relationship_summary="Combined importance=0.350 (RF=0.380, MI=0.320)",
        ),
    ]


@pytest.fixture
def sample_categorical() -> dict[str, CategoricalProfile]:
    """Sample CategoricalProfile dict for AI testing."""
    return {
        "cat_1": CategoricalProfile(
            column="cat_1",
            total_count=1000,
            unique_count=4,
            top_values=[("A", 400), ("B", 300), ("C", 200), ("D", 100)],
            entropy=1.85,
        ),
    }


@pytest.fixture
def sample_payload(
    sample_dqi: DataQualityIndex,
    sample_correlations: list[CorrelationPair],
    sample_drivers: list[FeatureDriver],
    sample_categorical: dict[str, CategoricalProfile],
) -> dict:
    """Pre-built sanitized payload for LLM testing."""
    return build_payload(
        dqi=sample_dqi,
        correlations=sample_correlations,
        drivers=sample_drivers,
        categorical=sample_categorical,
        target="target",
    )


@pytest.fixture
def sample_payload_no_drivers(
    sample_dqi: DataQualityIndex,
    sample_correlations: list[CorrelationPair],
    sample_categorical: dict[str, CategoricalProfile],
) -> dict:
    """Payload without drivers (target=None)."""
    return build_payload(
        dqi=sample_dqi,
        correlations=sample_correlations,
        drivers=None,
        categorical=sample_categorical,
        target=None,
    )


# Mock helper for LLM client
class MockCompletionClient:
    """Fake instructor-compatible client for testing.

    Matches the interface of `instructor.from_litellm(litellm.completion)`:
    client.chat.completions.create(...)
    """

    def __init__(self, return_value=None, raise_exception=None):
        self._return_value = return_value
        self._raise_exception = raise_exception
        self.call_count = 0
        # Build the nested structure: chat -> completions -> create
        self.chat = Mock()
        self.chat.completions = Mock()
        # Use a Mock for create so it tracks call_args
        self.chat.completions.create = Mock(side_effect=self._create)

    def _create(self, **kwargs):
        self.call_count += 1
        if self._raise_exception:
            raise self._raise_exception
        return self._return_value


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Patch lucidata.ai.llm_client._get_client to return a mock."""

    def _make_mock(return_value=None, raise_exception=None):
        mock = MockCompletionClient(return_value=return_value, raise_exception=raise_exception)
        monkeypatch.setattr("lucidata.ai.llm_client._get_client", lambda: mock)
        return mock

    return _make_mock
