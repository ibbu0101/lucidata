import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import polars as pl
import pytest

try:
    import pandas as pd
except ImportError:
    pd = None


np.random.seed(42)


@pytest.fixture
def df_continuous() -> pl.DataFrame:
    n = 1000
    return pl.DataFrame({
        "a": np.random.normal(0, 1, n),
        "b": np.random.exponential(2, n),
        "c": np.random.uniform(-5, 5, n),
    })


@pytest.fixture
def df_categorical() -> pl.DataFrame:
    n = 500
    return pl.DataFrame({
        "cat1": np.random.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "cat2": np.random.choice(["X", "Y"], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def df_mixed() -> pl.DataFrame:
    n = 200
    # Generate datetime range manually since pl.datetime_range doesn't have 'n' param
    import datetime
    start = datetime.datetime(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(n)]
    return pl.DataFrame({
        "numeric": np.random.normal(10, 3, n),
        "category": np.random.choice(["low", "med", "high"], n),
        "datetime": dates,
    })


@pytest.fixture
def df_null_heavy() -> pl.DataFrame:
    n = 100
    df = pl.DataFrame({
        "mostly_null": [1.0 if i % 10 == 0 else None for i in range(n)],
        "half_null": [float(i) if i % 2 == 0 else None for i in range(n)],
        "no_null": np.random.normal(0, 1, n),
    })
    return df


@pytest.fixture
def df_single_column() -> pl.DataFrame:
    return pl.DataFrame({"only": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture
def df_empty() -> pl.DataFrame:
    return pl.DataFrame(schema={"a": pl.Float64, "b": pl.Int64, "c": pl.String})


@pytest.fixture
def df_all_null() -> pl.DataFrame:
    return pl.DataFrame({
        "col1": [None] * 100,
        "col2": [None] * 100,
    })


@pytest.fixture
def df_constant_col() -> pl.DataFrame:
    return pl.DataFrame({
        "constant": [42] * 100,
        "normal": np.random.normal(0, 1, 100),
    })


@pytest.fixture
def df_with_duplicates() -> pl.DataFrame:
    base = pl.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
    })
    # Add 3 duplicate rows
    dups = pl.DataFrame({
        "x": [1, 2, 3],
        "y": [10, 20, 30],
    })
    return pl.concat([base, dups])


@pytest.fixture
def df_with_outliers() -> pl.DataFrame:
    n = 1000
    normal = np.random.normal(0, 1, n - 5)
    # 5 extreme outliers
    outliers = np.array([100, -100, 50, -50, 200])
    return pl.DataFrame({"values": np.concatenate([normal, outliers])})


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
    import sqlite3

    p = tmp_path / "test.db"
    conn = sqlite3.connect(p)
    pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]}).to_pandas().to_sql("mytable", conn, index=False)
    conn.close()
    return p