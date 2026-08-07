from pathlib import Path
from typing import Union

import polars as pl

from lucidata.core.exceptions import IngestionError

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

if pd is not None:
    DataFrameInput = Union[str, Path, pl.DataFrame, pd.DataFrame]
else:
    DataFrameInput = Union[str, Path, pl.DataFrame]


def ingest(source: DataFrameInput) -> pl.DataFrame:
    """Load data from file or pass through an existing DataFrame.

    Supports: CSV, Parquet, Excel (.xlsx/.xls), SQLite (.db/.sqlite/.sqlite3).
    Uses Polars natively with Pandas fallback for Excel and SQLite.
    """
    if isinstance(source, pl.DataFrame):
        return source

    if pd is not None and isinstance(source, pd.DataFrame):
        return pl.from_pandas(source)

    if isinstance(source, Path):
        path = source
    elif isinstance(source, str):
        path = Path(source)
    else:
        raise IngestionError(f"Unsupported source type: {type(source)}")

    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pl.read_csv(str(path))
        elif suffix == ".parquet":
            return pl.read_parquet(str(path))
        elif suffix in (".xlsx", ".xls"):
            return _read_excel(path, pd)
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            return _read_sqlite(path, pd)
        else:
            raise IngestionError(f"Unsupported file format: {suffix}")
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to ingest {path}: {e}") from e


def _read_excel(path: Path, pd) -> pl.DataFrame:
    try:
        return pl.read_excel(str(path))
    except ImportError:
        if pd is None:
            raise IngestionError(
                "Excel support requires pandas. Install with: pip install pandas openpyxl"
            )
        df = pd.read_excel(str(path))
        return pl.from_pandas(df)
    except Exception:
        if pd is None:
            raise
        df = pd.read_excel(str(path))
        return pl.from_pandas(df)


def _read_sqlite(path: Path, pd) -> pl.DataFrame:
    if pd is None:
        raise IngestionError("SQLite support requires pandas. Install with: pip install pandas")

    import sqlite3

    try:
        conn = sqlite3.connect(str(path))
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", conn
        )
        if tables.empty:
            raise IngestionError(f"No user tables found in {path}")
        table_name = tables.iloc[0]["name"]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return pl.from_pandas(df)
    except Exception as e:
        raise IngestionError(f"Failed to read SQLite database {path}: {e}") from e
