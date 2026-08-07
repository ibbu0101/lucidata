from pathlib import Path

import polars as pl
import pytest

from lucidata.core.exceptions import IngestionError
from lucidata.engine.ingest import ingest


def test_ingest_polars_identity() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = ingest(df)
    assert result is df  # same object, no copy


def test_ingest_pandas_conversion() -> None:
    pd = pytest.importorskip("pandas")
    pdf = pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]})
    result = ingest(pdf)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 2)


def test_ingest_csv(tmp_csv: Path) -> None:
    result = ingest(tmp_csv)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["a", "b"]


def test_ingest_parquet(tmp_parquet: Path) -> None:
    result = ingest(tmp_parquet)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["a", "b"]


def test_ingest_xlsx(tmp_xlsx: Path) -> None:
    result = ingest(tmp_xlsx)
    assert result.shape == (2, 2)
    assert list(result.columns) == ["a", "b"]


def test_ingest_sqlite(tmp_sqlite: Path) -> None:
    result = ingest(tmp_sqlite)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["id", "val"]


def test_ingest_string_path(tmp_csv: Path) -> None:
    result = ingest(str(tmp_csv))
    assert result.shape == (3, 2)


def test_ingest_pathlib_path(tmp_csv: Path) -> None:
    result = ingest(tmp_csv)
    assert result.shape == (3, 2)


def test_ingest_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_text("garbage")
    with pytest.raises(IngestionError, match="Unsupported file format"):
        ingest(p)


def test_ingest_missing_file() -> None:
    with pytest.raises(IngestionError, match="File not found"):
        ingest("does_not_exist.csv")


def test_ingest_corrupt_csv(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2,3\n4,5")  # malformed row
    with pytest.raises(IngestionError, match="Failed to ingest"):
        ingest(p)


def test_ingest_empty_csv(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")  # header only
    result = ingest(p)
    assert result.height == 0
    assert list(result.columns) == ["a", "b"]