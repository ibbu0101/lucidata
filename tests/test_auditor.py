import pytest

from lucidata.engine.auditor import audit, _health_grade


def test_audit_empty_dataframe(df_empty) -> None:
    result = audit(df_empty)
    assert result.total_rows == 0
    assert result.total_columns == 3
    assert result.overall_score == 100.0
    assert result.health_grade == "A"
    assert result.column_health == {}


def test_audit_single_column(df_single_column) -> None:
    result = audit(df_single_column)
    assert result.total_rows == 10
    assert result.total_columns == 1
    assert len(result.column_health) == 1
    ch = result.column_health["only"]
    assert ch.data_type.value == "numeric"
    assert ch.total_count == 10
    assert ch.null_count == 0
    assert ch.unique_count == 10
    assert ch.is_constant is False


def test_audit_all_null(df_all_null) -> None:
    result = audit(df_all_null)
    assert result.total_rows == 100
    assert result.total_columns == 2
    assert result.total_null_cells == 200
    assert result.total_null_percentage == 100.0
    # All-null also means 99% duplicate rows (all identical) and 100% constant columns
    # DQI = 100 - (0.4*100 + 0.3*99 + 0.1*100) = 20.3
    assert result.overall_score == 20.3
    assert result.health_grade == "F"
    assert result.duplicate_rows_count == 99
    for ch in result.column_health.values():
        assert ch.null_count == 100
        assert ch.null_percentage == 100.0
        assert ch.is_constant is True


def test_audit_constant_column(df_constant_col) -> None:
    result = audit(df_constant_col)
    assert result.total_columns == 2
    const_ch = result.column_health["constant"]
    assert const_ch.is_constant is True
    assert const_ch.unique_count == 1
    normal_ch = result.column_health["normal"]
    assert normal_ch.is_constant is False
    assert normal_ch.unique_count > 1
    # Constant column penalty applied
    assert result.overall_score < 100


def test_audit_with_duplicates(df_with_duplicates) -> None:
    result = audit(df_with_duplicates)
    # 5 unique + 3 duplicates = 8 rows
    assert result.total_rows == 8
    assert result.duplicate_rows_count == 3
    assert result.duplicate_rows_percentage == 37.5


def test_audit_with_outliers(df_with_outliers) -> None:
    result = audit(df_with_outliers)
    ch = result.column_health["values"]
    assert ch.iqr_outliers_count >= 5
    assert ch.data_type.value == "numeric"
    assert result.total_null_cells == 0


def test_audit_mixed_types(df_mixed) -> None:
    result = audit(df_mixed)
    types = {name: ch.data_type.value for name, ch in result.column_health.items()}
    assert types["numeric"] == "numeric"
    assert types["category"] == "categorical"
    assert types["datetime"] == "datetime"


def test_audit_continuous(df_continuous) -> None:
    result = audit(df_continuous)
    assert result.total_rows == 1000
    assert result.total_columns == 3
    for ch in result.column_health.values():
        assert ch.data_type.value == "numeric"
        assert ch.null_count == 0
    # No nulls, no duplicates expected, no outliers in well-behaved normal
    assert result.overall_score >= 90.0


def test_audit_null_heavy(df_null_heavy) -> None:
    result = audit(df_null_heavy)
    ch = result.column_health["mostly_null"]
    assert ch.null_count == 90
    assert ch.null_percentage == 90.0
    ch2 = result.column_health["half_null"]
    assert ch2.null_count == 50
    assert ch2.null_percentage == 50.0
    ch3 = result.column_health["no_null"]
    assert ch3.null_count == 0


@pytest.mark.parametrize("score,expected_grade", [
    (100, "A"),
    (95, "A"),
    (90, "A"),
    (89, "B"),
    (85, "B"),
    (80, "B"),
    (79, "C"),
    (75, "C"),
    (70, "C"),
    (69, "D"),
    (65, "D"),
    (60, "D"),
    (59, "F"),
    (0, "F"),
])
def test_health_grade_bands(score, expected_grade) -> None:
    assert _health_grade(score) == expected_grade


def test_audit_dqi_clamping() -> None:
    # Construct pathological data: all null, all duplicate, all outlier, all constant
    import polars as pl
    df = pl.DataFrame({
        "a": [None] * 10,
        "b": [None] * 10,
    })
    result = audit(df)
    # Even with 100% nulls, score floored at 0
    assert result.overall_score >= 0.0
    assert result.overall_score <= 100.0