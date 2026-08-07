import polars as pl

from lucidata.core.datatypes import DataType
from lucidata.core.schema import ColumnHealth, DataQualityIndex

DQI_WEIGHTS = {
    "null": 0.4,
    "duplicate": 0.3,
    "outlier": 0.2,
    "constant": 0.1,
}

GRADE_BANDS = {
    "A": 90,
    "B": 80,
    "C": 70,
    "D": 60,
}


def audit(df: pl.DataFrame) -> DataQualityIndex:
    """Compute Data Quality Index and per-column health metrics.

    Returns a validated DataQualityIndex Pydantic object.
    """
    total_rows = df.height
    total_columns = df.width

    if total_rows == 0 or total_columns == 0:
        return DataQualityIndex(
            overall_score=100.0,
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows_count=0,
            duplicate_rows_percentage=0.0,
            total_null_cells=0,
            total_null_percentage=0.0,
            health_grade="A",
            column_health={},
        )

    # Dataset-level metrics
    # Count extra duplicate rows only (total - unique), not all occurrences
    unique_rows = df.unique().height
    duplicate_rows_count = total_rows - unique_rows
    duplicate_rows_percentage = (duplicate_rows_count / total_rows) * 100 if total_rows > 0 else 0.0

    # Per-column analysis
    column_health: dict[str, ColumnHealth] = {}
    total_null_cells = 0
    total_cells = total_rows * total_columns
    total_outlier_cells = 0
    constant_col_count = 0

    for col_name in df.columns:
        col = df[col_name]
        ch = _audit_column(col_name, col, total_rows)
        column_health[col_name] = ch

        total_null_cells += ch.null_count
        if ch.data_type == DataType.NUMERIC:
            total_outlier_cells += ch.iqr_outliers_count
        if ch.is_constant:
            constant_col_count += 1

    total_null_percentage = (total_null_cells / total_cells) * 100 if total_cells > 0 else 0.0

    # Constant columns penalty: % of cells in constant columns
    constant_col_pct = (
        (constant_col_count * total_rows / total_cells * 100) if total_cells > 0 else 0.0
    )

    # Outlier percentage: % of numeric cells that are outliers
    numeric_cols = sum(1 for ch in column_health.values() if ch.data_type == DataType.NUMERIC)
    numeric_cells = numeric_cols * total_rows
    outlier_pct = (total_outlier_cells / numeric_cells * 100) if numeric_cells > 0 else 0.0

    # DQI calculation
    dqi = 100 - (
        DQI_WEIGHTS["null"] * total_null_percentage
        + DQI_WEIGHTS["duplicate"] * duplicate_rows_percentage
        + DQI_WEIGHTS["outlier"] * outlier_pct
        + DQI_WEIGHTS["constant"] * constant_col_pct
    )
    dqi = max(0.0, min(100.0, dqi))

    health_grade = _health_grade(dqi)

    return DataQualityIndex(
        overall_score=round(dqi, 2),
        total_rows=total_rows,
        total_columns=total_columns,
        duplicate_rows_count=duplicate_rows_count,
        duplicate_rows_percentage=round(duplicate_rows_percentage, 2),
        total_null_cells=total_null_cells,
        total_null_percentage=round(total_null_percentage, 2),
        health_grade=health_grade,
        column_health=column_health,
    )


def _audit_column(name: str, col: pl.Series, total_rows: int) -> ColumnHealth:
    null_count = col.null_count()
    null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0.0

    # Type inference
    data_type = _infer_type(col)
    unique_count = col.n_unique()

    is_constant = unique_count <= 1
    iqr_outliers = 0

    if data_type == DataType.NUMERIC:
        iqr_outliers = _iqr_outlier_count(col)

    return ColumnHealth(
        name=name,
        data_type=data_type,
        total_count=total_rows,
        null_count=null_count,
        null_percentage=round(null_percentage, 2),
        unique_count=unique_count,
        is_constant=is_constant,
        iqr_outliers_count=iqr_outliers,
        sample_values=[],  # Privacy-safe: left empty by default
    )


def _infer_type(col: pl.Series) -> DataType:
    dtype = col.dtype

    if dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ):
        return DataType.NUMERIC
    elif dtype == pl.Boolean:
        return DataType.BOOLEAN
    elif dtype in (pl.Datetime, pl.Date, pl.Time):
        return DataType.DATETIME
    elif dtype == pl.String:
        # Heuristic: try datetime parse on non-null sample
        non_null = col.drop_nulls()
        if non_null.len() > 0:
            sample = non_null.head(min(100, non_null.len()))
            try:
                parsed = sample.str.to_datetime(strict=False)
                parsed_count = parsed.drop_nulls().len()
                if parsed_count / sample.len() >= 0.95:
                    return DataType.DATETIME
            except Exception:
                pass
        # Cardinality heuristic
        unique_ratio = non_null.n_unique() / non_null.len() if non_null.len() > 0 else 0
        if unique_ratio > 0.5:
            return DataType.TEXT
        return DataType.CATEGORICAL
    elif dtype == pl.Categorical:
        return DataType.CATEGORICAL
    else:
        return DataType.UNKNOWN


def _iqr_outlier_count(col: pl.Series) -> int:
    """Count IQR outliers in a numeric column."""
    clean = col.drop_nulls()
    if clean.len() == 0:
        return 0

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = clean.filter((clean < lower) | (clean > upper))
    return outliers.len()


def _health_grade(score: float) -> str:
    for grade, threshold in GRADE_BANDS.items():
        if score >= threshold:
            return grade
    return "F"
