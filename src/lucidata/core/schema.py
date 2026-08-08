from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lucidata.core.datatypes import DataType


class ColumnHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: DataType
    total_count: int
    null_count: int
    null_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int
    is_constant: bool
    iqr_outliers_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)


class DataQualityIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(ge=0.0, le=100.0, description="0-100 overall health score")
    total_rows: int
    total_columns: int
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    total_null_cells: int
    total_null_percentage: float
    health_grade: str = Field(description="A, B, C, D, or F based on overall score")
    column_health: dict[str, ColumnHealth]


class CorrelationPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_a: str
    feature_b: str
    pearson_coef: float
    spearman_coef: float
    strength: str = Field(description="Weak, Moderate, Strong, or Very Strong")


class FeatureDriver(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_name: str
    importance_score: float = Field(ge=0.0, le=1.0)
    rank: int
    relationship_summary: str


class CategoricalProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    total_count: int
    unique_count: int
    top_values: list[tuple[str, int]]
    entropy: float
