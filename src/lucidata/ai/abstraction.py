from datetime import UTC, datetime

from lucidata.core.schema import (
    CategoricalProfile,
    ColumnHealth,
    CorrelationPair,
    DataQualityIndex,
    FeatureDriver,
)


def build_payload(  # noqa: PLR0913
    dqi: DataQualityIndex,
    correlations: list[CorrelationPair],
    drivers: list[FeatureDriver] | None,
    categorical: dict[str, CategoricalProfile],
    *,
    target: str | None = None,
    top_k_correlations: int = 5,
    top_k_drivers: int = 5,
) -> dict:
    """Build a sanitized JSON payload for LLM consumption.

    Only aggregated statistics are included. No raw row data, no PII,
    no sample values beyond what's in the frozen models (which default
    to empty lists per ColumnHealth definition).
    """
    metadata = {
        "target": target,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    shape = {
        "rows": dqi.total_rows,
        "columns": dqi.total_columns,
    }

    dqi_dict = {
        "overall_score": dqi.overall_score,
        "health_grade": dqi.health_grade,
        "total_null_cells": dqi.total_null_cells,
        "total_null_percentage": dqi.total_null_percentage,
        "duplicate_rows_count": dqi.duplicate_rows_count,
        "duplicate_rows_percentage": dqi.duplicate_rows_percentage,
    }

    columns = {}
    for name, health in dqi.column_health.items():
        if isinstance(health, ColumnHealth):
            columns[name] = {
                "name": health.name,
                "data_type": health.data_type.value,
                "total_count": health.total_count,
                "null_count": health.null_count,
                "null_percentage": health.null_percentage,
                "unique_count": health.unique_count,
                "is_constant": health.is_constant,
                "iqr_outliers_count": health.iqr_outliers_count,
            }
        else:
            columns[name] = health

    top_corr = sorted(
        correlations,
        key=lambda c: abs(c.pearson_coef),
        reverse=True,
    )[:top_k_correlations]

    top_correlations = [
        {
            "feature_a": c.feature_a,
            "feature_b": c.feature_b,
            "pearson_coef": c.pearson_coef,
            "spearman_coef": c.spearman_coef,
            "strength": c.strength,
        }
        for c in top_corr
    ]

    top_drivers = None
    if drivers:
        top_drv = sorted(drivers, key=lambda d: d.importance_score, reverse=True)[:top_k_drivers]
        top_drivers = [
            {
                "feature_name": d.feature_name,
                "importance_score": d.importance_score,
                "rank": d.rank,
                "relationship_summary": d.relationship_summary,
            }
            for d in top_drv
        ]

    categorical_dict = {}
    for name, profile in categorical.items():
        categorical_dict[name] = {
            "column": profile.column,
            "total_count": profile.total_count,
            "unique_count": profile.unique_count,
            "top_values": profile.top_values,
            "entropy": profile.entropy,
        }

    return {
        "metadata": metadata,
        "shape": shape,
        "dqi": dqi_dict,
        "columns": columns,
        "top_correlations": top_correlations,
        "top_drivers": top_drivers,
        "categorical": categorical_dict,
    }
