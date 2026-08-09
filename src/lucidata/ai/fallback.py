from lucidata.core.schema import (
    CorrelationPair,
    DataQualityIndex,
    ExecutiveSummaryNarrative,
    FeatureDriver,
)

# Thresholds for anomaly detection
NULL_PCT_ANOMALY_THRESHOLD = 5.0
NULL_PCT_ACTION_THRESHOLD = 10.0
DUP_PCT_ANOMALY_THRESHOLD = 1.0
DUP_PCT_ACTION_THRESHOLD = 5.0


def _format_pct(val: float) -> str:
    return f"{val:.1f}%"


def _grade_action(grade: str) -> str:
    if grade == "A":
        return "Dataset is production-ready; consider publishing or modeling immediately."
    if grade == "B":
        return "Minor quality issues detected; review anomalies before modeling."
    if grade == "C":
        return "Moderate quality concerns; address nulls and duplicates before analysis."
    if grade == "D":
        return "Significant quality issues; prioritize data cleaning and imputation."
    return "Critical quality failure; data requires substantial remediation before use."


def _collect_anomalies(dqi: DataQualityIndex, correlations: list[CorrelationPair]) -> list[str]:
    anomalies: list[str] = []

    if dqi.total_null_percentage > NULL_PCT_ANOMALY_THRESHOLD:
        anomalies.append(
            f"Null rate is {_format_pct(dqi.total_null_percentage)} across "
            f"{dqi.total_null_cells} cells."
        )

    if dqi.duplicate_rows_percentage > DUP_PCT_ANOMALY_THRESHOLD:
        anomalies.append(
            f"Duplicate rows: {dqi.duplicate_rows_count} "
            f"({_format_pct(dqi.duplicate_rows_percentage)})."
        )

    constant_cols = sum(1 for h in dqi.column_health.values() if h.is_constant)
    if constant_cols:
        anomalies.append(f"{constant_cols} constant column(s) detected (zero variance).")

    very_strong = [c for c in correlations if c.strength == "Very Strong"]
    if very_strong:
        anomalies.append(
            f"{len(very_strong)} feature pair(s) show very strong correlation "
            f"(|r| >= 0.85); consider multicollinearity."
        )

    if not anomalies:
        anomalies.append("No significant data quality anomalies detected.")

    return anomalies, constant_cols, very_strong


def _collect_highlights(
    correlations: list[CorrelationPair],
    drivers: list[FeatureDriver] | None,
) -> list[str]:
    highlights: list[str] = []

    if correlations:
        top3 = sorted(correlations, key=lambda c: abs(c.pearson_coef), reverse=True)[:3]
        for c in top3:
            direction = "positively" if c.pearson_coef > 0 else "negatively"
            highlights.append(
                f"'{c.feature_a}' and '{c.feature_b}' are {direction} "
                f"correlated (r={c.pearson_coef:.2f}, {c.strength.lower()})."
            )

    if drivers:
        top3 = sorted(drivers, key=lambda d: d.importance_score, reverse=True)[:3]
        for d in top3:
            highlights.append(
                f"Top driver: '{d.feature_name}' (importance={d.importance_score:.2f}, "
                f"rank={d.rank})."
            )

    if not highlights:
        highlights.append("No strong statistical relationships detected among features.")

    return highlights


def _collect_next_steps(
    grade: str,
    dqi: DataQualityIndex,
    constant_cols: int,
    very_strong: list[CorrelationPair],
    drivers: list[FeatureDriver] | None,
) -> list[str]:
    next_steps = [_grade_action(grade)]

    if dqi.total_null_percentage > NULL_PCT_ACTION_THRESHOLD:
        next_steps.append("Investigate missingness patterns and apply imputation strategy.")
    if dqi.duplicate_rows_percentage > DUP_PCT_ACTION_THRESHOLD:
        next_steps.append("Deduplicate rows before modeling.")
    if constant_cols:
        next_steps.append("Remove or engineer constant columns.")
    if very_strong:
        next_steps.append("Review highly correlated features for dimensionality reduction.")

    if not drivers:
        next_steps.append("Define a target variable to compute feature driver rankings.")

    return next_steps


def generate_heuristic_narrative(
    dqi: DataQualityIndex,
    correlations: list[CorrelationPair],
    drivers: list[FeatureDriver] | None,
) -> ExecutiveSummaryNarrative:
    """Generate a rule-based narrative when LLM is unavailable.

    Produces a valid ExecutiveSummaryNarrative from statistical aggregates
    without any external service calls.
    """
    total_rows = f"{dqi.total_rows:,}"
    total_cols = dqi.total_columns
    grade = dqi.health_grade
    dqi_score = f"{dqi.overall_score:.0f}"

    headline = f"{total_rows}-row × {total_cols}-col dataset · grade {grade} (DQI {dqi_score}/100)"

    highlights = _collect_highlights(correlations, drivers)
    anomalies, constant_cols, very_strong = _collect_anomalies(dqi, correlations)
    next_steps = _collect_next_steps(grade, dqi, constant_cols, very_strong, drivers)

    return ExecutiveSummaryNarrative(
        headline=headline,
        key_highlights=highlights,
        data_anomalies=anomalies,
        actionable_next_steps=next_steps,
    )
