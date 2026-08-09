import pytest
from pydantic import ValidationError

from lucidata.ai.abstraction import build_payload
from lucidata.ai.fallback import generate_heuristic_narrative
from lucidata.ai.llm_client import generate_narrative
from lucidata.config import settings
from lucidata.core.exceptions import LLMProviderError
from lucidata.core.schema import (
    CategoricalProfile,
    ColumnHealth,
    CorrelationPair,
    DataQualityIndex,
    DataType,
    ExecutiveSummaryNarrative,
    FeatureDriver,
)

# Test constants
DEFAULT_TOP_K = 5
TEST_TIMEOUT = 30.0
MIN_ANOMALIES_FOR_LOW_GRADE = 2
MAX_HIGHLIGHTS = 5


class TestBuildPayload:
    def test_build_payload_shape(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
        sample_drivers: list[FeatureDriver],
        sample_categorical: dict[str, CategoricalProfile],
    ) -> None:
        payload = build_payload(
            dqi=sample_dqi,
            correlations=sample_correlations,
            drivers=sample_drivers,
            categorical=sample_categorical,
            target="target",
        )

        required_keys = {
            "metadata",
            "shape",
            "dqi",
            "columns",
            "top_correlations",
            "top_drivers",
            "categorical",
        }
        assert set(payload.keys()) == required_keys

    def test_build_payload_drops_sample_values(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
        sample_drivers: list[FeatureDriver],
        sample_categorical: dict[str, CategoricalProfile],
    ) -> None:
        # Create a DQI with sample_values populated in one column (frozen allows construction)
        col_with_samples = ColumnHealth(
            name="feat_1",
            data_type=DataType.NUMERIC,
            total_count=1000,
            null_count=10,
            null_percentage=1.0,
            unique_count=950,
            is_constant=False,
            iqr_outliers_count=3,
            sample_values=[1.0, 2.0, 3.0],
        )
        modified_dqi = DataQualityIndex(
            overall_score=sample_dqi.overall_score,
            total_rows=sample_dqi.total_rows,
            total_columns=sample_dqi.total_columns,
            duplicate_rows_count=sample_dqi.duplicate_rows_count,
            duplicate_rows_percentage=sample_dqi.duplicate_rows_percentage,
            total_null_cells=sample_dqi.total_null_cells,
            total_null_percentage=sample_dqi.total_null_percentage,
            health_grade=sample_dqi.health_grade,
            column_health={**sample_dqi.column_health, "feat_1": col_with_samples},
        )

        payload = build_payload(
            dqi=modified_dqi,
            correlations=sample_correlations,
            drivers=sample_drivers,
            categorical=sample_categorical,
            target="target",
        )

        # sample_values should not appear in the payload
        feat_1 = payload["columns"]["feat_1"]
        assert "sample_values" not in feat_1

    def test_build_payload_no_raw_data_leakage(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
        sample_drivers: list[FeatureDriver],
        sample_categorical: dict[str, CategoricalProfile],
    ) -> None:
        payload = build_payload(
            dqi=sample_dqi,
            correlations=sample_correlations,
            drivers=sample_drivers,
            categorical=sample_categorical,
            target="target",
        )

        payload_json = str(payload)

        # Ensure no raw dataframe patterns leak
        assert "head()" not in payload_json
        assert "df[" not in payload_json
        assert "iloc" not in payload_json
        assert "to_numpy" not in payload_json
        assert "to_pandas" not in payload_json

    def test_build_payload_top_k_correlations(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
        sample_drivers: list[FeatureDriver],
        sample_categorical: dict[str, CategoricalProfile],
    ) -> None:
        # Add more correlations to test top_k limit
        many_corrs = sample_correlations + [
            CorrelationPair(
                feature_a=f"feat_{i}",
                feature_b="target",
                pearson_coef=0.9 - i * 0.1,
                spearman_coef=0.88 - i * 0.1,
                strength="Very Strong",
            )
            for i in range(3, 8)
        ]

        payload = build_payload(
            dqi=sample_dqi,
            correlations=many_corrs,
            drivers=sample_drivers,
            categorical=sample_categorical,
            target="target",
            top_k_correlations=DEFAULT_TOP_K,
        )

        assert len(payload["top_correlations"]) == DEFAULT_TOP_K
        # Should be sorted by |pearson|
        pearson_vals = [abs(c["pearson_coef"]) for c in payload["top_correlations"]]
        assert pearson_vals == sorted(pearson_vals, reverse=True)

    def test_build_payload_handles_no_target(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
        sample_drivers: list[FeatureDriver],
        sample_categorical: dict[str, CategoricalProfile],
    ) -> None:
        payload = build_payload(
            dqi=sample_dqi,
            correlations=sample_correlations,
            drivers=None,
            categorical=sample_categorical,
            target=None,
        )

        assert payload["top_drivers"] is None
        assert payload["metadata"]["target"] is None


class TestGenerateNarrative:
    def test_generate_narrative_happy_path(
        self,
        mock_llm_client,
        sample_payload: dict,
    ) -> None:
        expected = ExecutiveSummaryNarrative(
            headline="Test headline",
            key_highlights=["Highlight 1", "Highlight 2", "Highlight 3"],
            data_anomalies=["Anomaly 1"],
            actionable_next_steps=["Step 1", "Step 2"],
        )
        mock = mock_llm_client(return_value=expected)

        result = generate_narrative(sample_payload)

        assert result == expected
        assert mock.call_count == 1

    def test_generate_narrative_timeout_raises_LLMProviderError(
        self,
        mock_llm_client,
        sample_payload: dict,
    ) -> None:
        mock_llm_client(raise_exception=TimeoutError("timed out"))

        with pytest.raises(LLMProviderError) as exc_info:
            generate_narrative(sample_payload)

        assert "timed out" in str(exc_info.value).lower()

    def test_generate_narrative_validation_error_raises_LLMProviderError(
        self,
        mock_llm_client,
        sample_payload: dict,
    ) -> None:
        mock_llm_client(raise_exception=ValidationError.from_exception_data("Test", []))

        with pytest.raises(LLMProviderError) as exc_info:
            generate_narrative(sample_payload)

        assert "schema validation" in str(exc_info.value).lower()

    def test_generate_narrative_uses_settings_defaults(
        self,
        mock_llm_client,
        sample_payload: dict,
    ) -> None:
        expected = ExecutiveSummaryNarrative(
            headline="Test",
            key_highlights=["a", "b", "c"],
            data_anomalies=[],
            actionable_next_steps=[],
        )
        mock = mock_llm_client(return_value=expected)

        generate_narrative(sample_payload)

        # Verify the model string passed to litellm matches settings defaults
        call_kwargs = mock.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == f"{settings.default_provider}/{settings.default_model}"

    def test_generate_narrative_overrides_settings(
        self,
        mock_llm_client,
        sample_payload: dict,
    ) -> None:
        expected = ExecutiveSummaryNarrative(
            headline="Test",
            key_highlights=["a", "b", "c"],
            data_anomalies=[],
            actionable_next_steps=[],
        )
        mock = mock_llm_client(return_value=expected)

        generate_narrative(
            sample_payload,
            provider="openai",
            model="gpt-4o-mini",
            timeout=TEST_TIMEOUT,
        )

        call_kwargs = mock.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["timeout"] == TEST_TIMEOUT


class TestFallback:
    def test_fallback_headline_format(self, sample_dqi: DataQualityIndex) -> None:
        result = generate_heuristic_narrative(sample_dqi, [], None)

        assert "1,000-row" in result.headline
        assert "5-col" in result.headline
        assert "grade B" in result.headline
        assert "DQI 85" in result.headline

    def test_fallback_anomalies_populated_for_low_grade(self) -> None:
        dqi = DataQualityIndex(
            overall_score=45.0,
            total_rows=100,
            total_columns=3,
            duplicate_rows_count=10,
            duplicate_rows_percentage=10.0,
            total_null_cells=30,
            total_null_percentage=10.0,
            health_grade="F",
            column_health={
                "a": ColumnHealth(
                    name="a",
                    data_type=DataType.NUMERIC,
                    total_count=100,
                    null_count=30,
                    null_percentage=30.0,
                    unique_count=70,
                    is_constant=False,
                    iqr_outliers_count=0,
                ),
                "b": ColumnHealth(
                    name="b",
                    data_type=DataType.NUMERIC,
                    total_count=100,
                    null_count=0,
                    null_percentage=0.0,
                    unique_count=1,
                    is_constant=True,
                    iqr_outliers_count=0,
                ),
                "c": ColumnHealth(
                    name="c",
                    data_type=DataType.NUMERIC,
                    total_count=100,
                    null_count=0,
                    null_percentage=0.0,
                    unique_count=100,
                    is_constant=False,
                    iqr_outliers_count=0,
                ),
            },
        )

        result = generate_heuristic_narrative(dqi, [], None)

        assert len(result.data_anomalies) >= MIN_ANOMALIES_FOR_LOW_GRADE
        # Should mention null rate, duplicates, constant columns
        anomalies_text = " ".join(result.data_anomalies)
        assert "Null rate" in anomalies_text
        assert "Duplicate rows" in anomalies_text
        assert "constant column" in anomalies_text

    def test_fallback_handles_empty_inputs(self) -> None:
        dqi = DataQualityIndex(
            overall_score=100.0,
            total_rows=10,
            total_columns=2,
            duplicate_rows_count=0,
            duplicate_rows_percentage=0.0,
            total_null_cells=0,
            total_null_percentage=0.0,
            health_grade="A",
            column_health={
                "x": ColumnHealth(
                    name="x",
                    data_type=DataType.NUMERIC,
                    total_count=10,
                    null_count=0,
                    null_percentage=0.0,
                    unique_count=10,
                    is_constant=False,
                    iqr_outliers_count=0,
                ),
                "y": ColumnHealth(
                    name="y",
                    data_type=DataType.NUMERIC,
                    total_count=10,
                    null_count=0,
                    null_percentage=0.0,
                    unique_count=10,
                    is_constant=False,
                    iqr_outliers_count=0,
                ),
            },
        )

        result = generate_heuristic_narrative(dqi, [], None)

        assert isinstance(result, ExecutiveSummaryNarrative)
        assert len(result.key_highlights) >= 1
        assert len(result.data_anomalies) >= 1
        assert len(result.actionable_next_steps) >= 1

    def test_fallback_returns_valid_pydantic(self, sample_dqi: DataQualityIndex) -> None:
        result = generate_heuristic_narrative(sample_dqi, [], None)

        # Should round-trip through model_validate
        revalidated = ExecutiveSummaryNarrative.model_validate(result.model_dump())
        assert revalidated == result

    def test_fallback_highlights_top_correlations(
        self,
        sample_dqi: DataQualityIndex,
        sample_correlations: list[CorrelationPair],
    ) -> None:
        result = generate_heuristic_narrative(sample_dqi, sample_correlations, None)

        # Should mention the top correlation (feat_1 vs target, r=0.92)
        highlights_text = " ".join(result.key_highlights)
        assert "feat_1" in highlights_text
        assert "target" in highlights_text
        assert "0.92" in highlights_text or "positively" in highlights_text

    def test_fallback_highlights_top_drivers(
        self,
        sample_dqi: DataQualityIndex,
        sample_drivers: list[FeatureDriver],
    ) -> None:
        result = generate_heuristic_narrative(sample_dqi, [], sample_drivers)

        highlights_text = " ".join(result.key_highlights)
        assert "feat_1" in highlights_text
        assert "0.65" in highlights_text or "65" in highlights_text


class TestSchemaExecutiveSummaryNarrative:
    def test_executive_summary_narrative_frozen(self) -> None:
        narrative = ExecutiveSummaryNarrative(
            headline="Test",
            key_highlights=["a", "b", "c"],
            data_anomalies=[],
            actionable_next_steps=[],
        )

        with pytest.raises(ValidationError):
            narrative.headline = "Modified"  # type: ignore[misc]

    def test_executive_summary_narrative_min_lengths(self) -> None:
        # key_highlights must have at least 1 (we don't enforce 3-5 at model level,
        # but we can test it accepts valid lists)
        narrative = ExecutiveSummaryNarrative(
            headline="Test",
            key_highlights=["a", "b", "c", "d", "e"],  # 5 highlights
            data_anomalies=["anom"],
            actionable_next_steps=["step"],
        )
        assert len(narrative.key_highlights) == MAX_HIGHLIGHTS

    def test_executive_summary_narrative_all_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            ExecutiveSummaryNarrative(headline="Test")  # missing required fields


class TestAIInitExports:
    def test_ai_init_exports(self) -> None:
        assert callable(build_payload)
        assert callable(generate_narrative)
        assert callable(generate_heuristic_narrative)
        assert ExecutiveSummaryNarrative is not None
