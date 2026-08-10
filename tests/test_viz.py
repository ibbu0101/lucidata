import pytest

from lucidata.core.exceptions import VisualizationError
from lucidata.viz.plots import (
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_distributions,
    plot_feature_importance,
)
from lucidata.viz.themes import get_diverging_scale, get_palette, get_sequential_colors


class TestPlotDistributions:
    def test_returns_html_string(self, df_continuous) -> None:
        html = plot_distributions(df_continuous)
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html
        assert "Plotly.newPlot" in html

    def test_includes_all_numeric_columns(self, df_continuous) -> None:
        html = plot_distributions(df_continuous)
        for col in df_continuous.columns:
            assert col in html

    def test_skips_non_numeric_columns(self, df_mixed) -> None:
        html = plot_distributions(df_mixed)
        assert "numeric" in html
        assert "category" not in html or "category" not in html.split("subplot")[0]

    def test_kde_overlay_present_when_enabled(self, df_continuous) -> None:
        html = plot_distributions(df_continuous, kde=True)
        assert "KDE" in html or "gaussian_kde" not in html

    def test_kde_disabled_when_false(self, df_continuous) -> None:
        html = plot_distributions(df_continuous, kde=False)
        assert "KDE" not in html or "KDE" not in html.split("subplot")[0]

    def test_no_numeric_returns_placeholder(self, df_categorical) -> None:
        html = plot_distributions(df_categorical)
        assert "numeric" in html
        assert "Plotly.newPlot" in html

    def test_datetime_only_returns_placeholder(self, df_datetime_only) -> None:
        html = plot_distributions(df_datetime_only)
        assert "numeric" in html

    def test_columns_filter_respected(self, df_mixed) -> None:
        html = plot_distributions(df_mixed, columns=["numeric"])
        assert "numeric" in html
        assert "datetime" not in html


class TestPlotCategoricalBars:
    def test_returns_html_string(self, df_categorical) -> None:
        html = plot_categorical_bars(df_categorical)
        assert isinstance(html, str)
        assert "<div" in html
        assert "Plotly.newPlot" in html

    def test_default_top_k_is_10(self, df_categorical) -> None:
        html = plot_categorical_bars(df_categorical)
        for col in df_categorical.columns:
            assert col in html

    def test_respects_top_k_override(self, df_categorical) -> None:
        html = plot_categorical_bars(df_categorical, top_k=2)
        assert isinstance(html, str)

    def test_no_categorical_returns_placeholder(self, df_continuous) -> None:
        html = plot_categorical_bars(df_continuous)
        assert "categorical" in html
        assert "columns" in html

    def test_columns_filter_respected(self, df_mixed) -> None:
        html = plot_categorical_bars(df_mixed, columns=["category"])
        assert "category" in html
        assert "numeric" not in html


class TestPlotCorrelationHeatmap:
    def test_returns_html_string(self, df_correlated) -> None:
        html = plot_correlation_heatmap(df_correlated)
        assert isinstance(html, str)
        assert "<div" in html
        assert "Plotly.newPlot" in html

    def test_full_matrix_rendered_for_n_cols(self, df_correlated) -> None:
        html = plot_correlation_heatmap(df_correlated)
        for col in df_correlated.columns:
            assert col in html

    def test_threshold_grays_below_min_abs(self, df_correlated) -> None:
        html = plot_correlation_heatmap(df_correlated, min_abs=0.99)
        assert "Plotly.newPlot" in html

    def test_pearson_vs_spearman(self, df_correlated) -> None:
        html_p = plot_correlation_heatmap(df_correlated, method="pearson")
        html_s = plot_correlation_heatmap(df_correlated, method="spearman")
        assert html_p != html_s

    def test_insufficient_numeric_returns_placeholder(self, df_categorical) -> None:
        html = plot_correlation_heatmap(df_categorical)
        assert "At least 2 numeric columns" in html

    def test_single_numeric_returns_placeholder(self, df_single_column) -> None:
        html = plot_correlation_heatmap(df_single_column)
        assert "At least 2 numeric columns" in html

    def test_invalid_method_raises(self, df_correlated) -> None:
        with pytest.raises(VisualizationError):
            plot_correlation_heatmap(df_correlated, method="invalid")


class TestPlotFeatureImportance:
    def test_returns_html_string(self, sample_drivers) -> None:
        html = plot_feature_importance(sample_drivers)
        assert isinstance(html, str)
        assert "<div" in html
        assert "Plotly.newPlot" in html

    def test_top_k_limits_results(self, sample_drivers) -> None:
        html = plot_feature_importance(sample_drivers, top_k=1)
        assert "feat_1" in html
        assert "feat_2" not in html

    def test_empty_drivers_returns_placeholder(self) -> None:
        html = plot_feature_importance([])
        assert "No feature drivers available" in html

    def test_ranks_in_descending_order(self, sample_drivers) -> None:
        html = plot_feature_importance(sample_drivers)
        assert "#1" in html
        assert "#2" in html


class TestThemeHelpers:
    def test_palette_stable_and_matches_spec(self) -> None:
        palette = get_palette()
        assert palette["slate_blue"] == "#4A6FA5"
        assert palette["emerald_green"] == "#2E8B57"
        assert palette["muted_charcoal"] == "#36454F"
        assert palette["background"] == "#F5F7FA"
        assert palette["grid"] == "#D8DEE9"
        assert palette["text"] == "#2B2D42"
        assert palette["warning"] == "#D67D3E"

    def test_diverging_scale_three_stops(self) -> None:
        scale = get_diverging_scale()
        assert len(scale) == 3  # noqa: PLR2004
        assert scale[0] == "#D67D3E"
        assert scale[1] == "#F5F7FA"
        assert scale[2] == "#4A6FA5"

    def test_sequential_colors_cycles(self) -> None:
        colors = get_sequential_colors(15)  # noqa: PLR2004
        assert len(colors) == 15  # noqa: PLR2004
        base = get_sequential_colors(10)  # noqa: PLR2004
        assert colors[:10] == base
        assert colors[10:15] == base[:5]

    def test_get_palette_returns_copy(self) -> None:
        p1 = get_palette()
        p2 = get_palette()
        p1["slate_blue"] = "#000000"
        assert p2["slate_blue"] == "#4A6FA5"


class TestVisualizationErrorHandling:
    def test_invalid_include_plotlyjs_raises(self, df_continuous) -> None:
        with pytest.raises(VisualizationError) as exc_info:
            plot_distributions(df_continuous, include_plotlyjs="invalid")
        assert "include_plotlyjs must be" in str(exc_info.value)

    def test_heatmap_invalid_method_raises(self, df_correlated) -> None:
        with pytest.raises(VisualizationError) as exc_info:
            plot_correlation_heatmap(df_correlated, method="invalid")
        assert "method must be" in str(exc_info.value)
