import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, spearmanr

from lucidata.core.exceptions import VisualizationError
from lucidata.core.schema import FeatureDriver
from lucidata.engine.auditor import _infer_type
from lucidata.viz.themes import (
    PALETTE,
    get_diverging_scale,
    get_sequential_colors,
)

MIN_KDE_SAMPLES = 2
MIN_NUMERIC_COLS = 2
DEFAULT_BINS_AUTO = "auto"

DEFAULT_LAYOUT = {
    "template": "plotly_white",
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": PALETTE["text"],
        "size": 12,
    },
    "plot_bgcolor": PALETTE["background"],
    "paper_bgcolor": PALETTE["background"],
    "margin": {"l": 60, "r": 40, "t": 60, "b": 50},
    "xaxis": {"gridcolor": PALETTE["grid"], "zerolinecolor": PALETTE["grid"]},
    "yaxis": {"gridcolor": PALETTE["grid"], "zerolinecolor": PALETTE["grid"]},
}

HEATMAP_LAYOUT = {
    **DEFAULT_LAYOUT,
    "margin": {"l": 100, "r": 40, "t": 80, "b": 100},
}


def _apply_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply LUCIDATA theme to a figure."""
    layout = DEFAULT_LAYOUT.copy()
    if title:
        layout["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": PALETTE["text"]},
        }
    fig.update_layout(layout)
    return fig


def _apply_heatmap_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply theme tuned for correlation heatmap."""
    layout = HEATMAP_LAYOUT.copy()
    if title:
        layout["title"] = {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": PALETTE["text"]},
        }
    fig.update_layout(layout)
    fig.update_xaxes(tickangle=-45, tickfont={"size": 10})
    fig.update_yaxes(tickfont={"size": 10})
    return fig


def _resolve_numeric_columns(df: pl.DataFrame, columns: list[str] | None) -> list[str]:
    """Return column names that are numeric (or datetime converted to numeric epoch)."""
    available = [c for c in columns if c in df.columns] if columns is not None else df.columns
    numeric = []
    for col_name in available:
        dtype = _infer_type(df[col_name])
        if dtype == "numeric":
            numeric.append(col_name)
    return numeric


def _resolve_categorical_columns(df: pl.DataFrame, columns: list[str] | None) -> list[str]:
    """Return column names that are categorical or text."""
    available = [c for c in columns if c in df.columns] if columns is not None else df.columns
    categorical = []
    for col_name in available:
        dtype = _infer_type(df[col_name])
        if dtype in ("categorical", "text"):
            categorical.append(col_name)
    return categorical


def _kde_trace(values: np.ndarray, name: str, color: str) -> go.Scatter:
    """Generate a KDE trace for histogram overlay."""
    clean = values[np.isfinite(values)]
    if len(clean) < MIN_KDE_SAMPLES:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name=name,
            line={"color": color, "width": 1.5, "dash": "dot"},
        )

    try:
        kde = gaussian_kde(clean)
        x_range = np.linspace(clean.min(), clean.max(), 200)
        y = kde(x_range)
        return go.Scatter(
            x=x_range,
            y=y,
            mode="lines",
            name=f"{name} KDE",
            line={"color": color, "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
        )
    except Exception:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name=name,
            line={"color": color, "width": 1.5, "dash": "dot"},
        )


def _render_html(fig: go.Figure, include_plotlyjs: str) -> str:
    """Render figure to standalone HTML string."""
    valid_modes = ("cdn", "inline", "directory")
    if include_plotlyjs not in valid_modes:
        raise VisualizationError(
            f"include_plotlyjs must be one of {valid_modes}, got '{include_plotlyjs}'"
        )
    return fig.to_html(
        include_plotlyjs=include_plotlyjs, full_html=True, config={"displayModeBar": False}
    )


def _placeholder_html(message: str, include_plotlyjs: str = "cdn") -> str:
    """Generate minimal HTML with an informational message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": PALETTE["text"]},
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=PALETTE["background"],
        paper_bgcolor=PALETTE["background"],
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return _render_html(fig, include_plotlyjs)


def plot_distributions(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    *,
    kde: bool = True,
    bins: int | str = DEFAULT_BINS_AUTO,
    include_plotlyjs: str = "cdn",
) -> str:
    """Render histogram grid for numeric columns with optional KDE overlay.

    Args:
        df: Input Polars DataFrame.
        columns: Optional subset of columns to plot. Non-numeric columns are skipped.
        kde: Whether to overlay kernel density estimate.
        bins: Number of bins or binning strategy (passed to plotly).
        include_plotlyjs: Plotly.js inclusion mode ('cdn', 'inline', 'directory').

    Returns:
        Standalone HTML string containing the chart.
    """
    numeric_cols = _resolve_numeric_columns(df, columns)
    if not numeric_cols:
        msg = "No numeric columns available for distribution plots"
        return _placeholder_html(msg, include_plotlyjs)

    n = len(numeric_cols)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)

    colors = get_sequential_colors(n)
    for idx, col_name in enumerate(numeric_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = colors[idx]

        values = df[col_name].to_numpy()
        clean = values[np.isfinite(values)]

        if len(clean) == 0:
            continue

        fig.add_trace(
            go.Histogram(
                x=clean,
                name=col_name,
                marker_color=color,
                opacity=0.7,
                nbinsx=bins if isinstance(bins, int) else None,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        if kde:
            fig.add_trace(_kde_trace(values, col_name, color), row=row, col=col)

    fig = _apply_theme(fig, "Numeric Distributions")
    fig.update_layout(height=300 * n_rows, showlegend=False)
    return _render_html(fig, include_plotlyjs)


def plot_categorical_bars(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    *,
    top_k: int = 10,
    include_plotlyjs: str = "cdn",
) -> str:
    """Render horizontal bar charts of top-K categories per categorical column.

    Args:
        df: Input Polars DataFrame.
        columns: Optional subset of columns to plot. Non-categorical columns are skipped.
        top_k: Number of top categories to display per column.
        include_plotlyjs: Plotly.js inclusion mode ('cdn', 'inline', 'directory').

    Returns:
        Standalone HTML string containing the chart.
    """
    cat_cols = _resolve_categorical_columns(df, columns)
    if not cat_cols:
        msg = "No categorical/text columns available for bar charts"
        return _placeholder_html(msg, include_plotlyjs)

    n = len(cat_cols)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cat_cols)

    for idx, col_name in enumerate(cat_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        values = df[col_name].drop_nulls()
        if len(values) == 0:
            continue

        vc = values.value_counts(sort=True)
        top = vc.head(top_k)
        categories = top[col_name].to_list()
        counts = top["count"].to_list()

        categories_rev = categories[::-1]
        counts_rev = counts[::-1]
        color = get_sequential_colors(1)[0]

        fig.add_trace(
            go.Bar(
                y=categories_rev,
                x=counts_rev,
                orientation="h",
                name=col_name,
                marker_color=color,
                showlegend=False,
                hovertemplate="%{y}: %{x}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig = _apply_theme(fig, "Categorical Distributions (Top-K)")
    fig.update_layout(height=max(300, 250 * n_rows), showlegend=False)
    fig.update_xaxes(title_text="Count")
    return _render_html(fig, include_plotlyjs)


def _compute_correlation_matrix(
    df: pl.DataFrame, columns: list[str], method: str
) -> tuple[np.ndarray, list[str]]:
    """Compute full correlation matrix for given columns using numpy/scipy."""
    numeric_cols = _resolve_numeric_columns(df, columns)
    if len(numeric_cols) < MIN_NUMERIC_COLS:
        return np.array([]), []

    data = df.select(numeric_cols).to_numpy()
    if method == "pearson":
        corr_matrix = np.corrcoef(data, rowvar=False)
    elif method == "spearman":
        corr_matrix, _ = spearmanr(data, nan_policy="omit")
    else:
        raise VisualizationError(f"method must be 'pearson' or 'spearman', got '{method}'")

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix, numeric_cols


def plot_correlation_heatmap(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    *,
    min_abs: float = 0.35,
    method: str = "pearson",
    include_plotlyjs: str = "cdn",
) -> str:
    """Render full correlation matrix heatmap with threshold graying.

    Args:
        df: Input Polars DataFrame.
        columns: Optional subset of columns. Non-numeric columns are skipped.
        min_abs: Absolute correlation threshold; cells with |r| < min_abs are grayed.
        method: Correlation method ('pearson' or 'spearman').
        include_plotlyjs: Plotly.js inclusion mode ('cdn', 'inline', 'directory').

    Returns:
        Standalone HTML string containing the heatmap.
    """
    available_cols = columns if columns is not None else df.columns
    corr_matrix, numeric_cols = _compute_correlation_matrix(df, available_cols, method)

    if len(numeric_cols) < MIN_NUMERIC_COLS:
        return _placeholder_html(
            "At least 2 numeric columns required for correlation heatmap", include_plotlyjs
        )

    n = len(numeric_cols)

    z = corr_matrix
    text = [[f"{z[i, j]:.2f}" for j in range(n)] for i in range(n)]

    colorscale = get_diverging_scale()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=numeric_cols,
            y=numeric_cols,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            colorbar={"title": f"{method.capitalize()} r", "thickness": 15},
            hovertemplate="%{y} × %{x}<br>r=%{z:.3f}<extra></extra>",
        )
    )

    for i in range(n):
        for j in range(n):
            if i != j and abs(z[i, j]) < min_abs:
                fig.add_annotation(
                    x=numeric_cols[j],
                    y=numeric_cols[i],
                    text="",
                    showarrow=False,
                    bgcolor=PALETTE["neutral_mid"],
                    xref="x",
                    yref="y",
                    width=1,
                    height=1,
                )

    fig = _apply_heatmap_theme(fig, f"{method.capitalize()} Correlation Matrix")
    fig.update_layout(height=max(500, 60 * n + 200), width=max(600, 60 * n + 300))
    return _render_html(fig, include_plotlyjs)


def plot_feature_importance(
    drivers: list[FeatureDriver],
    *,
    top_k: int = 10,
    include_plotlyjs: str = "cdn",
) -> str:
    """Render horizontal bar chart of driver importance scores.

    Args:
        drivers: List of FeatureDriver objects (pre-sorted by importance desc).
        top_k: Maximum number of drivers to display.
        include_plotlyjs: Plotly.js inclusion mode ('cdn', 'inline', 'directory').

    Returns:
        Standalone HTML string containing the chart.
    """
    if not drivers:
        return _placeholder_html("No feature drivers available", include_plotlyjs)

    top = drivers[:top_k]
    names = [d.feature_name for d in top][::-1]
    scores = [d.importance_score for d in top][::-1]
    ranks = [d.rank for d in top][::-1]

    color = PALETTE["slate_blue"]

    fig = go.Figure(
        data=go.Bar(
            y=names,
            x=scores,
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )

    fig = _apply_theme(fig, "Feature Driver Importance")
    fig.update_layout(height=max(300, 35 * len(names) + 100), showlegend=False)
    x_max = max(scores) * 1.15 if scores else 1
    fig.update_xaxes(title_text="Normalized Importance", range=[0, x_max])
    fig.update_yaxes(tickfont={"size": 11})

    for _i, (name, score, rank) in enumerate(zip(names, scores, ranks, strict=True)):
        fig.add_annotation(
            x=score + max(scores) * 0.02,
            y=name,
            text=f"#{rank}",
            showarrow=False,
            font={"size": 10, "color": PALETTE["muted_charcoal"]},
            xanchor="left",
            yanchor="middle",
        )

    return _render_html(fig, include_plotlyjs)
