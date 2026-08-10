PALETTE: dict[str, str] = {
    "slate_blue": "#4A6FA5",
    "emerald_green": "#2E8B57",
    "muted_charcoal": "#36454F",
    "background": "#F5F7FA",
    "grid": "#D8DEE9",
    "text": "#2B2D42",
    "warning": "#D67D3E",
    "neutral_low": "#D8DEE9",
    "neutral_mid": "#F5F7FA",
    "neutral_high": "#4A6FA5",
}

DIVERGING_SCALE: list[str] = ["#D67D3E", "#F5F7FA", "#4A6FA5"]

SEQUENTIAL_COLORS: list[str] = [
    "#4A6FA5",
    "#2E8B57",
    "#D67D3E",
    "#8E7CC3",
    "#E07B91",
    "#6C757D",
    "#5D8AA8",
    "#C0392B",
    "#27AE60",
    "#8E44AD",
]


def get_palette() -> dict[str, str]:
    """Return a copy of the color palette to prevent mutation."""
    return PALETTE.copy()


def get_diverging_scale() -> list[str]:
    """Return the diverging color scale for heatmaps."""
    return DIVERGING_SCALE.copy()


def get_sequential_colors(n: int) -> list[str]:
    """Return up to n sequential colors, cycling if n > available."""
    if n <= len(SEQUENTIAL_COLORS):
        return SEQUENTIAL_COLORS[:n]
    cycles = (n + len(SEQUENTIAL_COLORS) - 1) // len(SEQUENTIAL_COLORS)
    return (SEQUENTIAL_COLORS * cycles)[:n]
