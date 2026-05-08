"""Shared color palette and style helpers for tutorial figures."""

COLORS = {
    "primary": "#2C7FB8",
    "secondary": "#E67E22",
    "accent": "#27AE60",
    "reference": "#C0392B",
    "purple": "#8E44AD",
    "gray": "#aaaaaa",
}


def apply_style(ax):
    """Apply consistent grid and formatting to an axis."""
    ax.grid(True, alpha=0.2)
