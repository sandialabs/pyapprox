"""Shared color palette and style helpers for tutorial figures."""

from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "primary": "#2C7FB8",
    "secondary": "#E67E22",
    "accent": "#27AE60",
    "reference": "#C0392B",
    "purple": "#8E44AD",
    "gray": "#aaaaaa",
}

# Field colormap following the pyapprox site convention: neon blue glow
# on a black background, zero = black. Shared by every tutorial that
# renders a scalar field, so the series reads as one system.
NEON_CMAP = LinearSegmentedColormap.from_list(
    "pyapprox_neon",
    ["#000000", "#001a4d", "#0057d9", "#00b3ff", "#7df9ff", "#e8ffff"],
)
# Positivity alarm: anything below the zero level renders magenta.
NEON_CMAP.set_under("#ff00ff")


def apply_style(ax):
    """Apply consistent grid and formatting to an axis."""
    ax.grid(True, alpha=0.2)
