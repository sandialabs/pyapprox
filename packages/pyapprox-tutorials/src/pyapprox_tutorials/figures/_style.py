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

# Diverging companions for SIGNED fields (vorticity, adjoint variables,
# sensitivities). NEON_CMAP is sequential, so it renders +x and -x with
# the same color -- for a vortex street, where the alternating signs are
# the whole structure, that erases the signal. These mirror NEON_CMAP's
# cyan ramp through black into a second hue, so zero stays black, both
# extremes glow, and the sign is legible.
#
# The positive half of each is NEON_CMAP's own sequence, so a diverging
# figure sits beside a sequential one without a palette clash.
NEON_DIVERGING = LinearSegmentedColormap.from_list(
    "pyapprox_neon_diverging",
    ["#fff3e8", "#ffd166", "#ff7b00", "#b02e00", "#2a0a00",
     "#000000",
     "#001a4d", "#0057d9", "#00b3ff", "#7df9ff", "#e8ffff"],
)

# Magenta/cyan variant: a more overtly neon pairing. Note magenta is
# also NEON_CMAP's under-range alarm, so prefer NEON_DIVERGING in
# figures that show both a sequential and a diverging field.
NEON_DIVERGING_MAGENTA = LinearSegmentedColormap.from_list(
    "pyapprox_neon_diverging_magenta",
    ["#ffe8ff", "#ff7df9", "#d900b3", "#4d001a",
     "#000000",
     "#001a4d", "#0057d9", "#00b3ff", "#7df9ff", "#e8ffff"],
)


def apply_style(ax):
    """Apply consistent grid and formatting to an axis."""
    ax.grid(True, alpha=0.2)
