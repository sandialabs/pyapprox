"""Backward compatibility re-exports.

Use pyapprox.benchmarks.instances.analytic instead.
"""

from pyapprox.benchmarks.instances.analytic.genz import (
    genz_oscillatory_2d,
    genz_product_peak_2d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_oscillatory_5d,
    genz_gaussian_peak_5d,
)

__all__ = [
    "genz_oscillatory_2d",
    "genz_product_peak_2d",
    "genz_corner_peak_2d",
    "genz_gaussian_peak_2d",
    "genz_oscillatory_5d",
    "genz_gaussian_peak_5d",
]
