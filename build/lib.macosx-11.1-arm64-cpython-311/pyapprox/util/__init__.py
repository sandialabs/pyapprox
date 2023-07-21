"""The :mod:`pyapprox.util` module implements numerous foundational utilities.
"""

from pyapprox.util.utilities import (
    cartesian_product, outer_product, check_gradients
)
from pyapprox.util.visualization import plot_2d_samples

__all__ = ["cartesian_product", "outer_product", "check_gradients",
           "plot_2d_samples"]
