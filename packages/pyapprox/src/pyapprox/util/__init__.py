"""Utility modules for pyapprox."""

from pyapprox.util.cartesian import (
    cartesian_product_indices,
    cartesian_product_samples,
    outer_product_weights,
)
from pyapprox.util.optional_deps import (
    import_optional_dependency,
    package_available,
)

__all__ = [
    "cartesian_product_indices",
    "cartesian_product_samples",
    "outer_product_weights",
    "package_available",
    "import_optional_dependency",
]
