"""Utility modules for pyapprox.typing."""

from pyapprox.typing.util.cartesian import (
    cartesian_product_indices,
    cartesian_product_samples,
    outer_product_weights,
)
from pyapprox.typing.util.optional_deps import (
    package_available,
    import_optional_dependency,
)

__all__ = [
    "cartesian_product_indices",
    "cartesian_product_samples",
    "outer_product_weights",
    "package_available",
    "import_optional_dependency",
]
