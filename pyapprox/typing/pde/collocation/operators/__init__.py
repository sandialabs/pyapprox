"""Operators module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.operators.jacobian_types import (
    SparseJacobian,
    DenseJacobian,
    DiagJacobian,
    ZeroJacobian,
)
from pyapprox.typing.pde.collocation.operators.field import (
    Field,
    scalar_field,
    input_field,
    constant_field,
    zero_field,
)
from pyapprox.typing.pde.collocation.operators.differential import (
    Gradient,
    Divergence,
    Laplacian,
    gradient,
    divergence,
    laplacian,
)

__all__ = [
    # Jacobian types
    "SparseJacobian",
    "DenseJacobian",
    "DiagJacobian",
    "ZeroJacobian",
    # Field class and factory functions
    "Field",
    "scalar_field",
    "input_field",
    "constant_field",
    "zero_field",
    # Differential operators
    "Gradient",
    "Divergence",
    "Laplacian",
    "gradient",
    "divergence",
    "laplacian",
]
