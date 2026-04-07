"""Operators module for spectral collocation methods."""

from pyapprox.pde.collocation.operators.differential import (
    Divergence,
    Gradient,
    Laplacian,
    divergence,
    gradient,
    laplacian,
)
from pyapprox.pde.collocation.operators.field import (
    Field,
    constant_field,
    input_field,
    scalar_field,
    zero_field,
)
from pyapprox.pde.collocation.operators.jacobian_types import (
    DenseJacobian,
    DiagJacobian,
    SparseJacobian,
    ZeroJacobian,
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
