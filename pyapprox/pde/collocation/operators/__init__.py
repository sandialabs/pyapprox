"""Operators module for spectral collocation methods."""

from pyapprox.pde.collocation.operators.jacobian_types import (
    SparseJacobian,
    DenseJacobian,
    DiagJacobian,
    ZeroJacobian,
)
from pyapprox.pde.collocation.operators.field import (
    Field,
    scalar_field,
    input_field,
    constant_field,
    zero_field,
)
from pyapprox.pde.collocation.operators.differential import (
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
