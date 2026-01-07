"""
Adjoint method infrastructure for local OED.

The adjoint method is used to efficiently compute gradients and HVPs
for criteria that involve solving a linear system.

For J(w) = x(w)^T @ M0(w) @ x(w) where M1(w) @ x(w) = vec:
    - Forward solve: x = M1^{-1} @ vec
    - Adjoint solve: lambda = M1^{-1} @ (2 * M0 @ x)
    - Gradient: dJ/dw_k = x^T @ M0_k @ x - lambda^T @ M1_k @ x

Classes
-------
QuadraticFunctional
    Computes J(x, w) = x^T @ M0(w) @ x and its derivatives.
LinearResidual
    Represents the linear system M1(w) @ x = vec.
AdjointModel
    Combines functional and residual to compute total gradient.
"""

from .functional import QuadraticFunctional
from .residual import LinearResidual
from .model import AdjointModel

__all__ = [
    "QuadraticFunctional",
    "LinearResidual",
    "AdjointModel",
]
