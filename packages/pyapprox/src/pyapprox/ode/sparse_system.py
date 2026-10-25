r"""The implicit system of a model whose Jacobian is a SciPy sparse matrix.

Bound to NumPy and SciPy (CLAUDE.md case 3): the mass, the Jacobian and
the shifted Jacobian are CSR operators of the SciPy stack, so this sits
apart from the backend-generic builders in ``semi_explicit_system``.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.ode.protocols import ODEResidualProtocol
from pyapprox.ode.semi_explicit_system import SemiExplicitSystem, mass_properties
from pyapprox.ode.sparse_jacobian_operator import SparseJacobianOperator
from pyapprox.util.linalg.operators import LinearSolverProtocol, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyVectorSpace,
    ScipyCSROperator,
    ScipySparseLU,
)

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]


def sparse_system(
    model: ODEResidualProtocol[_Np],
    solver: Optional[LinearSolverProtocol[_Np, CSRFormat]] = None,
) -> SemiExplicitSystem[_Np, CSRFormat]:
    """The implicit system of a model whose Jacobian is a sparse matrix.

    The mass, the Jacobian and the shifted Jacobian are CSR operators of
    the NumPy/SciPy stack in float64; the solver defaults to sparse LU.
    Its mass is read from the model's ``mass_matrix()``. Bound to NumPy
    and SciPy, so not differentiable.

    Parameters
    ----------
    model : ODEResidualProtocol
        Supplies :math:`f`, its ``jacobian`` as a SciPy sparse matrix,
        and its mass.
    solver : LinearSolverProtocol, optional
        A solver for CSR operators. Defaults to ``ScipySparseLU``.
    """
    mass = model.mass_matrix()
    stored = csr_matrix(mass.as_matrix())
    descriptor = SpaceDescriptor("state", stored.shape[0], "float64")
    return SemiExplicitSystem(
        model,
        ScipyCSROperator(
            stored, descriptor, properties=mass_properties(mass), scalar=np.float64
        ),
        SparseJacobianOperator(model, descriptor),
        NumpyVectorSpace(descriptor, np.float64),
        ScipySparseLU() if solver is None else solver,
    )
