"""An skfem system discretized on the NumPy/SciPy stack.

Bound to skfem, NumPy and SciPy: the state Jacobian and the mass are
assembled by skfem and stored as SciPy CSR operators, so nothing built
here is differentiable through.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly import ElementDofMapProtocol, ScipyAssemblyStack
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    SkfemJacobianAssembler,
)
from pyapprox.pde.galerkin.time_integration.skfem_mass_assembler import (
    SkfemMassAssembler,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]
_Dofs = NDArray[np.int64]


def skfem_discretization(
    system: GalerkinTransientSystemProtocol[_Np],
    basis: ElementDofMapProtocol[_Dofs],
    lumped_mass: bool = False,
) -> GalerkinStackDiscretization[_Np, _Np, CSRFormat]:
    """An skfem system on the SciPy stack in float64.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system, on the NumPy backend.
    basis : ElementDofMapProtocol
        The system's basis: its element DOF maps give the sparsity
        pattern.
    lumped_mass : bool, default False
        If True, step with the row-sum lumped mass instead of the
        consistent one.
    """
    stack = ScipyAssemblyStack(np.float64)
    descriptor = SpaceDescriptor("state", system.nstates(), "float64")
    return GalerkinStackDiscretization(
        system,
        stack,
        SkfemJacobianAssembler(system, stack, basis, descriptor),
        SkfemMassAssembler(system, stack, basis, descriptor, lumped=lumped_mass),
        descriptor,
    )
