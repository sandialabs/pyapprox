"""An skfem system discretized on the NumPy/SciPy stack.

Bound to skfem, NumPy and SciPy: the state Jacobian and the mass are
assembled by skfem and stored as SciPy CSR operators, so nothing built
here is differentiable through.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly import ScipyAssemblyStack
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.neutralized_rows import (
    NeutralizedRows,
    RowTreatmentFactory,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
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
from pyapprox.util.linalg.operators import LinearSolverProtocol, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]


def skfem_discretization(
    system: GalerkinTransientSystemProtocol[_Np],
    lumped_mass: bool = False,
) -> GalerkinStackDiscretization[_Np, _Np, CSRFormat]:
    """An skfem system on the SciPy stack in float64.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system, on the NumPy backend; its basis's element
        DOF maps give the sparsity pattern.
    lumped_mass : bool, default False
        If True, step with the row-sum lumped mass instead of the
        consistent one.
    """
    stack = ScipyAssemblyStack(np.float64)
    basis = system.basis()
    descriptor = SpaceDescriptor("state", system.nstates(), "float64")
    return GalerkinStackDiscretization(
        system,
        stack,
        SkfemJacobianAssembler(system, stack, basis, descriptor),
        SkfemMassAssembler(system, stack, basis, descriptor, lumped=lumped_mass),
        descriptor,
    )


def skfem_ode_residual(
    system: GalerkinTransientSystemProtocol[_Np],
    lumped_mass: bool = False,
    solver: Optional[LinearSolverProtocol[_Np, CSRFormat]] = None,
    row_treatment: RowTreatmentFactory[_Np] = NeutralizedRows,
) -> GalerkinPhysicsToODEResidualAdapter[_Np, CSRFormat]:
    """The time-integration adapter of an skfem system on the SciPy stack.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system, on the NumPy backend.
    lumped_mass : bool, default False
        If True, step with the row-sum lumped mass instead of the
        consistent one.
    solver : LinearSolverProtocol, optional
        Solves the implicit steps' shifted Jacobians. Defaults to sparse
        LU.
    row_treatment : RowTreatmentFactory, optional
        The essential-row strategy; defaults to
        :class:`~pyapprox.pde.galerkin.time_integration.neutralized_rows.NeutralizedRows`.
    """
    return GalerkinPhysicsToODEResidualAdapter(
        skfem_discretization(system, lumped_mass),
        ScipySparseLU() if solver is None else solver,
        row_treatment,
    )
