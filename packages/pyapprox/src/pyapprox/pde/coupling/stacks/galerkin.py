r"""An skfem Galerkin subdomain as a Dirichlet-to-Neumann response, or
as a port-loaded one (Neumann-to-Dirichlet).

The subdomain's rows are tested equations, so the residual left on a
port row pinned to the port data is the boundary flux tested against
that row's basis function: the constraint reaction, which
:class:`~pyapprox.pde.coupling.pinned_response.PinnedReactionFlux` returns
as the port's output. For the same reason a tested flux given as port
data enters as a natural load on the port rows
(:func:`skfem_loaded_response`).

Bound to skfem and the SciPy stack (:func:`~pyapprox.pde.galerkin.
time_integration.skfem_discretization.skfem_discretization`): states are
NumPy columns and the Jacobian is CSR, so nothing built here is
differentiable through.
"""

from typing import Any, Optional, Sequence

from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.pde.coupling.loaded_response import (
    PortRowsOutput,
    SteadyLoadedResponse,
)
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    PinnedReactionFlux,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.ports import BackendPortRows, PortMapProtocol
from pyapprox.pde.coupling.responses import Flux, PortQuantity, Trace
from pyapprox.pde.galerkin.protocols.system import GalerkinTransientSystemProtocol
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import CSRFormat, NumpyRowSet
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

# NumPy backend arrays: the dtype varies.
_Np = NDArray[Any]


def skfem_pinned_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    setups: SetupStrategyProtocol[SteadyRequest[_Np, CSRFormat], _Np, CSRFormat],
    nonlinear: NonlinearSolverProtocol[_Np, _Np],
    time: Optional[float] = None,
) -> SteadyPinnedResponse[_Np, _Np, CSRFormat]:
    """An skfem subdomain whose field DOFs ``ports`` are pinned to the
    port data, in that order, returning the reaction there.

    Parameters
    ----------
    system
        The composed skfem system. A port DOF must not be an essential
        row: wrap a condition covering one in
        :class:`~pyapprox.pde.boundary.except_dofs.ExceptDofs`.
    ports
        The port DOFs, in port order; for ``k`` components, node-major.
    setups
        Sets up the CSR Jacobian of the port-pinned residual, e.g.
        ``RecomputeAlways(ScipySparseLU())``.
    nonlinear
        Solves the subdomain problem.
    time
        The time the data are frozen at; omitted, the system must be
        declared time-invariant (:func:`~pyapprox.pde.models.galerkin.
        steady_residual.galerkin_steady_residual`).

    Raises
    ------
    ValueError
        If there are no port DOFs.
    """
    if len(ports) == 0:
        raise ValueError("an skfem response needs at least one port DOF")
    bkd = NumpyBkd()
    disc = skfem_discretization(system)
    states, theta = disc.space(), disc.parameter_space(len(ports))
    data: PortPinnedData[_Np, _Np] = PortPinnedData(
        NumpyRowSet(states.descriptor(), list(ports)),
        BackendPortRows(states.descriptor(), list(ports), bkd),
        BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc, time=time)
    return SteadyPinnedResponse(
        residual,
        data,
        PinnedReactionFlux(residual, data),
        states,
        setups,
        nonlinear,
        bkd,
    )


def skfem_loaded_response(
    system: GalerkinTransientSystemProtocol[_Np],
    ports: Sequence[int],
    setups: SetupStrategyProtocol[SteadyRequest[_Np, CSRFormat], _Np, CSRFormat],
    nonlinear: NonlinearSolverProtocol[_Np, _Np],
    load: Optional[PortMapProtocol[_Np]] = None,
    output: Optional[PortMapProtocol[_Np]] = None,
    receives: PortQuantity = Flux(),
    returns: PortQuantity = Trace(),
    time: Optional[float] = None,
) -> SteadyLoadedResponse[_Np, _Np, CSRFormat]:
    """An skfem subdomain whose port data enter as a natural load on its
    field DOFs ``ports``, in that order, returning the state there (see
    :mod:`~pyapprox.pde.coupling.loaded_response`).

    Parameters
    ----------
    system
        The composed skfem system. A port DOF must not be an essential
        row.
    ports
        The port DOFs, in port order; for ``k`` components, node-major.
    setups
        Sets up the CSR Jacobian, e.g. ``RecomputeAlways(ScipySparseLU())``.
    nonlinear
        Solves the subdomain problem.
    load
        Maps the port data to the load on the port rows; omitted, the
        identity: the data are a flux tested against the port DOFs' basis
        functions, in Neumann form (outward).
    output
        Maps the port DOFs' values to the output; omitted, the identity:
        the trace.
    receives, returns
        The kinds of the port data and the output; a flux in and a trace
        out by default.
    time
        As :func:`skfem_pinned_response`.

    Raises
    ------
    ValueError
        If there are no port DOFs, or as :class:`~pyapprox.pde.coupling.
        loaded_response.PortLoadResidual`.
    """
    if len(ports) == 0:
        raise ValueError("an skfem response needs at least one port DOF")
    bkd = NumpyBkd()
    disc = skfem_discretization(system)
    states, theta = disc.space(), disc.parameter_space(len(ports))
    data: PortPinnedData[_Np, _Np] = PortPinnedData(
        NumpyRowSet(states.descriptor(), list(ports)),
        BackendPortRows(states.descriptor(), list(ports), bkd),
        BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc, time=time)
    return SteadyLoadedResponse(
        residual,
        data,
        PortRowsOutput(data, output),
        states,
        setups,
        nonlinear,
        bkd,
        load,
        receives,
        returns,
    )
