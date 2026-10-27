"""PDE operator-inference problem bundle.

Bundles a parameterized Galerkin FEM semi-discretization with the
metadata operator inference needs: the monomial degree set of the
exact reduced dynamics, the input dimension, the boundary lift, and
the time-integration defaults.  Composition, not inheritance --
mirrors ``ODEForwardUQProblem``: the problem returns a solvable
``GalerkinTransientModel`` and the caller drives it (``model.solve_transient``,
``model.system()``); the problem itself never solves.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from numpy.typing import NDArray

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration import (
    GalerkinTransientModel,
    skfem_ode_residual,
)
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.protocols import DomainProtocol

# Bound to NumPy: the systems assemble with skfem.
_Np = NDArray[Any]


class PDEOpInfProblem:
    """Parameterized PDE problem for operator-inference recovery.

    The semi-discrete FOM is ``M du/dt = f(u; mu)`` on a FIXED basis:
    ``model`` rebuilds the system per parameter value on the one
    shared basis, so snapshots, projections, and the intrusive reduced
    operators all live on the identical discretization.

    Parameters
    ----------
    name : str
        Problem name.
    system_factory : Callable[[_Np], GalerkinTransientSystemProtocol[_Np]]
        Builds the system (physics with its boundary conditions) for
        parameters of shape ``(nparams, 1)`` on the shared basis.
    basis : GalerkinBasisProtocol[_Np]
        The shared finite element basis.
    prior : DistributionProtocol[_Np]
        Prior distribution over the parameters.
    domain : DomainProtocol[_Np]
        Parameter domain (bounds).
    time_config : TimeIntegrationConfig
        Default time integration configuration for FOM solves.
    initial_condition : _Np
        Initial state.  Shape: ``(nstates,)``.
    nominal_parameters : _Np
        Nominal parameter values.  Shape: ``(nparams, 1)``.
    degree_set : tuple[int, ...]
        Monomial degrees present in the EXACT reduced dynamics (e.g.
        ``(1, 2)`` for Burgers, ``(1, 3)`` for Chafee-Infante).  No
        degree-0 term: homogeneous problems have no constant column.
    ninputs : int
        Number of exogenous inputs entering the reduced dynamics
        (0 for the homogeneous problems).
    lift_vector : _Np
        Boundary lift ``l`` such that ``u = u_hom + l*g`` homogenizes
        the Dirichlet data.  Shape: ``(nstates,)``; zero when the
        boundary data is homogeneous.
    bkd : NumpyBkd
        Computational backend.
    input_func : Callable[[float], float], optional
        Exogenous input ``g(t)``; ``None`` when ``ninputs == 0``.
    input_deriv_func : Callable[[float], float], optional
        Input derivative ``g'(t)``; ``None`` when ``ninputs == 0``.
    description : str
        Human-readable description.
    reference : str
        Literature reference.
    """

    def __init__(
        self,
        name: str,
        system_factory: Callable[
            [_Np], GalerkinTransientSystemProtocol[_Np]
        ],
        basis: GalerkinBasisProtocol[_Np],
        prior: DistributionProtocol[_Np],
        domain: DomainProtocol[_Np],
        time_config: TimeIntegrationConfig[_Np],
        initial_condition: _Np,
        nominal_parameters: _Np,
        degree_set: Tuple[int, ...],
        ninputs: int,
        lift_vector: _Np,
        bkd: NumpyBkd,
        input_func: Optional[Callable[[float], float]] = None,
        input_deriv_func: Optional[Callable[[float], float]] = None,
        description: str = "",
        reference: str = "",
    ) -> None:
        self._name = name
        self._system_factory = system_factory
        self._basis = basis
        self._prior = prior
        self._domain = domain
        self._time_config = time_config
        self._initial_condition = initial_condition
        self._nominal_parameters = nominal_parameters
        self._degree_set = degree_set
        self._ninputs = ninputs
        self._lift_vector = lift_vector
        self._bkd = bkd
        self._input_func = input_func
        self._input_deriv_func = input_deriv_func
        self._description = description
        self._reference = reference

    def name(self) -> str:
        """Return the problem name."""
        return self._name

    def basis(self) -> GalerkinBasisProtocol[_Np]:
        """Return the shared finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return the number of state variables (dofs)."""
        return self._basis.ndofs()

    def prior(self) -> DistributionProtocol[_Np]:
        """Return prior distribution over parameters."""
        return self._prior

    def domain(self) -> DomainProtocol[_Np]:
        """Return parameter domain."""
        return self._domain

    def time_config(self) -> TimeIntegrationConfig[_Np]:
        """Return the default time integration configuration."""
        return self._time_config

    def initial_condition(self) -> _Np:
        """Return initial state. Shape: ``(nstates,)``."""
        return self._initial_condition

    def nominal_parameters(self) -> _Np:
        """Return nominal parameter values. Shape: ``(nparams, 1)``."""
        return self._nominal_parameters

    def nparams(self) -> int:
        """Return the number of parameters."""
        return int(self._nominal_parameters.shape[0])

    def degree_set(self) -> Tuple[int, ...]:
        """Return monomial degrees of the exact reduced dynamics."""
        return self._degree_set

    def ninputs(self) -> int:
        """Return the number of exogenous inputs."""
        return self._ninputs

    def input_func(self) -> Optional[Callable[[float], float]]:
        """Return the exogenous input ``g(t)``, or ``None``."""
        return self._input_func

    def input_deriv_func(self) -> Optional[Callable[[float], float]]:
        """Return the input derivative ``g'(t)``, or ``None``."""
        return self._input_deriv_func

    def lift_vector(self) -> _Np:
        """Return the boundary lift. Shape: ``(nstates,)``."""
        return self._lift_vector

    def bkd(self) -> NumpyBkd:
        """Return the computational backend."""
        return self._bkd

    def description(self) -> str:
        """Return human-readable description."""
        return self._description

    def reference(self) -> str:
        """Return literature reference."""
        return self._reference

    def model(
        self, parameters: Optional[_Np] = None
    ) -> GalerkinTransientModel[_Np]:
        """Build the transient FEM model at the given parameters.

        Rebuilds the system per parameter value on the SHARED basis
        and wraps it in a solvable ``GalerkinTransientModel`` with the
        consistent mass; the caller
        drives it (``model.solve_transient(problem.initial_condition(),
        problem.time_config())``) and reaches the semi-discrete
        operator through ``model.system()``.  Downstream recovery
        code must use ONE returned model for both snapshot generation
        and the intrusive reference -- two separately built systems
        agree only up to assembly determinism.

        Parameters
        ----------
        parameters : _Np, optional
            Parameter values of shape ``(nparams, 1)``.  ``None`` uses
            the nominal parameters.

        Returns
        -------
        GalerkinTransientModel
            The solvable model at the requested parameters.
        """
        if parameters is None:
            parameters = self._nominal_parameters
        return GalerkinTransientModel(
            skfem_ode_residual(self._system_factory(parameters))
        )
