"""PDE operator-inference problem bundle.

Bundles a parameterized Galerkin FEM semi-discretization with the
metadata operator inference needs: the monomial degree set of the
exact reduced dynamics, the input dimension, the boundary lift, and
the time-integration defaults.  Composition, not inheritance --
mirrors ``ODEForwardUQProblem``: the problem returns a solvable
``GalerkinModel`` and the caller drives it (``model.solve_transient``,
``model.physics()``); the problem itself never solves.
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, Tuple

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.protocols import DomainProtocol


class PDEOpInfProblem(Generic[Array]):
    """Parameterized PDE problem for operator-inference recovery.

    The semi-discrete FOM is ``M du/dt = f(u; mu)`` on a FIXED basis:
    ``model`` rebuilds the physics per parameter value on the one
    shared basis, so snapshots, projections, and the intrusive reduced
    operators all live on the identical discretization.

    Parameters
    ----------
    name : str
        Problem name.
    physics_factory : Callable[[Array], GalerkinPhysicsProtocol[Array]]
        Builds physics for parameters of shape ``(nparams, 1)`` on the
        shared basis.
    basis : GalerkinBasisProtocol[Array]
        The shared finite element basis.
    prior : DistributionProtocol[Array]
        Prior distribution over the parameters.
    domain : DomainProtocol[Array]
        Parameter domain (bounds).
    time_config : TimeIntegrationConfig
        Default time integration configuration for FOM solves.
    initial_condition : Array
        Initial state.  Shape: ``(nstates,)``.
    nominal_parameters : Array
        Nominal parameter values.  Shape: ``(nparams, 1)``.
    degree_set : tuple[int, ...]
        Monomial degrees present in the EXACT reduced dynamics (e.g.
        ``(1, 2)`` for Burgers, ``(1, 3)`` for Chafee-Infante).  No
        degree-0 term: homogeneous problems have no constant column.
    ninputs : int
        Number of exogenous inputs entering the reduced dynamics
        (0 for the homogeneous problems).
    lift_vector : Array
        Boundary lift ``l`` such that ``u = u_hom + l*g`` homogenizes
        the Dirichlet data.  Shape: ``(nstates,)``; zero when the
        boundary data is homogeneous.
    bkd : Backend[Array]
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
        physics_factory: Callable[[Array], GalerkinPhysicsProtocol[Array]],
        basis: GalerkinBasisProtocol[Array],
        prior: DistributionProtocol[Array],
        domain: DomainProtocol[Array],
        time_config: TimeIntegrationConfig,
        initial_condition: Array,
        nominal_parameters: Array,
        degree_set: Tuple[int, ...],
        ninputs: int,
        lift_vector: Array,
        bkd: Backend[Array],
        input_func: Optional[Callable[[float], float]] = None,
        input_deriv_func: Optional[Callable[[float], float]] = None,
        description: str = "",
        reference: str = "",
    ) -> None:
        self._name = name
        self._physics_factory = physics_factory
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

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the shared finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return the number of state variables (dofs)."""
        return self._basis.ndofs()

    def prior(self) -> DistributionProtocol[Array]:
        """Return prior distribution over parameters."""
        return self._prior

    def domain(self) -> DomainProtocol[Array]:
        """Return parameter domain."""
        return self._domain

    def time_config(self) -> TimeIntegrationConfig:
        """Return the default time integration configuration."""
        return self._time_config

    def initial_condition(self) -> Array:
        """Return initial state. Shape: ``(nstates,)``."""
        return self._initial_condition

    def nominal_parameters(self) -> Array:
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

    def lift_vector(self) -> Array:
        """Return the boundary lift. Shape: ``(nstates,)``."""
        return self._lift_vector

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def description(self) -> str:
        """Return human-readable description."""
        return self._description

    def reference(self) -> str:
        """Return literature reference."""
        return self._reference

    def model(
        self, parameters: Optional[Array] = None
    ) -> GalerkinModel[Array]:
        """Build the transient FEM model at the given parameters.

        Rebuilds the physics per parameter value on the SHARED basis
        and wraps it in a solvable ``GalerkinModel``; the caller
        drives it (``model.solve_transient(problem.initial_condition(),
        problem.time_config())``) and reaches the semi-discrete
        operator through ``model.physics()``.  Downstream recovery
        code must use ONE returned model for both snapshot generation
        and the intrusive reference -- two separately built physics
        instances agree only up to assembly determinism.

        Parameters
        ----------
        parameters : Array, optional
            Parameter values of shape ``(nparams, 1)``.  ``None`` uses
            the nominal parameters.

        Returns
        -------
        GalerkinModel
            The solvable model at the requested parameters.
        """
        if parameters is None:
            parameters = self._nominal_parameters
        return GalerkinModel(
            self._physics_factory(parameters), self._bkd
        )
