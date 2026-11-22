"""PDE operator-inference problem bundle.

Bundles a parameterized Galerkin FEM semi-discretization with the
metadata operator inference needs: the monomial degree set of the
exact reduced dynamics, the input dimension, the boundary lift, and
the time-integration defaults.  Composition, not inheritance --
mirrors ``ODEForwardUQProblem``: the problem returns the semi-discrete
system and its transient form, with the default stepping method and
steps, and the caller drives ``solve_galerkin_transient``; the problem
itself never solves.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from numpy.typing import NDArray

from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_system,
)
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox_benchmarks.protocols import DomainProtocol

# Bound to NumPy: the systems assemble with skfem.
_Np = NDArray[Any]


class PDEOpInfProblem:
    """Parameterized PDE problem for operator-inference recovery.

    The semi-discrete FOM is ``M du/dt = f(u; mu)`` on a FIXED basis:
    ``system`` rebuilds the system per parameter value on the one
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
    method : SteppingMethodProtocol
        Default stepping method for FOM solves, with its nonlinear
        solver when implicit.
    steps : FixedStep
        Default steps for FOM solves.
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

    Raises
    ------
    TypeError
        If ``method`` does not satisfy ``SteppingMethodProtocol``.
    """

    def __init__(
        self,
        name: str,
        system_factory: Callable[[_Np], GalerkinTransientSystemProtocol[_Np]],
        basis: GalerkinBasisProtocol[_Np],
        prior: DistributionProtocol[_Np],
        domain: DomainProtocol[_Np],
        method: SteppingMethodProtocol[_Np, _Np],
        steps: FixedStep,
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
        if not isinstance(method, SteppingMethodProtocol):
            raise TypeError(
                "method must satisfy SteppingMethodProtocol, got "
                f"{type(method).__name__}"
            )
        self._name = name
        self._system_factory = system_factory
        self._basis = basis
        self._prior = prior
        self._domain = domain
        self._method = method
        self._steps = steps
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

    def method(self) -> SteppingMethodProtocol[_Np, _Np]:
        """Return the default stepping method."""
        return self._method

    def steps(self) -> FixedStep:
        """Return the default steps."""
        return self._steps

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

    def system(
        self, parameters: Optional[_Np] = None
    ) -> GalerkinTransientSystemProtocol[_Np]:
        """Build the semi-discrete system at the given parameters.

        Rebuilds the system per parameter value on the SHARED basis.
        Downstream recovery code must use ONE returned system for both
        snapshot generation (through :meth:`transient`) and the
        intrusive reference -- two separately built systems agree only
        up to assembly determinism.

        Parameters
        ----------
        parameters : _Np, optional
            Parameter values of shape ``(nparams, 1)``.  ``None`` uses
            the nominal parameters.
        """
        if parameters is None:
            parameters = self._nominal_parameters
        return self._system_factory(parameters)

    def transient(
        self, system: GalerkinTransientSystemProtocol[_Np]
    ) -> GalerkinTransientSystem[_Np, CSRFormat]:
        """``system`` in the form ``solve_galerkin_transient`` steps,
        with the consistent mass on the SciPy stack; the caller drives
        it with :meth:`method`, :meth:`steps` and
        :meth:`initial_condition` (as a column)."""
        return galerkin_transient_system(skfem_discretization(system), ScipySparseLU())
