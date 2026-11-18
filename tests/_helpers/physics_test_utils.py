"""Test utilities for physics implementations.

Provides wrappers and base classes for testing physics with DerivativeChecker,
and a steady Newton solve with Dirichlet rows.
"""

from typing import Callable, Generic, Sequence

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrap physics for DerivativeChecker compatibility.

    Adapts the interior operator (residual/jacobian with state and time)
    to FunctionProtocol interface (samples array).

    Parameters
    ----------
    physics : CollocationInteriorOperatorProtocol[Array]
        Physics to wrap.
    time : float
        Time value to use in residual/jacobian calls.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> physics = SomePhysics(basis, bkd)
    >>> wrapper = PhysicsDerivativeWrapper(physics, time=0.0)
    >>> checker = DerivativeChecker(wrapper)
    >>> state = bkd.randn((physics.nstates(),))
    >>> errors = checker.check_derivatives(state[:, None])
    >>> assert float(bkd.min(errors[0])) < 1e-5
    """

    def __init__(
        self, physics: CollocationInteriorOperatorProtocol[Array], time: float = 0.0
    ) -> None:
        self._physics = physics
        self._time = time

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._physics.bkd()

    def nvars(self) -> int:
        """Return number of input variables (nstates)."""
        return self._physics.nstates()

    def nqoi(self) -> int:
        """Return number of output quantities (nstates)."""
        return self._physics.nstates()

    def __call__(self, samples: Array) -> Array:
        """Evaluate residual at samples.

        Parameters
        ----------
        samples : Array
            Samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Values of shape (nsamples, nqoi).
        """
        # Handle both (nvars,) and (nvars, nsamples) inputs
        if samples.ndim == 1:
            state = samples
        else:
            state = samples[:, 0]
        residual = self._physics.interior_residual(state, self._time)
        return residual[:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at sample.

        Parameters
        ----------
        sample : Array
            Sample of shape (nvars, 1) or (nvars,).

        Returns
        -------
        Array
            Jacobian of shape (nqoi, nvars).
        """
        if sample.ndim == 2:
            state = sample[:, 0]
        else:
            state = sample
        return self._physics.interior_jacobian(state, self._time)


class ManufacturedColumn(Generic[Array]):
    """One column of a manufactured ``f(coords, time)`` function, as an
    ``f(coords, time)`` supplier of nodal values.

    Wrap it in ``TimeDependent`` or ``TimeIndependent`` to declare its
    time dependence.

    Parameters
    ----------
    function : Callable[[Array, float], Array]
        Returns shape (npts, ncolumns).
    column : int
        The column to return.
    """

    def __init__(self, function: Callable[[Array, float], Array], column: int) -> None:
        self._function = function
        self._column = column

    def __call__(self, coords: Array, time: float) -> Array:
        """Shape: (npts,)"""
        return self._function(coords, time)[:, self._column]


def solve_steady_dirichlet(
    physics: CollocationInteriorOperatorProtocol[Array],
    ncomponents: int,
    parts: Sequence[DirichletPart[Array]],
    initial_state: Array,
    atol: float,
    rtol: float = 0.0,
    maxiters: int = 50,
) -> Array:
    """The steady state of ``physics`` with the Dirichlet ``parts`` as its
    only boundary rows, by Newton with a dense LU.

    Parameters
    ----------
    physics : CollocationInteriorOperatorProtocol
        The interior operator; every coefficient time-invariant.
    ncomponents : int
        The state's number of components.
    parts : sequence of DirichletPart
        The Dirichlet rows, each on one component.
    initial_state : Array
        Shape: (nstates,).
    atol, rtol : float
        Newton's residual tolerances.
    maxiters : int, default 50
        Newton's iteration limit.

    Returns
    -------
    Array
        The converged state. Shape: (nstates,).
    """
    bkd, nstates = physics.bkd(), physics.nstates()
    essential = DirichletRows(bkd, nstates // ncomponents, ncomponents, list(parts))
    steady = SteadySystem(
        collocation_steady_residual(CollocationSystem(physics, essential, [])),
        backend_state_space(bkd, nstates),
        RecomputeAlways(BackendDenseLU(bkd)),
    )
    newton: Newton[Array, Array] = Newton(atol=atol, rtol=rtol, maxiters=maxiters)
    point = steady.solve(newton, initial_state[:, None])
    assert point.converged(), "Newton did not converge"
    return point.state()[:, 0]


class PhysicsTestBase:
    """Base class for physics tests with derivative checking utilities.

    Provides helper methods for testing physics implementations using
    DerivativeChecker.

    Test methods receive ``bkd`` via pytest fixture injection.
    """

    def check_jacobian(
        self,
        bkd: Backend[Array],
        physics: CollocationInteriorOperatorProtocol[Array],
        state: Array,
        time: float = 0.0,
        atol: float = 1e-5,
    ) -> None:
        """Verify physics Jacobian using DerivativeChecker."""
        wrapper = PhysicsDerivativeWrapper(physics, time)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None])
        min_error = bkd.to_float(bkd.min(errors[0]))
        assert min_error < atol, (
            f"Jacobian error {min_error:.2e} exceeds tolerance {atol:.2e}"
        )

    def check_residual_zero(
        self,
        bkd: Backend[Array],
        physics: CollocationInteriorOperatorProtocol[Array],
        exact_state: Array,
        time: float = 0.0,
        atol: float = 1e-10,
    ) -> None:
        """Verify residual is zero at manufactured solution."""
        residual = physics.interior_residual(exact_state, time)
        norm = bkd.to_float(bkd.norm(residual))
        assert norm < atol, (
            f"Residual norm {norm:.2e} exceeds tolerance {atol:.2e}"
        )
