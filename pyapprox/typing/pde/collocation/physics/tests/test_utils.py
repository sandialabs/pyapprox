"""Test utilities for physics implementations.

Provides wrappers and base classes for testing physics with DerivativeChecker
and NewtonSolver.
"""

import unittest
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.physics.base import AbstractPhysics
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.optimization.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
)


class PhysicsDerivativeWrapper(Generic[Array]):
    """Wrap physics for DerivativeChecker compatibility.

    Adapts physics interface (residual/jacobian with state and time) to
    FunctionWithJacobianProtocol interface (samples array).

    Parameters
    ----------
    physics : AbstractPhysics[Array]
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

    def __init__(self, physics: AbstractPhysics[Array], time: float = 0.0):
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
        residual = self._physics.residual(state, self._time)
        return residual[:, None]

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
        return self._physics.jacobian(state, self._time)


class PhysicsNewtonResidual(Generic[Array]):
    """Wrap physics for NewtonSolver compatibility.

    Adapts physics with boundary conditions to NewtonSolverResidualProtocol.

    Parameters
    ----------
    physics : AbstractPhysics[Array]
        Physics with boundary conditions set via set_boundary_conditions().
    time : float
        Time value to use in residual/jacobian calls.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> physics = SomePhysics(basis, bkd)
    >>> physics.set_boundary_conditions([bc1, bc2])
    >>> wrapper = PhysicsNewtonResidual(physics, time=0.0)
    >>> solver = NewtonSolver(wrapper)
    >>> solver.set_options(maxiters=50, atol=1e-10)
    >>> solution = solver.solve(initial_guess)
    """

    def __init__(self, physics: AbstractPhysics[Array], time: float = 0.0):
        self._physics = physics
        self._time = time
        self._bkd = physics.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, iterate: Array) -> Array:
        """Compute residual at iterate with boundary conditions applied.

        Parameters
        ----------
        iterate : Array
            Current iterate. Shape: (nstates,).

        Returns
        -------
        Array
            Residual with BCs. Shape: (nstates,).
        """
        residual = self._physics.residual(iterate, self._time)
        jacobian = self._physics.jacobian(iterate, self._time)
        residual_bc, _ = self._physics.apply_boundary_conditions(
            residual, jacobian, iterate, self._time
        )
        return residual_bc

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        """Solve linear system J @ delta = prev_residual.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,).
        prev_residual : Array
            Residual at current state. Shape: (nstates,).

        Returns
        -------
        Array
            Newton step delta. Shape: (nstates,).
        """
        residual = self._physics.residual(state, self._time)
        jacobian = self._physics.jacobian(state, self._time)
        _, jacobian_bc = self._physics.apply_boundary_conditions(
            residual, jacobian, state, self._time
        )
        return self._bkd.solve(jacobian_bc, prev_residual)


class PhysicsTestBase(Generic[Array], unittest.TestCase):
    """Base class for physics tests with derivative checking utilities.

    Provides helper methods for testing physics implementations using
    DerivativeChecker.

    Subclasses must implement `bkd()` method and set `__test__ = True`.

    Examples
    --------
    >>> class TestMyPhysics(PhysicsTestBase):
    ...     __test__ = True
    ...     def bkd(self):
    ...         return NumpyBkd()
    ...
    ...     def test_jacobian(self):
    ...         physics = MyPhysics(basis, self.bkd())
    ...         state = self.bkd().randn((physics.nstates(),))
    ...         self.check_jacobian(physics, state)
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Return the computational backend.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    def check_jacobian(
        self,
        physics: AbstractPhysics[Array],
        state: Array,
        time: float = 0.0,
        atol: float = 1e-5,
    ) -> None:
        """Verify physics Jacobian using DerivativeChecker.

        Parameters
        ----------
        physics : AbstractPhysics[Array]
            Physics to test.
        state : Array
            State at which to check Jacobian. Shape: (nstates,).
        time : float
            Time value.
        atol : float
            Absolute tolerance for derivative error.

        Raises
        ------
        AssertionError
            If minimum derivative error exceeds tolerance.
        """
        wrapper = PhysicsDerivativeWrapper(physics, time)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None])
        min_error = float(self.bkd().min(errors[0]))
        self.assertLess(
            min_error,
            atol,
            f"Jacobian error {min_error:.2e} exceeds tolerance {atol:.2e}",
        )

    def check_residual_zero(
        self,
        physics: AbstractPhysics[Array],
        exact_state: Array,
        time: float = 0.0,
        atol: float = 1e-10,
    ) -> None:
        """Verify residual is zero at manufactured solution.

        Parameters
        ----------
        physics : AbstractPhysics[Array]
            Physics to test.
        exact_state : Array
            Exact solution. Shape: (nstates,).
        time : float
            Time value.
        atol : float
            Absolute tolerance for residual norm.

        Raises
        ------
        AssertionError
            If residual norm exceeds tolerance.
        """
        residual = physics.residual(exact_state, time)
        norm = float(self.bkd().norm(residual))
        self.assertLess(
            norm,
            atol,
            f"Residual norm {norm:.2e} exceeds tolerance {atol:.2e}",
        )

    def check_jacobian_at_boundary(
        self,
        physics: AbstractPhysics[Array],
        state: Array,
        time: float = 0.0,
        atol: float = 1e-5,
    ) -> None:
        """Verify Jacobian with boundary conditions applied.

        Tests that Jacobian is consistent after applying boundary conditions.

        Parameters
        ----------
        physics : AbstractPhysics[Array]
            Physics with boundary conditions set.
        state : Array
            State at which to check Jacobian. Shape: (nstates,).
        time : float
            Time value.
        atol : float
            Absolute tolerance for derivative error.
        """
        bkd = self.bkd()
        nstates = physics.nstates()

        # Compute analytical Jacobian with BCs
        residual = physics.residual(state, time)
        jacobian = physics.jacobian(state, time)
        residual_bc, jacobian_bc = physics.apply_boundary_conditions(
            residual, jacobian, state, time
        )

        # Compute finite difference Jacobian for residual with BCs
        eps = 1e-7
        jac_fd = bkd.zeros((nstates, nstates))

        for j in range(nstates):
            state_plus = bkd.copy(state)
            state_plus[j] = state_plus[j] + eps
            state_minus = bkd.copy(state)
            state_minus[j] = state_minus[j] - eps

            # Compute residual with BCs at perturbed states
            res_plus = physics.residual(state_plus, time)
            jac_plus = physics.jacobian(state_plus, time)
            res_plus_bc, _ = physics.apply_boundary_conditions(
                res_plus, jac_plus, state_plus, time
            )

            res_minus = physics.residual(state_minus, time)
            jac_minus = physics.jacobian(state_minus, time)
            res_minus_bc, _ = physics.apply_boundary_conditions(
                res_minus, jac_minus, state_minus, time
            )

            for i in range(nstates):
                jac_fd[i, j] = (res_plus_bc[i] - res_minus_bc[i]) / (2 * eps)

        # Compare
        bkd.assert_allclose(jacobian_bc, jac_fd, atol=atol)
