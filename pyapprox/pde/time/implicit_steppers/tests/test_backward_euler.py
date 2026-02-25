import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.pde.time.benchmarks.nonlinear_decoupled import (
    NonLinearDecoupledODE,
)
from pyapprox.pde.time.implicit_steppers.integrator import (
    ImplicitTimeIntegrator,
)


class TestImplicitTimeIntegration(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_decoupled_nonlinear_ode_backward_euler(self) -> None:
        """
        Test implicit time integration for a decoupled nonlinear ODE using
        Backward Euler.
        """
        bkd = self.bkd()

        # Define problem parameters
        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step

        # Create the residual
        residual = NonLinearDecoupledODE(nstates, True, bkd)
        residual.set_parameters(param)

        # Wrap the residual with Backward Euler
        backward_euler_residual = BackwardEulerResidual(residual)

        # Create the Newton solver
        newton_solver = NewtonSolver(backward_euler_residual)

        # Create the implicit time integrator
        time_integrator: ImplicitTimeIntegrator[Array] = (
            ImplicitTimeIntegrator(
                init_time=init_time,
                final_time=final_time,
                deltat=deltat,
                newton_solver=newton_solver,
                verbosity=0,
            )
        )

        # Initial state
        init_state = bkd.full((nstates,), param[1])

        # Solve the time integration problem
        states, times = time_integrator.solve(init_state)

        # Compute exact states
        scale = bkd.arange(1, nstates + 1, dtype=bkd.double_dtype())
        deltat1, deltat2 = times[1:] - times[:-1]
        t0, t1, t2 = times
        exact_states_list = [bkd.full((nstates,), param[1])]
        exact_states_list.append(
            (
                bkd.sqrt(
                    4
                    * deltat1
                    * scale
                    * param[0] ** 2
                    * exact_states_list[-1]
                    * (t1 + 2)
                    + 1
                )
                - 1
            )
            / (2 * deltat1 * scale * param[0] ** 2 * (t1 + 2))
        )
        exact_states_list.append(
            (
                bkd.sqrt(
                    4
                    * deltat2
                    * scale
                    * param[0] ** 2
                    * exact_states_list[-1]
                    * (t2 + 2)
                    + 1
                )
                - 1
            )
            / (2 * deltat2 * scale * param[0] ** 2 * (t2 + 2))
        )
        exact_states = bkd.stack(exact_states_list, axis=1)

        # Assert that the computed states match the exact states
        bkd.assert_allclose(states, exact_states, atol=1e-7, rtol=1e-7)


# Derived test class for NumPy backend
class TestImplicitTimeIntegrationNumpy(
    TestImplicitTimeIntegration[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestImplicitTimeIntegrationTorch(
    TestImplicitTimeIntegration[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
