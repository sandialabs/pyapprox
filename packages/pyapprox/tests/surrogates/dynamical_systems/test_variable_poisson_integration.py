"""Integration tests for VariablePoissonFixedHamiltonianSurrogate.

Tests:
1. Wrap in BatchedBoundODEResidual and integrate a 2D rotation system
2. FD-verified adjoint gradient via DerivativeChecker
3. Energy conservation drift bounded by CN truncation error
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.tableau import crank_nicolson
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.surrogates.dynamical_systems import solve_batched, step_times


def _make_rotation_surrogate(
    bkd: Backend[Array], omega: float = 1.0
) -> VariablePoissonFixedHamiltonianSurrogate[Array]:
    """2D rotation system: H=0.5*||x||^2, L=[[0,omega],[-omega,0]].

    RHS = L @ grad H = L @ x = [[0,omega],[-omega,0]] @ [q,p]
        = [omega*p, -omega*q]

    Analytic: q(t)=q0*cos(wt)+p0*sin(wt), p(t)=-q0*sin(wt)+p0*cos(wt)
    (same as SHO with omega=1 when omega=1).
    """
    n_dynamic = 2

    def grad_H(samples: Array) -> Array:
        return samples[:n_dynamic, :]

    def grad_H_jac(samples: Array) -> Array:
        nsamples = samples.shape[1]
        eye = bkd.eye(n_dynamic)
        return bkd.tile(
            bkd.reshape(eye, (1, n_dynamic, n_dynamic)),
            (nsamples, 1, 1),
        )

    surrogate = VariablePoissonFixedHamiltonianSurrogate(
        grad_hamiltonian=grad_H,
        grad_hamiltonian_jacobian=grad_H_jac,
        n_dynamic=n_dynamic,
        bkd=bkd,
    )
    eta = bkd.array([omega])
    return surrogate.with_params(eta)


def _rotation_analytic(
    q0: float, p0: float, omega: float, times_np: NDArray[np.float64]
) -> NDArray[np.float64]:
    q = q0 * np.cos(omega * times_np) + p0 * np.sin(omega * times_np)
    p = -q0 * np.sin(omega * times_np) + p0 * np.cos(omega * times_np)
    return np.stack([q, p], axis=0)


def _method(bkd: Backend[Array]) -> SteppingMethodProtocol[Array, Array]:
    """Crank-Nicolson with Newton at atol and rtol 1e-12."""
    newton: Newton[Array, Array] = Newton(maxiters=30, atol=1e-12, rtol=1e-12)
    return RungeKutta(crank_nicolson(), newton)


class TestVariablePoissonIntegration:
    def test_wrapped_in_batched_residual(self, bkd: Backend[Array]) -> None:
        """2D rotation plugs into BatchedBoundODEResidual and integrates."""
        omega = 1.0
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        q0, p0 = 1.0, 0.0
        init_state = bkd.array([q0, p0])
        approx_sol, times = solve_batched(
            wrapper, _method(bkd), init_state, step_times(0.0, 2 * np.pi, 0.01)
        )

        times_np: NDArray[np.float64] = np.asarray(bkd.to_numpy(times))
        ref_sol = bkd.array(_rotation_analytic(q0, p0, omega, times_np))
        max_err = bkd.to_float(bkd.max(bkd.abs(approx_sol - ref_sol)))
        assert max_err < 1e-4, f"max error {max_err:.2e} exceeds 1e-4"

    @pytest.mark.slow_on("TorchBkd")
    def test_adjoint_gradient(self, bkd: Backend[Array]) -> None:
        """FD-verified adjoint gradient via DerivativeChecker."""
        omega = 1.5
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        init_state = bkd.array([1.0, 0.0])
        times = step_times(0.0, 0.5, 0.02)
        fwd_sols, _ = solve_batched(wrapper, _method(bkd), init_state, times)
        nstates, ntimes = fwd_sols.shape
        obs_time_indices = bkd.arange(ntimes)
        obs_tuples = [(i, obs_time_indices) for i in range(nstates)]
        flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))

        rng = np.random.RandomState(123)
        perturbation = bkd.array(rng.randn(nstates * ntimes) * 0.01)
        flat_obs = flat_obs + perturbation

        loss = TrajectoryMatchingLoss(
            wrapper=wrapper,
            method=_method(bkd),
            times=times,
            init_states=init_state,
            obs_tuples=obs_tuples,
            observations=flat_obs,
            noise_std=1.0,
        )

        eta = surrogate.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = bkd.to_float(checker.error_ratio(errors[0]))
        assert ratio <= 1e-6, f"error_ratio={ratio:.2e}"

    def test_energy_conservation_drift(self, bkd: Backend[Array]) -> None:
        """H=0.5*||x||^2 is conserved; drift bounded by CN truncation error."""
        omega = 1.0
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        q0, p0 = 1.0, 0.0
        init_state = bkd.array([q0, p0])
        approx_sol, _ = solve_batched(
            wrapper, _method(bkd), init_state, step_times(0.0, 2 * np.pi, 0.01)
        )

        q_traj = approx_sol[0, :]
        p_traj = approx_sol[1, :]
        H_traj = 0.5 * (q_traj**2 + p_traj**2)
        H0 = 0.5 * (q0**2 + p0**2)
        drift = bkd.to_float(bkd.max(bkd.abs(H_traj - H0)))
        assert drift < 1e-4, f"energy drift {drift:.2e} exceeds 1e-4"
