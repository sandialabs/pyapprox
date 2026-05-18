"""Integration tests for VariablePoissonFixedHamiltonianSurrogate.

Tests:
1. Wrap in BatchedBoundODEResidual and integrate a 2D rotation system
2. FD-verified adjoint gradient via DerivativeChecker
3. Energy conservation drift bounded by CN truncation error
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


def _make_rotation_surrogate(bkd, omega=1.0):
    """2D rotation system: H=0.5*||x||^2, L=[[0,omega],[-omega,0]].

    RHS = L @ grad H = L @ x = [[0,omega],[-omega,0]] @ [q,p]
        = [omega*p, -omega*q]

    Analytic: q(t)=q0*cos(wt)+p0*sin(wt), p(t)=-q0*sin(wt)+p0*cos(wt)
    (same as SHO with omega=1 when omega=1).
    """
    n_dynamic = 2

    def grad_H(samples):
        return samples[:n_dynamic, :]

    def grad_H_jac(samples):
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
    surrogate = surrogate.with_params(eta)
    return surrogate


def _rotation_analytic(q0, p0, omega, times_np):
    q = q0 * np.cos(omega * times_np) + p0 * np.sin(omega * times_np)
    p = -q0 * np.sin(omega * times_np) + p0 * np.cos(omega * times_np)
    return np.stack([q, p], axis=0)


def _forward_solve_cn(wrapper, init_state, init_time, final_time, deltat):
    stepper = CrankNicolsonAdjoint(wrapper)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=30, atol=1e-12, rtol=1e-12)
    integrator = TimeIntegrator(
        init_time=init_time,
        final_time=final_time,
        deltat=deltat,
        newton_solver=newton,
    )
    return integrator.solve(init_state), integrator


class TestVariablePoissonIntegration:
    def test_wrapped_in_batched_residual(self, bkd):
        """2D rotation plugs into BatchedBoundODEResidual and integrates."""
        omega = 1.0
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        q0, p0 = 1.0, 0.0
        init_state = bkd.array([q0, p0])
        (approx_sol, times), _ = _forward_solve_cn(
            wrapper, init_state, 0.0, 2 * np.pi, 0.01
        )

        times_np = np.asarray(bkd.to_numpy(times))
        ref_sol = bkd.array(_rotation_analytic(q0, p0, omega, times_np))
        max_err = float(bkd.max(bkd.abs(approx_sol - ref_sol)))
        assert max_err < 1e-4, f"max error {max_err:.2e} exceeds 1e-4"

    @pytest.mark.slow_on("TorchBkd")
    def test_adjoint_gradient(self, bkd):
        """FD-verified adjoint gradient via DerivativeChecker."""
        omega = 1.5
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        init_state = bkd.array([1.0, 0.0])
        stepper = CrankNicolsonAdjoint(wrapper)
        newton = NewtonSolver(stepper)
        newton.set_options(maxiters=30, atol=1e-12, rtol=1e-12)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=0.5,
            deltat=0.02,
            newton_solver=newton,
        )

        fwd_sols, times = integrator.solve(init_state)
        nstates = fwd_sols.shape[0]
        ntimes = fwd_sols.shape[1]
        obs_time_indices = bkd.arange(ntimes)
        obs_tuples = [(i, obs_time_indices) for i in range(nstates)]
        flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))

        rng = np.random.RandomState(123)
        perturbation = bkd.array(rng.randn(nstates * ntimes) * 0.01)
        flat_obs = flat_obs + perturbation

        loss = TrajectoryMatchingLoss(
            wrapper=wrapper,
            integrator=integrator,
            init_states=init_state,
            obs_tuples=obs_tuples,
            observations=flat_obs,
            noise_std=1.0,
        )

        eta = surrogate.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    def test_energy_conservation_drift(self, bkd):
        """H=0.5*||x||^2 is conserved; drift bounded by CN truncation error."""
        omega = 1.0
        surrogate = _make_rotation_surrogate(bkd, omega)
        wrapper = BatchedBoundODEResidual(surrogate, n_dynamic=2)

        q0, p0 = 1.0, 0.0
        init_state = bkd.array([q0, p0])
        (approx_sol, times), _ = _forward_solve_cn(
            wrapper, init_state, 0.0, 2 * np.pi, 0.01
        )

        q_traj = approx_sol[0, :]
        p_traj = approx_sol[1, :]
        H_traj = 0.5 * (q_traj**2 + p_traj**2)
        H0 = 0.5 * (q0**2 + p0**2)
        drift = float(bkd.max(bkd.abs(H_traj - H0)))
        assert drift < 1e-4, f"energy drift {drift:.2e} exceeds 1e-4"
