"""End-to-end tests: derivative-matching fit -> ODE integration -> trajectory quality."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


def _true_vdp_rhs(states, mu, bkd):
    """Evaluate Van der Pol RHS: dx1/dt=x2, dx2/dt=mu*(1-x1^2)*x2 - x1."""
    x1 = states[0:1, :]
    x2 = states[1:2, :]
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return bkd.vstack([dx1, dx2])


def _solve_true_vdp(mu, y0, times_arr, bkd):
    """Reference VdP solution via scipy RK45."""
    times_np = np.asarray(bkd.to_numpy(times_arr))

    def vdp_rhs(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    sol = solve_ivp(
        vdp_rhs,
        [times_np[0], times_np[-1]],
        y0,
        t_eval=times_np,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    return bkd.array(sol.y)


def _build_expansion(bkd, nvars, nqoi, max_level):
    """Build unfitted polynomial expansion on [-3, 3]^nvars."""
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


def _forward_solve(wrapper, init_state, init_time, final_time, deltat, bkd):
    """Helper to set up stepper/Newton/integrator and solve."""
    stepper = BackwardEulerAdjoint(wrapper)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=30, atol=1e-10, rtol=1e-10)
    integrator = TimeIntegrator(
        init_time=init_time,
        final_time=final_time,
        deltat=deltat,
        newton_solver=newton,
    )
    return integrator.solve(init_state)


class TestEndToEndParameterFreeVdP:
    """Derivative-matching fit -> ODE integration, no mu parameter."""

    def test_trajectory_quality(self, bkd):
        n_dynamic = 2
        max_level = 5

        expansion = _build_expansion(bkd, nvars=2, nqoi=2, max_level=max_level)

        n_train = 500
        rng = np.random.RandomState(42)
        train_states = bkd.array(rng.uniform(-3, 3, (2, n_train)))
        train_derivs = _true_vdp_rhs(train_states, mu=1.0, bkd=bkd)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion, train_states, train_derivs)
        fitted = result.surrogate()

        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=n_dynamic)

        init_state = bkd.array([1.0, 0.0])
        final_time = 3.0
        deltat = 0.005
        approx_sols, times = _forward_solve(
            wrapper, init_state, 0.0, final_time, deltat, bkd
        )

        ref_sol = _solve_true_vdp(1.0, [1.0, 0.0], times, bkd)
        max_err = float(bkd.max(bkd.abs(approx_sols - ref_sol)))
        amplitude = float(bkd.max(bkd.abs(ref_sol)))
        assert max_err / amplitude < 0.05, (
            f"Relative error {max_err / amplitude:.4f} exceeds 0.05"
        )


class TestEndToEndParametricVdP:
    """Parametric derivative-matching fit -> batched ODE integration."""

    def test_trajectory_quality_at_held_out_mu(self, bkd):
        n_dynamic = 2
        max_level = 4

        expansion = _build_expansion(bkd, nvars=3, nqoi=2, max_level=max_level)

        train_mu_values = [0.5, 1.0, 1.5, 2.0]
        n_per_mu = 300
        rng = np.random.RandomState(42)
        all_states = []
        all_derivs = []
        for mu in train_mu_values:
            states_2d = bkd.array(rng.uniform(-3, 3, (2, n_per_mu)))
            mu_row = bkd.full((1, n_per_mu), mu)
            augmented = bkd.vstack([states_2d, mu_row])
            derivs = _true_vdp_rhs(states_2d, mu=mu, bkd=bkd)
            all_states.append(augmented)
            all_derivs.append(derivs)
        train_states = bkd.hstack(all_states)
        train_derivs = bkd.hstack(all_derivs)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion, train_states, train_derivs)
        fitted = result.surrogate()

        test_mu_values = [0.7, 1.3, 1.8]
        k = len(test_mu_values)
        test_mu_batch = bkd.array([test_mu_values])
        wrapper = BatchedBoundODEResidual(
            fitted, n_dynamic=n_dynamic, mu_batch=test_mu_batch
        )

        init_flat = bkd.array([1.0, 0.0] * k)
        final_time = 2.0
        deltat = 0.005
        approx_sols, times = _forward_solve(
            wrapper, init_flat, 0.0, final_time, deltat, bkd
        )

        for i, mu_val in enumerate(test_mu_values):
            ref_sol = _solve_true_vdp(mu_val, [1.0, 0.0], times, bkd)
            approx_traj = approx_sols[i * 2 : (i + 1) * 2, :]
            max_err = float(bkd.max(bkd.abs(approx_traj - ref_sol)))
            amplitude = float(bkd.max(bkd.abs(ref_sol)))
            assert max_err / amplitude < 0.05, (
                f"mu={mu_val}: relative error {max_err / amplitude:.4f} "
                f"exceeds 0.05"
            )


def _build_trajectory_matching_loss(wrapper, init_states, init_time, final_time,
                                     deltat, bkd):
    """Build TrajectoryMatchingLoss observing all states at all time steps."""
    stepper = BackwardEulerAdjoint(wrapper)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=30, atol=1e-10, rtol=1e-10)
    integrator = TimeIntegrator(
        init_time=init_time,
        final_time=final_time,
        deltat=deltat,
        newton_solver=newton,
    )

    fwd_sols, times = integrator.solve(init_states)
    nstates = fwd_sols.shape[0]
    ntimes = fwd_sols.shape[1]
    obs_time_indices = bkd.arange(ntimes)
    obs_tuples = [(i, obs_time_indices) for i in range(nstates)]

    flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))

    # Perturb observations so gradient is nonzero at current params
    rng = np.random.RandomState(123)
    perturbation = bkd.array(rng.randn(nstates * ntimes) * 0.01)
    flat_obs = flat_obs + perturbation

    return TrajectoryMatchingLoss(
        wrapper=wrapper,
        integrator=integrator,
        init_states=init_states,
        obs_tuples=obs_tuples,
        observations=flat_obs,
        noise_std=1.0,
    )


class TestAdjointGradient:
    """Validate adjoint gradient via DerivativeChecker."""

    @pytest.mark.slow_on("TorchBkd")
    def test_adjoint_gradient_k1(self, bkd):
        """Single-trajectory adjoint gradient matches FD via DerivativeChecker."""
        expansion = _build_expansion(bkd, nvars=3, nqoi=2, max_level=3)

        n_train = 300
        rng = np.random.RandomState(42)
        train_states = bkd.array(rng.uniform(-2, 2, (2, n_train)))
        mu_row = bkd.full((1, n_train), 1.0)
        aug_states = bkd.vstack([train_states, mu_row])
        train_derivs = _true_vdp_rhs(train_states, mu=1.0, bkd=bkd)

        fitter = LeastSquaresFitter(bkd)
        fitted = fitter.fit(expansion, aug_states, train_derivs).surrogate()

        n_dynamic = 2
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_trajectory_matching_loss(
            wrapper, init_state, 0.0, 0.3, 0.01, bkd
        )

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    @pytest.mark.slow_on("TorchBkd")
    def test_adjoint_gradient_k4(self, bkd):
        """Batched (k=4) adjoint gradient matches FD via DerivativeChecker."""
        expansion = _build_expansion(bkd, nvars=3, nqoi=2, max_level=3)

        train_mu_values = [0.5, 1.0, 1.5, 2.0]
        n_per_mu = 200
        rng = np.random.RandomState(42)
        all_states, all_derivs = [], []
        for mu in train_mu_values:
            s2d = bkd.array(rng.uniform(-2, 2, (2, n_per_mu)))
            mu_row = bkd.full((1, n_per_mu), mu)
            all_states.append(bkd.vstack([s2d, mu_row]))
            all_derivs.append(_true_vdp_rhs(s2d, mu=mu, bkd=bkd))
        train_states = bkd.hstack(all_states)
        train_derivs = bkd.hstack(all_derivs)

        fitter = LeastSquaresFitter(bkd)
        fitted = fitter.fit(expansion, train_states, train_derivs).surrogate()

        n_dynamic, k = 2, 4
        mu_batch = bkd.array([[0.5, 1.0, 1.5, 2.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic, mu_batch=mu_batch)

        init_flat = bkd.array([0.5, 0.3] * k)
        loss = _build_trajectory_matching_loss(
            wrapper, init_flat, 0.0, 0.2, 0.01, bkd
        )

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"
