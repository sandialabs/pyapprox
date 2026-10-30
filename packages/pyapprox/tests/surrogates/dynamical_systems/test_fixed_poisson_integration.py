"""Integration tests for FixedPoissonVariableHamiltonianSurrogate.

Diagnostic staircase that localizes failures:
1. Closed-form fitter recovers SHO coefficients
2. Forward integration matches analytic SHO
3. Adjoint gradient w.r.t. H coefs (FD-verified)
4. Energy conservation drift bounded by truncation error
"""

from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.tableau import crank_nicolson
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.fitters.fixed_poisson_variable_hamiltonian_fitter import (  # noqa: E501
    FixedPoissonVariableHamiltonianDerivativeMatchingFitter,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.surrogates.fixed_poisson_variable_hamiltonian import (  # noqa: E501
    FixedPoissonVariableHamiltonianSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.surrogates.dynamical_systems import solve_batched, step_times


def _make_sho_basis(
    bkd: Backend[Array], nvars: int, max_level: int
) -> BasisExpansion[Array]:
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    all_indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    indices = all_indices[:, 1:]
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=1)


def _sho_analytic(
    q0: float, p0: float, omega: float, times_np: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Analytic SHO solution."""
    q = q0 * np.cos(omega * times_np) + (p0 / omega) * np.sin(omega * times_np)
    p = -q0 * omega * np.sin(omega * times_np) + p0 * np.cos(omega * times_np)
    return np.stack([q, p], axis=0)


def _method(bkd: Backend[Array]) -> SteppingMethodProtocol[Array, Array]:
    """Crank-Nicolson with Newton at atol and rtol 1e-12."""
    newton: Newton[Array, Array] = Newton(maxiters=30, atol=1e-12, rtol=1e-12)
    return RungeKutta(crank_nicolson(), newton)


def _forward_solve_cn(
    wrapper: BatchedBoundODEResidual[Array],
    init_state: Array,
    init_time: float,
    final_time: float,
    deltat: float,
) -> Tuple[Array, Array]:
    """The Crank-Nicolson trajectory, shape (nstates, ntimes), and its
    times."""
    times = step_times(init_time, final_time, deltat)
    return solve_batched(wrapper, _method(wrapper.bkd()), init_state, times)


def _fit_sho(
    bkd: Backend[Array], max_level: int = 4
) -> FixedPoissonVariableHamiltonianSurrogate[Array]:
    """Fit SHO with omega=1 via closed-form derivative matching."""
    expansion = _make_sho_basis(bkd, nvars=2, max_level=max_level)
    surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(expansion)

    rng = np.random.RandomState(42)
    nsamples = 500
    q = rng.uniform(-2.0, 2.0, nsamples)
    p = rng.uniform(-2.0, 2.0, nsamples)
    states = bkd.array(np.stack([q, p], axis=0))
    derivs = bkd.array(np.stack([p, -q], axis=0))

    dataset = SnapshotDataset(states, derivs, bkd)
    fitter = FixedPoissonVariableHamiltonianDerivativeMatchingFitter(bkd)
    result = fitter.fit(surrogate, dataset)
    return result.surrogate()


def _build_trajectory_loss(
    wrapper: BatchedBoundODEResidual[Array],
    init_states: Array,
    init_time: float,
    final_time: float,
    deltat: float,
    bkd: Backend[Array],
) -> TrajectoryMatchingLoss[Array]:
    fwd_sols, _ = _forward_solve_cn(
        wrapper, init_states, init_time, final_time, deltat
    )
    nstates, ntimes = fwd_sols.shape
    obs_time_indices = bkd.arange(ntimes)
    obs_tuples = [(i, obs_time_indices) for i in range(nstates)]
    flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))

    rng = np.random.RandomState(123)
    perturbation = bkd.array(rng.randn(nstates * ntimes) * 0.01)
    flat_obs = flat_obs + perturbation

    return TrajectoryMatchingLoss(
        wrapper=wrapper,
        method=_method(bkd),
        times=step_times(init_time, final_time, deltat),
        init_states=init_states,
        obs_tuples=obs_tuples,
        observations=flat_obs,
        noise_std=1.0,
    )


class TestFixedPoissonIntegration:
    def test_fit_recovers_sho_coefficients_closed_form(
        self, bkd: Backend[Array]
    ) -> None:
        """Pre-gate #1: fitter recovers H coefs to machine precision."""
        fitted = _fit_sho(bkd)
        rng = np.random.RandomState(99)
        ntest = 100
        q = rng.uniform(-2.0, 2.0, ntest)
        p = rng.uniform(-2.0, 2.0, ntest)
        test_states = bkd.array(np.stack([q, p], axis=0))
        expected = bkd.array(np.stack([p, -q], axis=0))
        pred = fitted(test_states)
        bkd.assert_allclose(pred, expected, atol=1e-10)

    def test_forward_integration_matches_analytic_sho(
        self, bkd: Backend[Array]
    ) -> None:
        """Pre-gate #2: fitted surrogate integrates to match analytic SHO."""
        fitted = _fit_sho(bkd)
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2)

        q0, p0, omega = 1.0, 0.0, 1.0
        init_state = bkd.array([q0, p0])
        approx_sol, times = _forward_solve_cn(
            wrapper, init_state, 0.0, 2 * np.pi, 0.01
        )

        times_np: NDArray[np.float64] = np.asarray(bkd.to_numpy(times))
        ref_sol = bkd.array(_sho_analytic(q0, p0, omega, times_np))
        max_err = bkd.to_float(bkd.max(bkd.abs(approx_sol - ref_sol)))
        assert max_err < 1e-4, f"max error {max_err:.2e} exceeds 1e-4"

    @pytest.mark.slow_on("TorchBkd")
    def test_adjoint_gradient_sho_eta_only(self, bkd: Backend[Array]) -> None:
        """Pre-gate #3: FD-verify adjoint gradient w.r.t. H coefs only."""
        fitted = _fit_sho(bkd, max_level=3)
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2)

        init_state = bkd.array([1.0, 0.0])
        loss = _build_trajectory_loss(wrapper, init_state, 0.0, 0.5, 0.02, bkd)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = bkd.to_float(checker.error_ratio(errors[0]))
        assert ratio <= 1e-6, f"error_ratio={ratio:.2e}"

    def test_energy_conservation_drift(self, bkd: Backend[Array]) -> None:
        """Energy drift along trajectory bounded by CN truncation error."""
        fitted = _fit_sho(bkd)
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2)

        q0, p0 = 1.0, 0.0
        init_state = bkd.array([q0, p0])
        approx_sol, _ = _forward_solve_cn(wrapper, init_state, 0.0, 2 * np.pi, 0.01)

        q_traj = approx_sol[0, :]
        p_traj = approx_sol[1, :]
        H_traj = 0.5 * (q_traj**2 + p_traj**2)
        H0 = 0.5 * (q0**2 + p0**2)
        drift = bkd.to_float(bkd.max(bkd.abs(H_traj - H0)))
        assert drift < 1e-4, f"energy drift {drift:.2e} exceeds 1e-4"
