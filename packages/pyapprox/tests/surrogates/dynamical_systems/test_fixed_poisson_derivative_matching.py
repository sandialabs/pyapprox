"""Tests for FixedPoissonVariableHamiltonianSurrogate derivative matching."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.fitters.fixed_poisson_variable_hamiltonian_fitter import (  # noqa: E501
    FixedPoissonVariableHamiltonianDerivativeMatchingFitter,
)
from pyapprox.surrogates.dynamical_systems.losses.derivative_matching import (
    DerivativeMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.surrogates.fixed_poisson_variable_hamiltonian import (  # noqa: E501
    FixedPoissonVariableHamiltonianSurrogate,
)


def _make_sho_basis(bkd, nvars, max_level, n_params=0):
    """Build scalar polynomial basis excluding dynamically-inert terms.

    Any H term that depends only on non-dynamic variables (parameters)
    has zero gradient w.r.t. the dynamic variables (q, p), so it
    contributes nothing to the equations of motion dq/dt = dH/dp,
    dp/dt = -dH/dq. Including such terms makes the design matrix
    rank-deficient, which causes lstsq to behave inconsistently
    across LAPACK implementations (gelsy vs gelsd).
    """
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    all_indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    n_dynamic = nvars - n_params
    dynamic_degree = bkd.sum(all_indices[:n_dynamic, :], axis=0)
    keep = dynamic_degree > 0
    indices = all_indices[:, keep]
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=1)


def _sho_data(bkd, omega, nsamples, rng):
    """Generate SHO derivative matching data.

    H(q,p) = 0.5*(omega^2*q^2 + p^2)
    Canonical J: dq/dt = dH/dp = p, dp/dt = -dH/dq = -omega^2*q
    """
    q = rng.uniform(-2.0, 2.0, nsamples)
    p = rng.uniform(-2.0, 2.0, nsamples)
    states = bkd.array(np.stack([q, p], axis=0))
    dq = p
    dp = -(omega**2) * q
    derivs = bkd.array(np.stack([dq, dp], axis=0))
    return states, derivs


class TestFixedPoissonDerivativeMatching:
    def test_fitter_recovers_sho(self, bkd):
        """Closed-form fitter recovers SHO H coefs to machine precision."""
        omega = 1.0
        rng = np.random.RandomState(42)
        nsamples = 300
        states, derivs = _sho_data(bkd, omega, nsamples, rng)

        expansion = _make_sho_basis(bkd, nvars=2, max_level=3)
        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion
        )

        dataset = SnapshotDataset(states, derivs, bkd)
        fitter = FixedPoissonVariableHamiltonianDerivativeMatchingFitter(bkd)
        result = fitter.fit(surrogate, dataset)
        fitted = result.surrogate()

        test_rng = np.random.RandomState(99)
        test_states, test_derivs = _sho_data(bkd, omega, 100, test_rng)
        pred = fitted(test_states)
        bkd.assert_allclose(pred, test_derivs, atol=1e-10)

    def test_jacobian_via_derivative_checker(self, bkd):
        """FD-verify DerivativeMatchingLoss jacobian for fixed-Poisson."""
        rng = np.random.RandomState(42)
        expansion = _make_sho_basis(bkd, nvars=2, max_level=3)
        nterms = expansion.nterms()
        coef = bkd.array(rng.randn(nterms, 1) * 0.1)
        expansion.set_coefficients(coef)

        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion
        )

        nsamples = 200
        states = bkd.array(rng.uniform(-2, 2, (2, nsamples)))
        derivs = surrogate(states)
        perturbation = bkd.array(rng.randn(2, nsamples) * 0.01)
        derivs = derivs + perturbation

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(surrogate, dataset)

        eta = surrogate.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    def test_fitter_recovers_parametric_sho(self, bkd):
        """Fitter recovers H coefs for parametric SHO H(q,p,k)=0.5*(p^2+k*q^2)."""
        rng = np.random.RandomState(42)

        expansion = _make_sho_basis(bkd, nvars=3, max_level=3, n_params=1)
        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion, has_time_input=False, n_params=1
        )

        k_values = [0.5, 1.0, 2.0, 4.0]
        nsamples_per_k = 200
        all_states = []
        all_derivs = []
        for k_val in k_values:
            q = rng.uniform(-2.0, 2.0, nsamples_per_k)
            p = rng.uniform(-2.0, 2.0, nsamples_per_k)
            k_row = np.full(nsamples_per_k, k_val)
            states = bkd.array(np.stack([q, p, k_row], axis=0))
            dq = p
            dp = -k_val * q
            derivs = bkd.array(np.stack([dq, dp], axis=0))
            all_states.append(states)
            all_derivs.append(derivs)
        train_states = bkd.hstack(all_states)
        train_derivs = bkd.hstack(all_derivs)

        dataset = SnapshotDataset(train_states, train_derivs, bkd)
        fitter = FixedPoissonVariableHamiltonianDerivativeMatchingFitter(bkd)
        result = fitter.fit(surrogate, dataset)
        fitted = result.surrogate()

        test_k = 1.5
        ntest = 100
        q = rng.uniform(-2.0, 2.0, ntest)
        p = rng.uniform(-2.0, 2.0, ntest)
        test_states = bkd.array(
            np.stack([q, p, np.full(ntest, test_k)], axis=0)
        )
        test_derivs = bkd.array(np.stack([p, -test_k * q], axis=0))
        pred = fitted(test_states)
        bkd.assert_allclose(pred, test_derivs, atol=1e-10)

    def test_parametric_jacobian_via_derivative_checker(self, bkd):
        """FD-verify jacobian for parametric fixed-Poisson surrogate."""
        rng = np.random.RandomState(42)

        expansion = _make_sho_basis(bkd, nvars=3, max_level=3, n_params=1)
        nterms = expansion.nterms()
        coef = bkd.array(rng.randn(nterms, 1) * 0.1)
        expansion.set_coefficients(coef)

        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion, has_time_input=False, n_params=1
        )

        nsamples = 200
        q = rng.uniform(-2, 2, nsamples)
        p = rng.uniform(-2, 2, nsamples)
        k = rng.uniform(0.5, 2.0, nsamples)
        states = bkd.array(np.stack([q, p, k], axis=0))
        derivs = surrogate(states)
        perturbation = bkd.array(rng.randn(2, nsamples) * 0.01)
        derivs = derivs + perturbation

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(surrogate, dataset)

        eta = surrogate.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"
