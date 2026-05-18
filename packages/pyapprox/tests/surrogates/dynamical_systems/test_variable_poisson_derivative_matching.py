"""Tests for VariablePoissonFixedHamiltonianSurrogate derivative matching."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.fitters.variable_poisson_fixed_hamiltonian_fitter import (  # noqa: E501
    VariablePoissonFixedHamiltonianDerivativeMatchingFitter,
)
from pyapprox.surrogates.dynamical_systems.losses.derivative_matching import (
    DerivativeMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)


def _identity_grad_H(samples):
    """grad H = x (H = 0.5*||x||^2), state rows only."""
    return samples


def _identity_hessian_H(samples):
    """Hessian of H = 0.5*||x||^2 is identity for each sample."""
    n_dynamic = samples.shape[0]
    nsamples = samples.shape[1]
    eye = np.eye(n_dynamic)
    return np.tile(eye[None, :, :], (nsamples, 1, 1))


def _identity_grad_H_bkd(bkd, n_dynamic):
    """Return backend-compatible grad_H and grad_H_jac for H=0.5*||x||^2."""
    def grad_H(samples):
        return samples[:n_dynamic, :]

    def grad_H_jac(samples):
        nsamples = samples.shape[1]
        eye = bkd.eye(n_dynamic)
        return bkd.tile(bkd.reshape(eye, (1, n_dynamic, n_dynamic)),
                        (nsamples, 1, 1))

    return grad_H, grad_H_jac


class TestVariablePoissonDerivativeMatching:
    def test_fitter_recovers_skew(self, bkd):
        """Fitter recovers 6 skew entries for 4D system with H=0.5*||x||^2."""
        n_dynamic = 4
        rng = np.random.RandomState(42)

        true_eta = np.array([0.5, -1.0, 0.3, 0.7, -0.2, 1.5])

        grad_H, grad_H_jac = _identity_grad_H_bkd(bkd, n_dynamic)
        surrogate = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=grad_H,
            grad_hamiltonian_jacobian=grad_H_jac,
            n_dynamic=n_dynamic,
            bkd=bkd,
        )
        true_surrogate = surrogate.with_params(bkd.array(true_eta))

        nsamples = 300
        states = bkd.array(rng.uniform(-2, 2, (n_dynamic, nsamples)))
        derivs = true_surrogate(states)

        dataset = SnapshotDataset(states, derivs, bkd)
        fitter = VariablePoissonFixedHamiltonianDerivativeMatchingFitter(bkd)
        result = fitter.fit(surrogate, dataset)
        fitted = result.surrogate()

        test_states = bkd.array(rng.uniform(-2, 2, (n_dynamic, 100)))
        pred = fitted(test_states)
        expected = true_surrogate(test_states)
        bkd.assert_allclose(pred, expected, atol=1e-10)

    def test_jacobian_via_derivative_checker(self, bkd):
        """FD-verify DerivativeMatchingLoss jacobian for variable-Poisson.

        Targets are generated at true_eta; FD check runs at init_eta
        (offset from true) so the loss has nonzero gradient.
        """
        n_dynamic = 4
        rng = np.random.RandomState(42)

        true_eta = np.array([0.5, -1.0, 0.3, 0.7, -0.2, 1.5])
        init_eta = true_eta + rng.randn(6) * 0.1

        grad_H, grad_H_jac = _identity_grad_H_bkd(bkd, n_dynamic)
        true_surrogate = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=grad_H,
            grad_hamiltonian_jacobian=grad_H_jac,
            n_dynamic=n_dynamic,
            bkd=bkd,
        )
        true_surrogate = true_surrogate.with_params(bkd.array(true_eta))

        nsamples = 200
        states = bkd.array(rng.uniform(-2, 2, (n_dynamic, nsamples)))
        derivs = true_surrogate(states)

        init_surrogate = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=grad_H,
            grad_hamiltonian_jacobian=grad_H_jac,
            n_dynamic=n_dynamic,
            bkd=bkd,
        )
        init_surrogate = init_surrogate.with_params(bkd.array(init_eta))

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(init_surrogate, dataset)

        eta = init_surrogate.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    def test_fitter_with_parametric_hamiltonian(self, bkd):
        """Fitter recovers skew entries with n_aux=1 (parametric H).

        System: 4D state + 1 auxiliary param 'a'.
        H(x, a) = 0.5*(x0^2 + x1^2 + x2^2 + x3^2 + a*x0*x2)
        grad_H = [x0 + 0.5*a*x2, x1, x2 + 0.5*a*x0, x3]
        """
        n_dynamic = 4
        n_aux = 1
        rng = np.random.RandomState(42)

        true_eta = np.array([0.5, -1.0, 0.3, 0.7, -0.2, 1.5])

        def grad_H(aug_samples):
            x = aug_samples[:4, :]
            a = aug_samples[4:5, :]
            g0 = x[0:1, :] + 0.5 * a * x[2:3, :]
            g1 = x[1:2, :]
            g2 = x[2:3, :] + 0.5 * a * x[0:1, :]
            g3 = x[3:4, :]
            return bkd.vstack([g0, g1, g2, g3])

        def grad_H_jac(aug_samples):
            nsamples = aug_samples.shape[1]
            a = aug_samples[4, :]
            H_xx = bkd.tile(
                bkd.reshape(bkd.eye(4), (1, 4, 4)), (nsamples, 1, 1)
            )
            H_xx = bkd.index_update(H_xx, (slice(None), 0, 2), 0.5 * a)
            H_xx = bkd.index_update(H_xx, (slice(None), 2, 0), 0.5 * a)
            return H_xx

        surrogate = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=grad_H,
            grad_hamiltonian_jacobian=grad_H_jac,
            n_dynamic=n_dynamic,
            bkd=bkd,
            n_aux=n_aux,
        )
        true_surrogate = surrogate.with_params(bkd.array(true_eta))

        nsamples = 400
        states_4d = rng.uniform(-2, 2, (4, nsamples))
        a_vals = rng.uniform(0.1, 2.0, (1, nsamples))
        aug_states = bkd.array(np.vstack([states_4d, a_vals]))
        derivs = true_surrogate(aug_states)

        dataset = SnapshotDataset(aug_states, derivs, bkd)
        fitter = VariablePoissonFixedHamiltonianDerivativeMatchingFitter(bkd)
        result = fitter.fit(surrogate, dataset)
        fitted = result.surrogate()

        ntest = 100
        test_states = rng.uniform(-2, 2, (4, ntest))
        test_a = rng.uniform(0.1, 2.0, (1, ntest))
        test_aug = bkd.array(np.vstack([test_states, test_a]))
        pred = fitted(test_aug)
        expected = true_surrogate(test_aug)
        bkd.assert_allclose(pred, expected, atol=1e-10)

    def test_build_L_autograd(self, torch_bkd):
        """Verify gradient flows through _build_L via torch autograd."""
        import torch

        bkd = torch_bkd
        n_dynamic = 3

        grad_H, grad_H_jac = _identity_grad_H_bkd(bkd, n_dynamic)
        surrogate = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=grad_H,
            grad_hamiltonian_jacobian=grad_H_jac,
            n_dynamic=n_dynamic,
            bkd=bkd,
        )

        eta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64,
                           requires_grad=True)
        surrogate.hyp_list().set_active_values(eta)

        L = surrogate._build_L()
        loss = (L**2).sum()
        loss.backward()

        assert eta.grad is not None, "No gradient on eta"
        expected_grad = torch.tensor([2.0 * 1.0 * 2, 2.0 * 2.0 * 2,
                                      2.0 * 3.0 * 2],
                                     dtype=torch.float64)
        bkd.assert_allclose(eta.grad, expected_grad, atol=1e-12)
