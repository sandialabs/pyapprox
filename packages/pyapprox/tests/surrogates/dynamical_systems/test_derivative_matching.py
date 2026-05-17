"""Tests for DerivativeMatchingLoss with BasisExpansion."""

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
from pyapprox.surrogates.dynamical_systems.losses.derivative_matching import (
    DerivativeMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)


def _make_expansion(bkd, nvars, nqoi, max_level=3):
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)
    rng = np.random.RandomState(99)
    coef = bkd.array(rng.randn(expansion.nterms(), nqoi) * 0.1)
    expansion.set_coefficients(coef)
    return expansion


def _vdp_rhs(states, mu, bkd):
    x1 = states[0:1, :]
    x2 = states[1:2, :]
    return bkd.vstack([x2, mu * (1 - x1**2) * x2 - x1])


class TestDerivativeMatchingLoss:
    def test_protocol_conformance(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        assert isinstance(expansion, LearnedFunctionProtocol)

    def test_loss_at_true_params_is_small(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2, max_level=5)

        n_train = 500
        rng = np.random.RandomState(42)
        states = bkd.array(rng.uniform(-2, 2, (2, n_train)))
        derivs = _vdp_rhs(states, mu=1.0, bkd=bkd)

        from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
            LeastSquaresFitter,
        )
        fitter = LeastSquaresFitter(bkd)
        fitted = fitter.fit(expansion, states, derivs).surrogate()

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(fitted, dataset)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))
        val = loss(sample)
        assert float(val[0, 0]) < 1e-4

    def test_loss_increases_at_perturbed_params(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2, max_level=4)

        n_train = 300
        rng = np.random.RandomState(42)
        states = bkd.array(rng.uniform(-2, 2, (2, n_train)))
        derivs = _vdp_rhs(states, mu=1.0, bkd=bkd)

        from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
            LeastSquaresFitter,
        )
        fitted = LeastSquaresFitter(bkd).fit(expansion, states, derivs).surrogate()

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(fitted, dataset)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))
        val_at_opt = float(loss(sample)[0, 0])

        perturbed = eta + bkd.array(np.random.RandomState(7).randn(eta.shape[0]) * 0.1)
        sample_perturbed = bkd.reshape(perturbed, (perturbed.shape[0], 1))
        val_perturbed = float(loss(sample_perturbed)[0, 0])

        assert val_perturbed > val_at_opt

    def test_jacobian_via_derivative_checker(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2, max_level=3)

        n_train = 200
        rng = np.random.RandomState(42)
        states = bkd.array(rng.uniform(-2, 2, (2, n_train)))
        derivs = _vdp_rhs(states, mu=1.0, bkd=bkd)

        from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
            LeastSquaresFitter,
        )
        fitted = LeastSquaresFitter(bkd).fit(expansion, states, derivs).surrogate()

        dataset = SnapshotDataset(states, derivs, bkd)
        loss = DerivativeMatchingLoss(fitted, dataset)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    def test_parametric_jacobian_via_derivative_checker(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2, max_level=3)

        rng = np.random.RandomState(42)
        n_train = 300
        states_2d = bkd.array(rng.uniform(-2, 2, (2, n_train)))
        mu_row = bkd.full((1, n_train), 1.0)
        aug_states = bkd.vstack([states_2d, mu_row])
        derivs = _vdp_rhs(states_2d, mu=1.0, bkd=bkd)

        from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
            LeastSquaresFitter,
        )
        fitted = LeastSquaresFitter(bkd).fit(expansion, aug_states, derivs).surrogate()

        dataset = SnapshotDataset(aug_states, derivs, bkd)
        loss = DerivativeMatchingLoss(fitted, dataset)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    def test_invalid_nvars_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        states = bkd.array(np.zeros((2, 10)))
        derivs = bkd.array(np.zeros((2, 10)))
        dataset = SnapshotDataset(states, derivs, bkd)
        with pytest.raises(ValueError, match="nstates_input"):
            DerivativeMatchingLoss(expansion, dataset)

    def test_invalid_nqoi_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=3)
        states = bkd.array(np.zeros((2, 10)))
        derivs = bkd.array(np.zeros((2, 10)))
        dataset = SnapshotDataset(states, derivs, bkd)
        with pytest.raises(ValueError, match="nstates_output"):
            DerivativeMatchingLoss(expansion, dataset)
