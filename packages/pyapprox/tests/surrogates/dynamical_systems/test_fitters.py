"""Tests for LinearInParamsFitter."""

import numpy as np

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.encoders import LinearEncoder
from pyapprox.surrogates.dynamical_systems.fitters.linear_fitter import (
    LinearInParamsFitter,
)
from pyapprox.surrogates.dynamical_systems.vector_fields import (
    BasisExpansionVectorField,
)


def _make_vf(bkd, nvars, max_level):
    marginals = [UniformMarginal(-2.0, 2.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=nvars)
    return BasisExpansionVectorField(exp)


class TestLinearFitterVanDerPol:
    def test_dense_coefficient_recovery(self, bkd, van_der_pol_data):
        states, derivatives, _ = van_der_pol_data
        vf = _make_vf(bkd, nvars=2, max_level=3)
        dataset = SnapshotDataset(states, derivatives, bkd)
        fitter = LinearInParamsFitter(bkd)
        result = fitter.fit(vf, dataset)
        fitted_vf = result.surrogate()
        bkd.assert_allclose(
            fitted_vf(states), derivatives, rtol=1e-8,
        )

    def test_immutable_pattern(self, bkd, van_der_pol_data):
        states, derivatives, _ = van_der_pol_data
        vf = _make_vf(bkd, nvars=2, max_level=3)
        original_coef = bkd.copy(
            vf.expansion().get_coefficients()
        )
        dataset = SnapshotDataset(states, derivatives, bkd)
        fitter = LinearInParamsFitter(bkd)
        result = fitter.fit(vf, dataset)
        bkd.assert_allclose(
            vf.expansion().get_coefficients(), original_coef,
        )
        assert result.surrogate() is not vf

    def test_sparse_omp_selects_correct_terms(self, numpy_bkd):
        bkd = numpy_bkd
        np.random.seed(42)
        nsamples = 200
        x1 = np.random.uniform(-2.0, 2.0, nsamples)
        x2 = np.random.uniform(-2.0, 2.0, nsamples)
        mu = 1.0
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        states = bkd.array(np.stack([x1, x2], axis=0))
        derivatives = bkd.array(np.stack([dx1, dx2], axis=0))

        vf = _make_vf(bkd, nvars=2, max_level=3)
        dataset = SnapshotDataset(states, derivatives, bkd)

        from pyapprox.surrogates.affine.expansions.fitters.omp import (
            OMPFitter,
        )

        # Multi-QoI OMP via OMPFitter (delegates to OMPSolver internally)
        omp = OMPFitter(bkd, max_nonzeros=6, rtol=1e-10)
        result = omp.fit(vf.expansion(), states, derivatives)
        fitted_vf = vf.with_params(result.params())
        bkd.assert_allclose(
            fitted_vf(states), derivatives, rtol=1e-6,
        )


class TestLinearFitterParametricVanDerPol:
    def test_parametric_dense(self, numpy_bkd):
        bkd = numpy_bkd
        np.random.seed(42)
        gammas = [0.5, 1.0, 1.5, 2.0]
        nsamples_per = 100
        all_states = []
        all_derivs = []
        for gamma in gammas:
            x1 = np.random.uniform(-2.0, 2.0, nsamples_per)
            x2 = np.random.uniform(-2.0, 2.0, nsamples_per)
            g = np.full(nsamples_per, gamma)
            dx1 = x2
            dx2 = gamma * (1 - x1**2) * x2 - x1
            dg = np.zeros(nsamples_per)
            all_states.append(np.stack([x1, x2, g], axis=0))
            all_derivs.append(np.stack([dx1, dx2, dg], axis=0))

        states = bkd.array(np.concatenate(all_states, axis=1))
        derivatives = bkd.array(np.concatenate(all_derivs, axis=1))

        # max_level=4 needed: gamma*x1^2*x2 has multi-index (2,1,1), total degree 4
        vf = _make_vf(bkd, nvars=3, max_level=4)
        dataset = SnapshotDataset(states, derivatives, bkd)
        fitter = LinearInParamsFitter(bkd)
        result = fitter.fit(vf, dataset)
        fitted_vf = result.surrogate()
        bkd.assert_allclose(
            fitted_vf(states), derivatives, rtol=1e-6,
        )

        # Third row (gamma derivative) should be near zero
        gamma_derivs = fitted_vf(states)[2:3, :]
        bkd.assert_allclose(
            gamma_derivs,
            bkd.zeros(gamma_derivs.shape),
            atol=1e-10,
        )


class TestLinearFitterLotkaVolterra:
    def test_dense(self, bkd, lotka_volterra_data):
        states, derivatives, _ = lotka_volterra_data
        vf = _make_vf(bkd, nvars=2, max_level=3)
        dataset = SnapshotDataset(states, derivatives, bkd)
        fitter = LinearInParamsFitter(bkd)
        result = fitter.fit(vf, dataset)
        fitted_vf = result.surrogate()
        bkd.assert_allclose(
            fitted_vf(states), derivatives, rtol=1e-6,
        )


class TestLinearFitterWithEncoder:
    def test_rotated_van_der_pol(self, numpy_bkd):
        bkd = numpy_bkd
        np.random.seed(42)
        nsamples = 200
        x1 = np.random.uniform(-2.0, 2.0, nsamples)
        x2 = np.random.uniform(-2.0, 2.0, nsamples)
        mu = 1.0
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        states_orig = bkd.array(np.stack([x1, x2], axis=0))
        derivs_orig = bkd.array(np.stack([dx1, dx2], axis=0))

        # Apply a 2x2 orthogonal rotation
        theta = np.pi / 6
        Q = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        P = bkd.array(Q)
        encoder = LinearEncoder(P, bkd)

        # Project dataset to latent space
        dataset_orig = SnapshotDataset(states_orig, derivs_orig, bkd)
        dataset_latent = dataset_orig.project(encoder)

        # Fit in latent space
        vf_latent = _make_vf(bkd, nvars=2, max_level=3)
        fitter = LinearInParamsFitter(bkd)
        result = fitter.fit(vf_latent, dataset_latent)
        fitted_vf = result.surrogate()

        # Verify fit quality in latent space
        bkd.assert_allclose(
            fitted_vf(dataset_latent.states()),
            dataset_latent.derivatives(),
            rtol=1e-6,
        )
