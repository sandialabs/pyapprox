"""Tests for SUPN core class.

Tests cover creation, evaluation, parameter management, and shape conventions.
"""

import numpy as np
import pytest

from pyapprox.surrogates.supn import SUPN, StandardChebyshev1D, create_supn


class TestStandardChebyshev1D:
    """Tests for StandardChebyshev1D basis."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_evaluation_shape(self, bkd) -> None:
        """Test output shape (nsamples, nterms)."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(5)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        result = basis(samples)
        assert result.shape == (10, 5)

    def test_T0_is_one(self, bkd) -> None:
        """T_0(x) = 1 for all x."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 7)))
        result = basis(samples)
        bkd.assert_allclose(result[:, 0], bkd.ones((7,)))

    def test_T1_is_x(self, bkd) -> None:
        """T_1(x) = x."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(3)
        x_vals = np.random.uniform(-1, 1, (1, 7))
        samples = bkd.asarray(x_vals)
        result = basis(samples)
        bkd.assert_allclose(result[:, 1], samples[0])

    def test_known_values(self, bkd) -> None:
        """Verify T_n(x) against cos(n*arccos(x))."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(6)
        x_vals = np.array([[-0.5, 0.0, 0.3, 0.9]])
        samples = bkd.asarray(x_vals)
        result = basis(samples)

        for n in range(6):
            expected = bkd.asarray(np.cos(n * np.arccos(x_vals[0])))
            bkd.assert_allclose(result[:, n], expected, atol=1e-14)

    def test_jacobian_shape(self, bkd) -> None:
        """Test jacobian_batch output shape (nsamples, nterms)."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(5)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        result = basis.jacobian_batch(samples)
        assert result.shape == (10, 5)

    def test_jacobian_T0_is_zero(self, bkd) -> None:
        """dT_0/dx = 0."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 7)))
        result = basis.jacobian_batch(samples)
        bkd.assert_allclose(result[:, 0], bkd.zeros((7,)))

    def test_jacobian_T1_is_one(self, bkd) -> None:
        """dT_1/dx = 1."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 7)))
        result = basis.jacobian_batch(samples)
        bkd.assert_allclose(result[:, 1], bkd.ones((7,)))

    def test_jacobian_finite_differences(self, bkd) -> None:
        """Verify dT_n/dx against finite differences."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(6)
        x_vals = np.array([[-0.5, 0.3, 0.8]])
        samples = bkd.asarray(x_vals)
        jac = basis.jacobian_batch(samples)

        eps = 1e-7
        samples_p = bkd.asarray(x_vals + eps)
        samples_m = bkd.asarray(x_vals - eps)
        fd_jac = (basis(samples_p) - basis(samples_m)) / (2 * eps)
        bkd.assert_allclose(jac, fd_jac, rtol=1e-6)

    def test_zero_terms(self, bkd) -> None:
        """nterms=0 returns empty array."""
        basis = StandardChebyshev1D(bkd)
        basis.set_nterms(0)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))
        result = basis(samples)
        assert result.shape == (5, 0)


class TestSUPN:
    """Tests for SUPN core class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_create_supn_basic(self, bkd) -> None:
        """Test create_supn factory function."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        assert supn.nvars() == 2
        assert supn.width() == 3
        assert supn.nqoi() == 1
        assert supn.nterms() > 0

    def test_create_supn_multi_qoi(self, bkd) -> None:
        """Test create_supn with nqoi > 1."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=2)
        assert supn.nqoi() == 2

    def test_nparams(self, bkd) -> None:
        """Test nparams = nqoi*width + width*nterms."""
        supn = create_supn(nvars=2, width=4, max_level=3, bkd=bkd, nqoi=2)
        N = supn.width()
        M = supn.nterms()
        Q = supn.nqoi()
        assert supn.nparams() == Q * N + N * M

    def test_call_shape(self, bkd) -> None:
        """Test __call__ output shape (nqoi, nsamples)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        result = supn(samples)
        assert result.shape == (1, 10)

    def test_call_shape_multi_qoi(self, bkd) -> None:
        """Test __call__ shape for nqoi > 1."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        result = supn(samples)
        assert result.shape == (3, 10)

    def test_call_finite(self, bkd) -> None:
        """Test that output contains no NaN/Inf."""
        supn = create_supn(nvars=3, width=5, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 20)))
        result = supn(samples)
        assert bkd.all_bool(bkd.isfinite(result))

    def test_eval_from_basis_matrix(self, bkd) -> None:
        """Test eval_from_basis_matrix matches __call__."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        expected = supn(samples)
        B = supn.basis()(samples)
        result = supn.eval_from_basis_matrix(B)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_with_params_roundtrip(self, bkd) -> None:
        """Test flatten/unflatten is exact inverse."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = supn._flatten_params()
        supn2 = supn.with_params(params)

        bkd.assert_allclose(supn2.outer_coefs(), supn.outer_coefs())
        bkd.assert_allclose(supn2.inner_coefs(), supn.inner_coefs())

    def test_with_params_changes_output(self, bkd) -> None:
        """Test that with_params produces different output."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 5)))
        out1 = supn(samples)

        new_params = bkd.asarray(np.random.randn(supn.nparams()))
        supn2 = supn.with_params(new_params)
        out2 = supn2(samples)

        # Outputs should differ
        assert not bkd.allclose(out1, out2, rtol=0.0, atol=1e-10)

    def test_with_params_wrong_size(self, bkd) -> None:
        """Test ValueError for wrong parameter count."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        with pytest.raises(ValueError, match="Expected"):
            supn.with_params(bkd.asarray(np.zeros(3)))

    def test_custom_indices(self, bkd) -> None:
        """Test create_supn with custom indices."""
        indices = bkd.asarray(np.array([[0, 1, 0], [0, 0, 1]]))
        supn = create_supn(
            nvars=2, width=2, max_level=0, bkd=bkd, indices=indices
        )
        assert supn.nterms() == 3

    def test_width_validation(self, bkd) -> None:
        """Test that width < 1 raises ValueError."""
        from pyapprox.surrogates.affine.basis.multiindex import MultiIndexBasis
        from pyapprox.surrogates.affine.indices.utils import (
            compute_hyperbolic_indices,
        )

        bases_1d = [StandardChebyshev1D(bkd)]
        indices = compute_hyperbolic_indices(1, 2, 1.0, bkd)
        basis = MultiIndexBasis(bases_1d, bkd, indices)
        with pytest.raises(ValueError, match="width must be >= 1"):
            SUPN(basis, width=0, bkd=bkd)

    def test_1d_supn(self, bkd) -> None:
        """Test 1D SUPN evaluation against hand computation."""
        supn = create_supn(nvars=1, width=1, max_level=1, bkd=bkd, nqoi=1)
        # max_level=1 gives 2 terms: T_0=1, T_1=x
        assert supn.nterms() == 2

        # Set known params: c=2.0, a=[0.5, 1.0]
        params = bkd.asarray(np.array([2.0, 0.5, 1.0]))
        supn = supn.with_params(params)

        x_val = 0.3
        samples = bkd.asarray(np.array([[x_val]]))
        result = supn(samples)

        # f(x) = 2 * tanh(0.5*1 + 1.0*x) = 2*tanh(0.5 + 0.3)
        expected = 2.0 * np.tanh(0.5 + 1.0 * x_val)
        bkd.assert_allclose(result, bkd.asarray([[expected]]), rtol=1e-12)
