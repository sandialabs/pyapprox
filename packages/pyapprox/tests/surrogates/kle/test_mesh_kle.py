"""Tests for MeshKLE."""

import numpy as np
import pytest
from pyapprox.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)
from pyapprox.surrogates.kernels.matern import (
    ExponentialKernel,
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kle.analytical import (
    AnalyticalExponentialKLE1D,
)
from pyapprox.surrogates.kle.mesh_kle import MeshKLE


def _gauss_legendre_quad(lb, ub, npts, bkd):
    """Gauss-Legendre quadrature on [lb, ub] for Lebesgue integration.

    Returns
    -------
    pts : Array, shape (1, npts)
        Quadrature points.
    weights : Array, shape (npts,)
        Quadrature weights (1D).
    """
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(npts)
    quad_rule = GaussQuadratureRule(poly)
    pts, wts = quad_rule(npts)
    # pts shape (1, npts), wts shape (npts, 1) - on [-1, 1]
    # The Legendre polynomial weights integrate against probability density
    # 1/2 on [-1,1] (sum to 1). For Lebesgue integration on [lb, ub],
    # multiply by (ub - lb) to get weights summing to (ub - lb).
    dom_len = ub - lb
    half_len = dom_len / 2.0
    mid = (lb + ub) / 2.0
    pts = pts * half_len + mid
    wts = wts * dom_len
    return pts, wts[:, 0]  # return weights as 1D


def _trapezoid_rule(lb, ub, npts):
    """Trapezoid rule on [lb, ub].

    Returns
    -------
    pts : ndarray, shape (npts,)
    weights : ndarray, shape (npts,)
    """
    pts = np.linspace(lb, ub, npts)
    deltax = pts[1] - pts[0]
    weights = np.ones(npts) * deltax
    weights[0] /= 2
    weights[-1] /= 2
    return pts, weights


class TestMeshKLE:

    def test_mesh_kle_1D_exponential(self, bkd) -> None:
        """Creates MeshKLE with ExponentialKernel and compares eigenvalues
        against analytical KLE1D. Also checks basis orthonormality
        with quadrature weights.
        """
        level = 10
        nterms = 3
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2**level + 1

        # Get Gauss-Legendre quadrature
        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        # Create ExponentialKernel (nu=0.5)
        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )

        # Analytical reference
        kle_exact = AnalyticalExponentialKLE1D(
            corr_len=len_scale, sigma2=sigma, dom_len=ub - lb, nterms=nterms
        )
        mesh_1d = bkd.to_numpy(mesh_coords[0, :])
        exact_basis = kle_exact.basis_values(mesh_1d)
        exact_eig_vals = kle_exact.eigenvalues()

        # Check eigenvalues match
        bkd.assert_allclose(
            kle.eigenvalues(),
            bkd.array(exact_eig_vals),
            rtol=3e-5,
        )

        # Check analytical basis is orthonormal under quadrature weights
        exact_basis_arr = bkd.array(exact_basis)
        identity = bkd.array(np.eye(nterms))
        bkd.assert_allclose(
            exact_basis_arr.T @ (quad_weights[:, None] * exact_basis_arr),
            identity,
            atol=1e-6,
        )

        # Check KLE eigenvectors are orthonormal under quadrature weights
        eig_vecs = kle.eigenvectors()
        bkd.assert_allclose(
            eig_vecs.T @ (quad_weights[:, None] * eig_vecs),
            identity,
            atol=1e-6,
        )

    def test_mesh_kle_1D_discretization_independence(self, bkd) -> None:
        """Tests that two different mesh resolutions with trapezoid rule
        give the same eigenvalues.
        """
        level1, level2 = 6, 8
        nterms = 3
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0

        # Fine mesh
        npts2 = 2 ** (level2 + 1) + 1
        pts2, wts2 = _trapezoid_rule(lb, ub, npts2)
        mesh_coords2 = bkd.array(pts2[None, :])
        quad_weights2 = bkd.array(wts2)

        lenscale_arr = bkd.array([len_scale])
        kernel2 = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)
        kle2 = MeshKLE(
            mesh_coords2,
            kernel2,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights2,
            bkd=bkd,
        )

        # Coarse mesh
        npts1 = 2**level1 + 1
        pts1, wts1 = _trapezoid_rule(lb, ub, npts1)
        mesh_coords1 = bkd.array(pts1[None, :])
        quad_weights1 = bkd.array(wts1)

        kernel1 = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)
        kle1 = MeshKLE(
            mesh_coords1,
            kernel1,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights1,
            bkd=bkd,
        )

        # Eigenvectors should be orthonormal under respective weights
        eig_vecs2 = kle2.eigenvectors()
        eig_vecs1 = kle1.eigenvectors()

        bkd.assert_allclose(
            bkd.sum(quad_weights2[:, None] * eig_vecs2**2, axis=0),
            bkd.ones(nterms),
            atol=1e-6,
        )
        bkd.assert_allclose(
            bkd.sum(quad_weights1[:, None] * eig_vecs1**2, axis=0),
            bkd.ones(nterms),
            atol=1e-6,
        )

        # Eigenvalues should match across resolutions
        bkd.assert_allclose(
            kle2._sqrt_eig_vals,
            kle1._sqrt_eig_vals,
            atol=3e-4,
        )

    def test_mesh_kle_multiple_kernels(self, bkd) -> None:
        """Test MeshKLE works with all supported kernel types."""
        nterms = 3
        npts = 50
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])

        kernel_classes = [
            SquaredExponentialKernel,
            Matern52Kernel,
            Matern32Kernel,
            ExponentialKernel,
        ]

        for KernelClass in kernel_classes:
            lenscale = bkd.array([0.5])
            kernel = KernelClass(lenscale, (0.01, 10.0), 1, bkd)
            kle = MeshKLE(
                mesh_coords,
                kernel,
                nterms=nterms,
                bkd=bkd,
            )
            # Eigenvalues should be positive
            assert bkd.all_bool(kle.eigenvalues() > 0)
            # Check shapes
            assert kle.eigenvectors().shape == (npts, nterms)
            assert kle.weighted_eigenvectors().shape == (npts, nterms)
            assert kle.eigenvalues().shape == (nterms,)

            # Evaluate
            coef = bkd.array(np.random.randn(nterms, 5))
            result = kle(coef)
            assert result.shape == (npts, 5)

    def test_mesh_kle_2D(self, bkd) -> None:
        """Test MeshKLE with a 2D mesh."""
        nterms = 3
        nx, ny = 5, 5
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        mesh_coords = bkd.array(np.vstack([xx.ravel(), yy.ravel()]))  # shape (2, 25)

        lenscale = bkd.array([0.5, 0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 2, bkd)
        kle = MeshKLE(mesh_coords, kernel, nterms=nterms, bkd=bkd)

        assert bkd.all_bool(kle.eigenvalues() > 0)
        assert kle.eigenvectors().shape == (25, nterms)

    def test_mesh_kle_use_log(self, bkd) -> None:
        """Test that use_log=True gives exp() of use_log=False result."""
        nterms = 3
        npts = 30
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)

        kle_no_log = MeshKLE(
            mesh_coords,
            kernel,
            nterms=nterms,
            use_log=False,
            bkd=bkd,
        )
        kle_log = MeshKLE(
            mesh_coords,
            kernel,
            nterms=nterms,
            use_log=True,
            bkd=bkd,
        )

        coef = bkd.array(np.random.randn(nterms, 4))
        result_no_log = kle_no_log(coef)
        result_log = kle_log(coef)

        bkd.assert_allclose(result_log, bkd.exp(result_no_log))

    def test_mesh_kle_edge_cases(self, bkd) -> None:
        """Test edge cases: nterms=1, nterms=all, non-zero mean."""
        npts = 20
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)

        # nterms=1
        kle1 = MeshKLE(mesh_coords, kernel, nterms=1, bkd=bkd)
        coef = bkd.array(np.random.randn(1, 3))
        result = kle1(coef)
        assert result.shape == (npts, 3)

        # nterms=None resolves to the terms the kernel can supply, not
        # one per point: a squared exponential at this lengthscale is
        # rank deficient, so the remaining points contribute modes with
        # no variance rather than usable ones.
        kle_all = MeshKLE(mesh_coords, kernel, bkd=bkd)
        assert 1 <= kle_all.nterms() < npts
        # Every retained term carries variance, which is what the count
        # is chosen to guarantee.
        assert float(bkd.to_numpy(kle_all.eigenvalues()).min()) > 0.0

        # Non-zero mean
        mean = 5.0
        kle_mean = MeshKLE(
            mesh_coords,
            kernel,
            mean_field=mean,
            nterms=3,
            bkd=bkd,
        )
        zero_coef = bkd.zeros((3, 1))
        result = kle_mean(zero_coef)
        bkd.assert_allclose(result, bkd.full((npts, 1), mean))

    def test_mesh_kle_coef_validation(self, bkd) -> None:
        """Test that invalid coefficients raise errors."""
        npts = 20
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)
        kle = MeshKLE(mesh_coords, kernel, nterms=3, bkd=bkd)

        # Wrong ndim
        with pytest.raises(ValueError):
            kle(bkd.array(np.random.randn(3)))

        # Wrong nterms
        with pytest.raises(ValueError):
            kle(bkd.array(np.random.randn(5, 2)))


class TestMeshKLETermsCarryVariance:
    """The basis never contains a mode the caller did not get.

    A KLE scales each eigenvector by ``sqrt(eigenvalue)``, so a term
    whose eigenvalue is zero to machine precision is a column of zeros.
    Smooth kernels are severely rank deficient, so this is easy to hit
    by accident rather than a pathological case.
    """

    def _kernel(self, bkd, lenscale=0.5):
        return SquaredExponentialKernel(
            bkd.array([lenscale]), (0.01, 10.0), 1, bkd
        )

    def test_over_requesting_raises(self, bkd) -> None:
        """Asking past the numerical rank is refused, not padded."""
        npts = 40
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        with pytest.raises(ValueError, match="rather than variance"):
            MeshKLE(mesh_coords, self._kernel(bkd), nterms=npts, bkd=bkd)

    def test_error_names_the_usable_count(self, bkd) -> None:
        """The message must say what to ask for instead.

        An error that only reports failure leaves the caller guessing a
        smaller number; the usable count is already known here.
        """
        npts = 40
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        usable = MeshKLE(
            mesh_coords, self._kernel(bkd), bkd=bkd
        ).nterms()
        with pytest.raises(ValueError, match=str(usable)):
            MeshKLE(mesh_coords, self._kernel(bkd), nterms=npts, bkd=bkd)

    def test_usable_count_is_accepted(self, bkd) -> None:
        """The count the error suggests must itself be requestable.

        Otherwise the guidance sends the caller into a second failure.
        """
        npts = 40
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        kernel = self._kernel(bkd)
        usable = MeshKLE(mesh_coords, kernel, bkd=bkd).nterms()
        kle = MeshKLE(mesh_coords, kernel, nterms=usable, bkd=bkd)
        assert kle.nterms() == usable
        assert float(bkd.to_numpy(kle.eigenvalues()).min()) > 0.0
        # One more must fail, so the count is the true boundary rather
        # than merely a safe under-estimate.
        with pytest.raises(ValueError, match="rather than variance"):
            MeshKLE(mesh_coords, kernel, nterms=usable + 1, bkd=bkd)

    def test_rougher_kernel_supplies_more_terms(self, bkd) -> None:
        """The count tracks the kernel, not a fixed cap.

        A shorter lengthscale decays more slowly, so it must yield
        strictly more usable terms on the same points.
        """
        npts = 40
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        smooth = MeshKLE(
            mesh_coords, self._kernel(bkd, 1.0), bkd=bkd
        ).nterms()
        rough = MeshKLE(
            mesh_coords, self._kernel(bkd, 0.1), bkd=bkd
        ).nterms()
        assert rough > smooth

    def test_full_rank_kernel_keeps_every_term(self, bkd) -> None:
        """Truncation is rank driven, so full rank must not truncate.

        An exponential kernel is only continuous, not smooth, and its
        spectrum stays well clear of machine precision at this size.
        """
        npts = 20
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        kernel = ExponentialKernel(bkd.array([0.5]), (0.01, 10.0), 1, bkd)
        kle = MeshKLE(mesh_coords, kernel, bkd=bkd)
        assert kle.nterms() == npts

    def test_weighted_case_also_truncates(self, bkd) -> None:
        """Quadrature weights must not bypass the check.

        The weighted path solves a different (symmetrized) operator, so
        it needs its own rank resolution rather than inheriting the
        unweighted one.
        """
        npts = 40
        mesh_coords, quad_weights = _gauss_legendre_quad(0, 1, npts, bkd)
        kernel = self._kernel(bkd)
        kle = MeshKLE(
            mesh_coords, kernel, quad_weights=quad_weights, bkd=bkd
        )
        assert 1 <= kle.nterms() < npts
        assert float(bkd.to_numpy(kle.eigenvalues()).min()) > 0.0
        with pytest.raises(ValueError, match="rather than variance"):
            MeshKLE(
                mesh_coords,
                kernel,
                nterms=npts,
                quad_weights=quad_weights,
                bkd=bkd,
            )
