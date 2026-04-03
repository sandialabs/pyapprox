"""Tests for PiecewiseDensityBasis."""

import numpy as np
import pytest

from pyapprox.probability.density.piecewise_density_basis import (
    PiecewiseDensityBasis,
)
from pyapprox.probability.density.protocols import (
    DensityBasisProtocol,
)
from pyapprox.surrogates.affine.expansions.pce_density import (
    composite_gauss_legendre,
)
from tests._helpers.markers import slow_test  # noqa: F401


class TestPiecewiseDensityBasis:

    def test_evaluate_shape_linear(self, bkd) -> None:
        basis = PiecewiseDensityBasis(-1.0, 1.0, 11, degree=1, bkd=bkd)
        y = bkd.reshape(bkd.linspace(-0.5, 0.5, 20), (1, -1))
        vals = basis.evaluate(y)
        bkd.assert_allclose(
            bkd.asarray([vals.shape[0]]),
            bkd.asarray([11]),
        )
        bkd.assert_allclose(
            bkd.asarray([vals.shape[1]]),
            bkd.asarray([20]),
        )

    def test_evaluate_shape_quadratic(self, bkd) -> None:
        basis = PiecewiseDensityBasis(-1.0, 1.0, 11, degree=2, bkd=bkd)
        y = bkd.reshape(bkd.linspace(-0.5, 0.5, 20), (1, -1))
        vals = basis.evaluate(y)
        bkd.assert_allclose(
            bkd.asarray([vals.shape[0]]),
            bkd.asarray([basis.nbasis()]),
        )
        bkd.assert_allclose(
            bkd.asarray([vals.shape[1]]),
            bkd.asarray([20]),
        )

    def test_mass_matrix_linear_analytical(self, bkd) -> None:
        """Verify tridiagonal entries match analytical formulas."""
        nbasis = 5
        basis = PiecewiseDensityBasis(0.0, 4.0, nbasis, degree=1, bkd=bkd)
        M = basis.mass_matrix()
        h = 1.0  # uniform spacing: 4.0 / (5-1) = 1.0

        # Boundary diagonals
        bkd.assert_allclose(
            bkd.asarray([float(bkd.to_numpy(M[0, 0]))]),
            bkd.asarray([h / 3.0]),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            bkd.asarray([float(bkd.to_numpy(M[4, 4]))]),
            bkd.asarray([h / 3.0]),
            rtol=1e-12,
        )

        # Interior diagonals: (h + h) / 3 = 2h/3
        for ii in range(1, 4):
            bkd.assert_allclose(
                bkd.asarray([float(bkd.to_numpy(M[ii, ii]))]),
                bkd.asarray([2.0 * h / 3.0]),
                rtol=1e-12,
            )

        # Off-diagonals: h/6
        for ii in range(4):
            bkd.assert_allclose(
                bkd.asarray([float(bkd.to_numpy(M[ii, ii + 1]))]),
                bkd.asarray([h / 6.0]),
                rtol=1e-12,
            )
            bkd.assert_allclose(
                bkd.asarray([float(bkd.to_numpy(M[ii + 1, ii]))]),
                bkd.asarray([h / 6.0]),
                rtol=1e-12,
            )

    def test_mass_matrix_linear_symmetry(self, bkd) -> None:
        basis = PiecewiseDensityBasis(-2.0, 3.0, 21, degree=1, bkd=bkd)
        M = basis.mass_matrix()
        bkd.assert_allclose(M, bkd.transpose(M, (1, 0)), rtol=1e-14)

    def test_mass_matrix_linear_vs_numerical(self, bkd) -> None:
        """Verify analytical mass matrix entries against numerical integration."""
        basis = PiecewiseDensityBasis(-1.0, 2.0, 7, degree=1, bkd=bkd)
        M = basis.mass_matrix()
        n = basis.nbasis()

        for ii in range(n):
            for jj in range(ii, min(ii + 2, n)):
                _ii, _jj = ii, jj

                def integrand(y_np: np.ndarray, i=_ii, j=_jj) -> np.ndarray:
                    y_arr = bkd.reshape(bkd.asarray(y_np), (1, -1))
                    Phi = basis.evaluate(y_arr)
                    return bkd.to_numpy(Phi[i]) * bkd.to_numpy(Phi[j])

                y_min, y_max = basis.domain()
                numerical = composite_gauss_legendre(
                    integrand, y_min, y_max, n_intervals=500, n_points=5
                )
                bkd.assert_allclose(
                    bkd.asarray([float(bkd.to_numpy(M[ii, jj]))]),
                    bkd.asarray([numerical]),
                    atol=1e-10,
                    rtol=1e-4,
                )

    @pytest.mark.slow_on("TorchBkd")
    def test_mass_matrix_quadratic_vs_numerical(self, bkd) -> None:
        """Verify quadratic mass matrix against numerical integration."""
        basis = PiecewiseDensityBasis(-1.0, 2.0, 7, degree=2, bkd=bkd)
        M = basis.mass_matrix()
        n = basis.nbasis()

        for ii in range(n):
            for jj in range(max(0, ii - 2), min(ii + 3, n)):
                _ii, _jj = ii, jj

                def integrand(y_np: np.ndarray, i=_ii, j=_jj) -> np.ndarray:
                    y_arr = bkd.reshape(bkd.asarray(y_np), (1, -1))
                    Phi = basis.evaluate(y_arr)
                    return bkd.to_numpy(Phi[i]) * bkd.to_numpy(Phi[j])

                y_min, y_max = basis.domain()
                numerical = composite_gauss_legendre(
                    integrand, y_min, y_max, n_intervals=500, n_points=5
                )
                bkd.assert_allclose(
                    bkd.asarray([float(bkd.to_numpy(M[ii, jj]))]),
                    bkd.asarray([numerical]),
                    atol=1e-10,
                    rtol=1e-4,
                )

    def test_mass_matrix_quadratic_symmetry(self, bkd) -> None:
        basis = PiecewiseDensityBasis(-2.0, 3.0, 21, degree=2, bkd=bkd)
        M = basis.mass_matrix()
        bkd.assert_allclose(M, bkd.transpose(M, (1, 0)), rtol=1e-14)

    def test_protocol_conformance(self, bkd) -> None:
        basis = PiecewiseDensityBasis(0.0, 1.0, 5, degree=1, bkd=bkd)
        assert isinstance(basis, DensityBasisProtocol)

    def test_domain(self, bkd) -> None:
        basis = PiecewiseDensityBasis(-3.0, 7.0, 11, degree=1, bkd=bkd)
        y_min, y_max = basis.domain()
        bkd.assert_allclose(
            bkd.asarray([y_min, y_max]),
            bkd.asarray([-3.0, 7.0]),
        )

    def test_nbasis_quadratic_forced_odd(self, bkd) -> None:
        """Quadratic basis forces nbasis to be odd."""
        basis = PiecewiseDensityBasis(0.0, 1.0, 10, degree=2, bkd=bkd)
        assert basis.nbasis() % 2 == 1

    def test_invalid_degree(self, bkd) -> None:
        with pytest.raises(ValueError):
            PiecewiseDensityBasis(0.0, 1.0, 5, degree=3, bkd=bkd)

    def test_partition_of_unity_linear(self, bkd) -> None:
        """Sum of hat functions should equal 1 on interior."""
        basis = PiecewiseDensityBasis(0.0, 1.0, 11, degree=1, bkd=bkd)
        y = bkd.reshape(bkd.linspace(0.01, 0.99, 50), (1, -1))
        vals = basis.evaluate(y)
        row_sums = bkd.sum(vals, axis=0)
        bkd.assert_allclose(row_sums, bkd.ones(50), rtol=1e-12)
