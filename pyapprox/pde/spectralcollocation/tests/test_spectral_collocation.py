import unittest
import numpy as np

from pyapprox.pde.spectralcollocation.spectral_collocation import (
    chebyshev_derivative_matrix,
    chebyshev_second_derivative_matrix,
    lagrange_polynomial_derivative_matrix_1d,
    lagrange_polynomial_derivative_matrix_2d,
    fourier_derivative_matrix, fourier_second_order_derivative_matrix,
    fourier_basis
)
from pyapprox.util.utilities import approx_jacobian, cartesian_product


class TestSpectralCollocation(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.eps = 2 * np.finfo(float).eps

    def test_derivative_matrix(self):
        order = 4
        derivative_matrix = chebyshev_derivative_matrix(order)[1]
        true_matrix = \
            [[5.5,        -6.82842712,  2.,         -1.17157288,  0.5],
             [1.70710678, -0.70710678, -1.41421356,  0.70710678, -0.29289322],
             [-0.5,         1.41421356, -0.,         -1.41421356,  0.5],
             [0.29289322, -0.70710678,  1.41421356,  0.70710678, -1.70710678],
             [-0.5,         1.17157288, -2.,          6.82842712, -5.5]]
        # I return points and calculate derivatives using reverse order of
        # points compared to what is used by Matlab cheb function thus the
        # derivative matrix I return will be the negative of the matlab version
        assert np.allclose(-derivative_matrix, true_matrix)

    def test_second_derivative_matrix(self):
        degree = 32
        # applying D1 twice for large degree suffers significant rounding error
        # TODO: sue methods in Section 3.3.5 of
        # Roger Peyret. Spectral Methods for Incompressible Viscous Flow forcing
        # to reduce roundoff errors
        pts, D1_mat = chebyshev_derivative_matrix(degree)
        D2_mat = chebyshev_second_derivative_matrix(degree)[1]

        # print(np.linalg.norm(D1_mat.dot(D1_mat)-D2_mat))
        assert np.allclose(D2_mat, D1_mat.dot(D1_mat))

        def fun(xx):
            return xx**(degree-2)

        def second_deriv(xx):
            return (degree-3)*(degree-2)*xx**(degree-4)
        # print(D2_mat.dot(fun(pts)))
        # print(D1_mat.dot(D1_mat.dot(fun(pts))))
        assert np.allclose(D2_mat.dot(fun(pts)), second_deriv(pts))

    def test_fourier_derivative_matrix(self):
        def fun(xx):
            return np.exp(np.sin(xx))

        def deriv(xx):
            vals = fun(xx)
            return np.cos(xx)*vals

        def deriv2(xx):
            vals = fun(xx)
            return (np.cos(xx)**2-np.sin(xx))*vals

        order = 31
        pts, D1_mat = fourier_derivative_matrix(order)
        vals = fun(pts)
        # print(D1_mat.dot(vals))
        # print(deriv(pts))
        assert np.allclose(D1_mat.dot(vals), deriv(pts))
        pts, D2_mat = fourier_second_order_derivative_matrix(order)
        assert np.allclose(D2_mat.dot(vals), deriv2(pts))

        # cannot interpolate at xx=np.pi*2
        # but the value at this point is the same as at x=0
        xx = np.linspace(0, np.pi*2-1e-15, 51)
        basis_matrix = fourier_basis(order, xx)
        # print(basis_matrix.dot(vals), fun(xx))
        assert np.allclose(basis_matrix.dot(vals), fun(xx))

        # test scaling of derivative_matrices
        pts, D1_mat = fourier_derivative_matrix(order)
        pts *= 2
        D1_mat *= 2*np.pi/(4*np.pi)
        vals = fun(pts)
        assert np.allclose(D1_mat.dot(vals), deriv(pts))

    def test_lagrange_polynomial_derivative_matrix_1d(self):
        order = 5
        degree = order-2

        def fun(x): return x**degree
        def deriv(x): return degree*x**(degree-1)

        pts, deriv_mat_1d = chebyshev_derivative_matrix(order)

        deriv_mat_barycentric, basis_vals = (
            lagrange_polynomial_derivative_matrix_1d(pts, pts))
        assert np.allclose(basis_vals.dot(fun(pts)), fun(pts))
        assert np.allclose(deriv_mat_1d, deriv_mat_barycentric)
        assert np.allclose(deriv_mat_1d.dot(fun(pts)), deriv(pts))

        interior_pts = pts[1:-1]
        deriv_mat_1d_interior = np.zeros((pts.shape[0], pts.shape[0]))
        dmat_int = lagrange_polynomial_derivative_matrix_1d(
                pts, interior_pts)[0]
        assert np.allclose(
            dmat_int.dot(fun(interior_pts)), deriv(pts))

        interior_pts = pts[1:-1]
        deriv_mat_1d_interior = np.zeros((pts.shape[0], pts.shape[0]))
        dmat_int = lagrange_polynomial_derivative_matrix_1d(
            interior_pts, pts)[0]
        print(dmat_int)
        assert np.allclose(
            dmat_int.dot(fun(pts)), deriv(interior_pts))

    def test_lagrange_polynomial_derivative_matrix_2d(self):
        np.set_printoptions(linewidth=300, threshold=2000)
        order = [4, 4]
        abscissa_1d = [chebyshev_derivative_matrix(o)[0][1:-1] for o in order]
        eval_samples = cartesian_product(
            [chebyshev_derivative_matrix(o)[0] for o in order])
        deriv_mat, basis_vals, abscissa = (
            lagrange_polynomial_derivative_matrix_2d(
                eval_samples, abscissa_1d))

        def wrapper(xx):
            basis_vals = lagrange_polynomial_derivative_matrix_2d(
                xx, abscissa_1d)[1]
            vals = basis_vals[0, :]
            return vals

        jac1, jac2 = [], []
        for sample in eval_samples.T:
            tmp = approx_jacobian(wrapper, sample[:, None]).T
            jac1.append(tmp[0])
            jac2.append(tmp[1])
        jac = np.array([jac1, jac2])
        assert np.allclose(jac, deriv_mat, atol=1e-7)

        # print(np.round(deriv_mat[0], decimals=2))

        def fun(xx): return (xx[0, :]**2*xx[1, :])[:, None]

        def deriv(xx):
            return np.vstack(((2*xx[0, :]*xx[1, :])[None, :],
                              (xx[0, :]**2)[None, :]))
        assert np.allclose(
            basis_vals.dot(fun(abscissa)), fun(eval_samples))
        assert np.allclose(
            deriv_mat.dot(fun(abscissa)[:, 0]), deriv(eval_samples))

        order = [4, 4]
        abscissa_1d = [chebyshev_derivative_matrix(o)[0] for o in order]
        eval_samples = cartesian_product(
            [chebyshev_derivative_matrix(o-2)[0] for o in order])
        deriv_mat, basis_vals, abscissa = (
            lagrange_polynomial_derivative_matrix_2d(
                eval_samples, abscissa_1d))

        print(np.round(deriv_mat[0], decimals=2))

        def fun(xx): return (xx[0, :]**2*xx[1, :])[:, None]

        def deriv(xx):
            return np.vstack(((2*xx[0, :]*xx[1, :])[None, :],
                              (xx[0, :]**2)[None, :]))
        assert np.allclose(
            basis_vals.dot(fun(abscissa)), fun(eval_samples))
        assert np.allclose(
            deriv_mat.dot(fun(abscissa)[:, 0]), deriv(eval_samples))

        def wrapper(xx):
            basis_vals = lagrange_polynomial_derivative_matrix_2d(
                xx, abscissa_1d)[1]
            vals = basis_vals[0, :]
            return vals

        jac1, jac2 = [], []
        for sample in eval_samples.T:
            tmp = approx_jacobian(wrapper, sample[:, None]).T
            jac1.append(tmp[0])
            jac2.append(tmp[1])
        jac = np.array([jac1, jac2])
        print(np.round(jac[0], decimals=2))
        assert np.allclose(jac, deriv_mat, atol=1e-7)


if __name__ == "__main__":
    spectral_collocation_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralCollocation)
    unittest.TextTestRunner(verbosity=2).run(spectral_collocation_test_suite)
