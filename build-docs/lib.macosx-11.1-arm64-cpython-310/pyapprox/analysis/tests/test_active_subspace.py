import numpy as np
from scipy import stats
import unittest

from pyapprox.surrogates.interp.manipulate_polynomials import group_like_terms
from pyapprox.surrogates.interp.monomial import (
    monomial_mean_uniform_variables,
    monomial_basis_matrix, univariate_monomial_basis_matrix
)
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, argsort_indices_leixographically
)
from pyapprox.util.sys_utilities import package_available
from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.analysis.active_subspace import (
    get_random_active_subspace_eigenvecs,
    transform_active_subspace_samples_to_original_coordinates,
    get_zonotope_vertices_and_bounds, get_uniform_samples_on_zonotope,
    evaluate_active_subspace_density_1d_example,
    coeffs_of_active_subspace_polynomial,
    moments_of_active_subspace, evaluate_active_subspace_density_1d,
    inner_products_on_active_subspace,
    sample_based_inner_products_on_active_subspace
)


skiptest = unittest.skipIf(
    not package_available("active_subspaces"),
    reason="active_subspace package missing")


class TestActiveSubspace(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_get_chebyhsev_center_of_inactive_subspace(self):
        # Define an active subspace
        num_vars = 6
        num_active_vars = 2

        W, W1, W2 = get_random_active_subspace_eigenvecs(
            num_vars, num_active_vars)

        num_active_samples = 100
        # generate samples in the active subspace zonotope
        samples = np.random.uniform(-1., 1., (num_vars, num_active_samples))
        active_samples = np.dot(W1.T, samples)
        # get active coordinates of hypercube vertices.
        # Warning. Because of approximate nature of find_zonotope_vertices
        # algorithm not all mapped vertices may lie within the
        # linear constraints that the find vertices algorithm generates
        hypercube_vertices_1d = np.array([-1., 1.])
        hypercube_vertices = cartesian_product(
            [hypercube_vertices_1d]*num_vars, 1)
        mapped_vertices = np.dot(W1.T, hypercube_vertices)
        active_samples = np.hstack((active_samples, mapped_vertices))
        samples = transform_active_subspace_samples_to_original_coordinates(
            active_samples, W)
        # check the sample in the original space fits inside the
        # hypercube. Allow for slight machine precision error
        print(samples.min(), samples.max())
        assert np.all(
            np.absolute(samples) <= 1+1e-8)

    @skiptest
    def test_get_uniform_samples_on_zonotope(self):
        num_vars = 6
        num_active_vars = 2
        W, W1, W2 = get_random_active_subspace_eigenvecs(
            num_vars, num_active_vars)
        vertices = get_zonotope_vertices_and_bounds(W1)[0]

        num_samples = 100
        active_samples = get_uniform_samples_on_zonotope(num_samples, vertices)
        samples = transform_active_subspace_samples_to_original_coordinates(
            active_samples, W)
        # check the sample in the original space fits inside the
        # hypercube. Allow for slight machine precision error
        assert np.all(
            np.absolute(samples) <= 1+4*np.finfo(float).eps)

    def test_evaluate_active_subspace_density_1d(self):
        alpha = 2
        beta = 5

        def beta_density_function_1d(x):
            return stats.beta.pdf((x+1.)/2., alpha, beta)/2
        def beta_density_fn(x): return beta_density_function_1d(
            x[0, :])*beta_density_function_1d(x[1, :])

        evaluate_active_subspace_density_1d_example(
            beta_density_fn, 1e-6, test=True)

        def uniform_density_fn(x): return np.ones(x.shape[1])*0.25
        evaluate_active_subspace_density_1d_example(
            uniform_density_fn, 1e-3, test=True)

    def test_coeffs_of_active_subspace_polynomial(self):
        num_vars = 3
        num_active_vars = 2
        degree = 4
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]
        W1 = np.array([[1, 2, 3], [4, 5, 6]]).T
        as_poly_indices = compute_hyperbolic_indices(
            num_active_vars, degree, 1.0)
        sorted_as_poly_idx = argsort_indices_leixographically(as_poly_indices)

        # (dx+ey+fz)^2
        coeffs, indices = coeffs_of_active_subspace_polynomial(
            W1.T, as_poly_indices[:, sorted_as_poly_idx[3]])
        sorted_idx = argsort_indices_leixographically(indices)
        true_coeffs = np.array([W1[2, 1]**2, 2*W1[1, 1]*W1[2, 1], W1[1, 1]**2,
                                2*W1[0, 1]*W1[2, 1], 2*W1[0, 1]*W1[1, 1],
                                W1[0, 1]**2])
        assert np.allclose(true_coeffs, coeffs[sorted_idx])

        # (ax+by+cz)*(dx+ey+fz)=
        # a d x^2+a e x y+a f x z+b d x y+b e y^2+b f y z+c d x z+c e y z + c f z^2=
        # cfz^2 + (ce+bf)yz + bey^2 + (af+cd)xz + (ae+bd)xy + adx^2
        coeffs, indices = coeffs_of_active_subspace_polynomial(
            W1.T, as_poly_indices[:, sorted_as_poly_idx[4]])
        indices, coeffs = group_like_terms(coeffs, indices)
        sorted_idx = argsort_indices_leixographically(indices)
        a, b, c = W1[:, 0]
        d, e, f = W1[:, 1]
        true_coeffs = np.array([c*f, c*e+b*f, b*e, a*f+c*d, a*e+b*d, a*d])
        assert np.allclose(true_coeffs, coeffs[sorted_idx].squeeze())

        # (ax+by+cz)^4
        # a^4 x^4 + 4 a^3 c x^3 z + 4 b a^3 x^3 y + 6 a^2 c^2 x^2 z^2 + 12 b a^2 c x^2 y z + 6 b^2 a^2 x^2 y^2 + 4 a c^3 x z^3 + 12 b a c^2 x y z^2 + 12 b^2 a c x y^2 z + 4 b^3 a x y^3 + c^4 z^4 + 4 b c^3 y z^3 + 6 b^2 c^2 y^2 z^2 + 4 b^3 c y^3 z + b^4 y^4
        coeffs, indices = coeffs_of_active_subspace_polynomial(
            W1.T, as_poly_indices[:, sorted_as_poly_idx[14]])
        sorted_idx = argsort_indices_leixographically(indices)
        #print_sorted_indices(indices, num_vars, sorted_idx)
        true_coeffs = np.array(
            [c**4, 4.*b*c**3, 6.*b**2*c**2, 4.*b**3*c, b**4, 4*a*c**3,
             12.*b*a*c**2, 12.*b**2*a*c, 4.*b**3*a, 6.*a**2*c**2,
             12.*b*a**2*c, 6*b**2*a**2, 4*a**3*c, 4*b*a**3, a**4])
        assert np.allclose(true_coeffs, coeffs[sorted_idx])

        # (dx+ey+fz)^4
        # d^4 x^4 + 4 d^3 f x^3 z + 4 e d^3 x^3 y + 6 d^2 f^2 x^2 z^2 + 12 e d^2 f x^2 y z + 6 e^2 d^2 x^2 y^2 + 4 d f^3 x z^3 + 12 e d f^2 x y z^2 + 12 e^2 d f x y^2 z + 4 e^3 d x y^3 + f^4 z^4 + 4 e f^3 y z^3 + 6 e^2 f^2 y^2 z^2 + 4 e^3 f y^3 z + e^4 y^4
        coeffs, indices = coeffs_of_active_subspace_polynomial(
            W1.T, as_poly_indices[:, sorted_as_poly_idx[10]])
        sorted_idx = argsort_indices_leixographically(indices)
        true_coeffs = np.array(
            [f**4, 4.*e*f**3, 6.*e**2*f**2, 4.*e**3*f, e**4, 4*d*f**3,
             12.*e*d*f**2, 12.*e**2*d*f, 4.*e**3*d, 6.*d**2*f**2,
             12.*e*d**2*f, 6*e**2*d**2, 4*d**3*f, 4*e*d**3, d**4])
        assert np.allclose(true_coeffs, coeffs[sorted_idx])

    def test_moments_of_active_subspace(self):
        num_vars = 3
        num_active_vars = 2
        degree = 2
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]
        W1 = np.arange(1, 7).reshape(3, 2, order='F')

        as_poly_indices = compute_hyperbolic_indices(
            num_active_vars, degree, 1.0)
        moments = moments_of_active_subspace(
            W1.T, as_poly_indices, monomial_mean_uniform_variables)

        sorted_idx = argsort_indices_leixographically(
            as_poly_indices)
        # (ax+by+cz)^2=a^2 x^2 + 2 a b x y + 2 a c x z + b^2 y^2 + 2 b c y z + c^2 z^2
        # int (ax+by+cz)^2*1/2dx = a^2*1/3 + b^2*1/3 + c^2*1/3
        true_moments = [1, 0, 0,
                        # notice how if W1 has columns with unit norm np.sum(W1[:,1]**2) will always be one.
                        np.sum(W1[:, 1]**2)*1./3.,
                        1/3*(W1.prod(axis=1)).sum(),
                        np.sum(W1[:, 0]**2)*1./3.]
        assert np.allclose(moments[sorted_idx], true_moments)

        num_vars = 3
        num_active_vars = 2
        degree = 4
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]
        W1[:, 0] = [1, 2, 3]
        W1[:, 1] = [4, 5, 6]
        as_poly_indices = compute_hyperbolic_indices(
            num_active_vars, degree, 1.0)
        moments = moments_of_active_subspace(
            W1.T, as_poly_indices, monomial_mean_uniform_variables)
        sorted_idx = argsort_indices_leixographically(as_poly_indices)
        a, b, c = W1[:, 0]
        d, e, f = W1[:, 1]
        dummy = np.inf  # yet to analytically compute true moments for these indices
        true_moments = np.array([1, 0, 0, np.sum(W1[:, 1]**2)*1./3.,
                                 1./3.*(c*f+b*e+a*d),
                                 np.sum(W1[:, 0]**2)*1./3., 0., 0., 0., 0.,
                                 (3*d**4+3*e**4+10*e**2*f**2+3 *
                                  f**4+10*d**2*(e**2+f**2))/15.,
                                 dummy, dummy, dummy,
                                 (3*a**4+3*b**4+10*b**2*c**2+3*c**4+10*a**2*(b**2+c**2))/15.])
        moments = moments[sorted_idx]
        # ignore dummy values until I compute them analytically
        II = np.where(true_moments != np.Inf)[0]
        assert np.allclose(moments[II], true_moments[II])

    def test_moments_of_active_subspace_II(self):
        num_vars = 4
        num_active_vars = 2
        degree = 4
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]

        as_poly_indices = compute_hyperbolic_indices(
            num_active_vars, degree, 1.0)
        moments = moments_of_active_subspace(
            W1.T, as_poly_indices, monomial_mean_uniform_variables)

        x1d, w1d = np.polynomial.legendre.leggauss(30)
        w1d /= 2
        gl_samples = cartesian_product([x1d]*num_vars)
        gl_weights = outer_product([w1d]*num_vars)
        as_gl_samples = np.dot(W1.T, gl_samples)

        vandermonde = monomial_basis_matrix(as_poly_indices, as_gl_samples)
        quad_poly_moments = np.empty(vandermonde.shape[1])
        for ii in range(vandermonde.shape[1]):
            quad_poly_moments[ii] = np.dot(vandermonde[:, ii], gl_weights)
        assert np.allclose(moments, quad_poly_moments)

    def test_evaluate_active_subspace_density_1d_moments(self):
        num_vars = 2
        num_active_vars = 1
        degree = 3
        W, W1, W2 = get_random_active_subspace_eigenvecs(
            num_vars, num_active_vars)

        def density_fn(x): return np.ones(x.shape[1])*0.25

        indices = compute_hyperbolic_indices(num_active_vars, degree, 1.0)

        x1d, w1d = np.polynomial.legendre.leggauss(100)
        w1d /= 2

        as_density_vals = evaluate_active_subspace_density_1d(
            W, density_fn, plot_steps=False, points_for_eval=x1d)
        basis_matrix = univariate_monomial_basis_matrix(degree, x1d)

        assert np.allclose(
            moments_of_active_subspace(
                W1.T, indices, monomial_mean_uniform_variables),
            np.dot(basis_matrix.T, w1d))

    def test_inner_products_on_active_subspace(self):
        num_vars = 4
        num_active_vars = 2
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]

        as_poly_indices = np.asarray([
            [0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]
        ]).T

        x1d, w1d = np.polynomial.legendre.leggauss(10)
        w1d /= 2
        gl_samples = cartesian_product([x1d]*num_vars)
        gl_weights = outer_product([w1d]*num_vars)
        as_gl_samples = np.dot(W1.T, gl_samples)

        inner_product_indices = np.empty(
            (num_active_vars, as_poly_indices.shape[1]**2), dtype=int)
        for ii in range(as_poly_indices.shape[1]):
            for jj in range(as_poly_indices.shape[1]):
                inner_product_indices[:, ii*as_poly_indices.shape[1]+jj] =\
                    as_poly_indices[:, ii]+as_poly_indices[:, jj]

        vandermonde = monomial_basis_matrix(
            inner_product_indices, as_gl_samples)

        inner_products = inner_products_on_active_subspace(
            W1.T, as_poly_indices, monomial_mean_uniform_variables)

        for ii in range(as_poly_indices.shape[1]):
            for jj in range(as_poly_indices.shape[1]):
                assert np.allclose(
                    inner_products[ii, jj],
                    np.dot(vandermonde[:, ii*as_poly_indices.shape[1]+jj],
                           gl_weights))

    def test_inner_products_on_active_subspace_using_samples(self):

        def generate_samples(num_samples):
            from pyapprox.expdesign.low_discrepancy_sequences import \
                transformed_halton_sequence
            samples = transformed_halton_sequence(None, num_vars, num_samples)
            samples = samples*2.-1.
            return samples

        num_vars = 4
        num_active_vars = 2
        A = np.random.normal(0, 1, (num_vars, num_vars))
        Q, R = np.linalg.qr(A)
        W1 = Q[:, :num_active_vars]

        as_poly_indices = np.asarray([
            [0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]
        ]).T

        x1d, w1d = np.polynomial.legendre.leggauss(10)
        w1d /= 2
        gl_samples = cartesian_product([x1d]*num_vars)
        gl_weights = outer_product([w1d]*num_vars)
        as_gl_samples = np.dot(W1.T, gl_samples)

        inner_product_indices = np.empty(
            (num_active_vars, as_poly_indices.shape[1]**2), dtype=int)
        for ii in range(as_poly_indices.shape[1]):
            for jj in range(as_poly_indices.shape[1]):
                inner_product_indices[:, ii*as_poly_indices.shape[1]+jj] =\
                    as_poly_indices[:, ii]+as_poly_indices[:, jj]

        vandermonde = monomial_basis_matrix(
            inner_product_indices, as_gl_samples)

        num_sobol_samples = 100000
        inner_products = sample_based_inner_products_on_active_subspace(
            W1, monomial_basis_matrix, as_poly_indices, num_sobol_samples,
            generate_samples)

        for ii in range(as_poly_indices.shape[1]):
            for jj in range(as_poly_indices.shape[1]):
                assert np.allclose(
                    inner_products[ii, jj],
                    np.dot(vandermonde[:, ii*as_poly_indices.shape[1]+jj],
                           gl_weights), atol=1e-4)


if __name__ == "__main__":
    active_subspace_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestActiveSubspace)
    unittest.TextTestRunner(verbosity=2).run(active_subspace_test_suite)
