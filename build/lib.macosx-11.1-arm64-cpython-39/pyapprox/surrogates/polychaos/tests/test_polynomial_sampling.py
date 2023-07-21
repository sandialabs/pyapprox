import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.surrogates.orthopoly.leja_sequences import (
    christoffel_function, christoffel_weights, get_lu_leja_samples,
    interpolate_lu_leja_samples,
    get_quadrature_weights_from_lu_leja_samples,
)
from pyapprox.surrogates.polychaos.polynomial_sampling import (
    get_fekete_samples, interpolate_fekete_samples,
    get_quadrature_weights_from_fekete_samples,
    get_oli_leja_samples
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion,
    define_poly_options_from_variable_transformation,
    define_poly_options_from_variable
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation, RosenblattTransform,
    ComposeTransforms
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.benchmarks.genz import GenzFunction
from pyapprox.variables.tests.test_rosenblatt_transformation import (
    rosenblatt_example_2d
)


class TestPolynomialSampling(unittest.TestCase):

    def test_christoffel_function(self):
        num_vars = 1
        degree = 2
        alpha_poly = 0
        beta_poly = 0
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        num_samples = 11
        samples = np.linspace(-1., 1., num_samples)[np.newaxis, :]
        basis_matrix = poly.basis_matrix(samples)
        true_weights = 1./np.linalg.norm(basis_matrix, axis=1)**2
        weights = 1./christoffel_function(samples, poly.basis_matrix)
        assert weights.shape[0] == num_samples
        assert np.allclose(true_weights, weights)

        # For a Gaussian quadrature rule of degree p that exactly
        # integrates all polynomials up to and including degree 2p-1
        # the quadrature weights are the christoffel function
        # evaluated at the quadrature samples
        quad_samples, quad_weights = gauss_jacobi_pts_wts_1D(
            degree, alpha_poly, beta_poly)
        quad_samples = quad_samples[np.newaxis, :]
        basis_matrix = poly.basis_matrix(quad_samples)
        weights = 1./christoffel_function(quad_samples, poly.basis_matrix)
        assert np.allclose(weights, quad_weights)

    def test_fekete_gauss_lobatto(self):
        num_vars = 1
        degree = 3
        num_candidate_samples = 10000
        def generate_candidate_samples(
            n): return np.linspace(-1., 1., n)[np.newaxis, :]

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        def precond_func(matrix, samples): return 0.25*np.ones(matrix.shape[0])
        samples, _ = get_fekete_samples(
            poly.basis_matrix, generate_candidate_samples,
            num_candidate_samples, preconditioning_function=precond_func)
        assert samples.shape[1] == degree+1

        # The samples should be close to the Gauss-Lobatto samples
        gauss_lobatto_samples = np.asarray(
            [-1.0, - 0.447213595499957939281834733746,
             0.447213595499957939281834733746, 1.0])
        assert np.allclose(np.sort(samples), gauss_lobatto_samples, atol=1e-1)

    def test_fekete_interpolation(self):
        num_vars = 2
        degree = 30

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        def generate_candidate_samples(n): return np.cos(
            np.random.uniform(0., np.pi, (num_vars, n)))

        # must use canonical_basis_matrix to generate basis matrix
        def precond_func(matrix, samples): return christoffel_weights(matrix)
        canonical_samples, data_structures = get_fekete_samples(
            poly.canonical_basis_matrix, generate_candidate_samples,
            num_candidate_samples, preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical(canonical_samples)

        assert samples.max() <= 1 and samples.min() >= 0.

        c = np.random.uniform(0., 1., (num_vars, 1))
        c *= 20/c.sum()
        w = np.zeros_like(c)
        w[0] = np.random.uniform(0., 1., 1)
        genz_function = GenzFunction()
        genz_function.set_coefficients(num_vars, 1, 'none')
        # genz_function._c, genz_function._w = c, w
        genz_name = 'oscillatory'
        fun = partial(genz_function, genz_name)
        values = fun(samples)

        # Ensure coef produce an interpolant
        coef = interpolate_fekete_samples(
            canonical_samples, values, data_structures)
        poly.set_coefficients(coef)
        assert np.allclose(poly(samples), values)

        valid_samples = var_trans.variable.rvs(1e3)
        assert np.allclose(poly(valid_samples), fun(valid_samples))

        quad_w = get_quadrature_weights_from_fekete_samples(
            canonical_samples, data_structures)
        values_at_quad_x = values[:, 0]
        # increase degree if want smaller atol
        # print(np.dot(values_at_quad_x, quad_w) -
        #       genz_function.integrate(genz_name))
        assert np.allclose(
            np.dot(values_at_quad_x, quad_w),
            genz_function.integrate(genz_name), atol=1e-4)

    def test_lu_leja_interpolation(self):
        num_vars = 2
        degree = 15

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        def generate_candidate_samples(n): return np.cos(
            np.random.uniform(0., np.pi, (num_vars, n)))

        # must use canonical_basis_matrix to generate basis matrix
        num_leja_samples = indices.shape[1]-1
        def precond_func(matrix, samples): return christoffel_weights(matrix)
        samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix, generate_candidate_samples,
            num_candidate_samples, num_leja_samples,
            preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical(samples)

        assert samples.max() <= 1 and samples.min() >= 0.

        c = np.random.uniform(0., 1., num_vars)
        c *= 20/c.sum()
        w = np.zeros_like(c)
        w[0] = np.random.uniform(0., 1., 1)
        genz_function = GenzFunction()
        genz_function.set_coefficients(num_vars, 1, 'none')
        # genz_function._c, genz_function._w = c, w
        genz_name = 'oscillatory'
        fun = partial(genz_function, genz_name)

        values = fun(samples)

        # Ensure coef produce an interpolant
        coef = interpolate_lu_leja_samples(samples, values, data_structures)

        # Ignore basis functions (columns) that were not considered during the
        # incomplete LU factorization
        poly.set_indices(poly.indices[:, :num_leja_samples])
        poly.set_coefficients(coef)

        assert np.allclose(poly(samples), values)

        quad_w = get_quadrature_weights_from_lu_leja_samples(
            samples, data_structures)
        values_at_quad_x = values[:, 0]

        # will get closer if degree is increased
        # print (np.dot(values_at_quad_x,quad_w),genz_function.integrate())
        assert np.allclose(
            np.dot(values_at_quad_x, quad_w),
            genz_function.integrate(genz_name),
            atol=1e-4)

    def test_lu_leja_interpolation_with_intial_samples(self):
        num_vars = 2
        degree = 15

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        def generate_candidate_samples(n): return np.cos(
            np.random.uniform(0., np.pi, (num_vars, n)))

        # enforcing lu interpolation to interpolate a set of initial points
        # before selecting best samples from candidates can cause ill
        # conditioning to avoid this issue build a leja sequence and use
        # this as initial samples and then recompute sequence with different
        # candidates must use canonical_basis_matrix to generate basis matrix
        num_initial_samples = 5
        initial_samples = None
        num_leja_samples = indices.shape[1]-1
        def precond_func(matrix, samples): return christoffel_weights(matrix)
        initial_samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix, generate_candidate_samples,
            num_candidate_samples, num_initial_samples,
            preconditioning_function=precond_func,
            initial_samples=initial_samples)

        samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix, generate_candidate_samples,
            num_candidate_samples, num_leja_samples,
            preconditioning_function=precond_func,
            initial_samples=initial_samples)

        assert np.allclose(samples[:, :num_initial_samples], initial_samples)

        samples = var_trans.map_from_canonical(samples)

        assert samples.max() <= 1 and samples.min() >= 0.

        c = np.random.uniform(0., 1., num_vars)
        c *= 20/c.sum()
        w = np.zeros_like(c)
        w[0] = np.random.uniform(0., 1., 1)
        genz_function = GenzFunction()
        genz_function.set_coefficients(num_vars, 1, 'none')
        # genz_function._c, genz_function._w = c, w
        genz_name = 'oscillatory'
        fun = partial(genz_function, genz_name)
        values = fun(samples)

        # Ensure coef produce an interpolant
        coef = interpolate_lu_leja_samples(samples, values, data_structures)

        # Ignore basis functions (columns) that were not considered during the
        # incomplete LU factorization
        poly.set_indices(poly.indices[:, :num_leja_samples])
        poly.set_coefficients(coef)

        assert np.allclose(poly(samples), values)

        quad_w = get_quadrature_weights_from_lu_leja_samples(
            samples, data_structures)
        values_at_quad_x = values[:, 0]
        assert np.allclose(
            np.dot(values_at_quad_x, quad_w),
            genz_function.integrate(genz_name),
            atol=1e-4)

    def test_oli_leja_interpolation(self):
        num_vars = 2
        degree = 5

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000

        # oli_leja requires candidates in user space
        def generate_candidate_samples(n): return (np.cos(
            np.random.uniform(0., np.pi, (num_vars, n)))+1)/2.

        # must use canonical_basis_matrix to generate basis matrix
        num_leja_samples = indices.shape[1]-3
        def precond_func(samples): return 1./christoffel_function(
            samples, poly.basis_matrix)
        samples, data_structures = get_oli_leja_samples(
            poly, generate_candidate_samples,
            num_candidate_samples, num_leja_samples,
            preconditioning_function=precond_func)

        assert samples.max() <= 1 and samples.min() >= 0.

        # c = np.random.uniform(0., 1., num_vars)
        # c *= 20/c.sum()
        # w = np.zeros_like(c)
        # w[0] = np.random.uniform(0., 1., 1)
        # genz_function = GenzFunction('oscillatory', num_vars, c=c, w=w)
        # values = genz_function(samples)
        # exact_integral = genz_function.integrate()

        values = np.sum(samples**2, axis=0)[:, None]
        # exact_integral = num_vars/3

        # Ensure we have produced an interpolant
        oli_solver = data_structures[0]
        poly = oli_solver.get_current_interpolant(samples, values)
        assert np.allclose(poly(samples), values)

        # quad_w = get_quadrature_weights_from_oli_leja_samples(
        #     samples, data_structures)
        # print(data_structures, num_leja_samples)
        # values_at_quad_x = values[:, 0]
        # print(np.dot(values_at_quad_x, quad_w), exact_integral)
        # assert np.allclose(
        #     np.dot(values_at_quad_x, quad_w), exact_integral)

    def test_fekete_rosenblatt_interpolation(self):
        np.random.seed(2)
        degree = 3

        __, __, joint_density, limits = rosenblatt_example_2d(num_samples=1)
        num_vars = len(limits)//2

        rosenblatt_opts = {'limits': limits, 'num_quad_samples_1d': 20}
        var_trans_1 = RosenblattTransform(
            joint_density, num_vars, rosenblatt_opts)
        # rosenblatt maps to [0,1] but polynomials of bounded variables
        # are in [-1,1] so add second transformation for this second mapping
        var_trans_2 = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        var_trans = ComposeTransforms([var_trans_1, var_trans_2])

        poly = PolynomialChaosExpansion()
        # use var_trans2 to configure polynomial recursions
        # but use var_trans to map samples
        poly_opts = {'var_trans': var_trans}
        basis_opts = define_poly_options_from_variable(var_trans_2.variable)
        poly_opts['poly_types'] = basis_opts
        poly.configure(poly_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        num_candidate_samples = 10000
        def generate_candidate_samples(n): return np.cos(
            np.random.uniform(0., np.pi, (num_vars, n)))

        def precond_func(matrix, samples): return christoffel_weights(matrix)
        canonical_samples, data_structures = get_fekete_samples(
            poly.canonical_basis_matrix, generate_candidate_samples,
            num_candidate_samples, preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical(canonical_samples)
        assert np.allclose(
            canonical_samples, var_trans.map_to_canonical(samples))

        assert samples.max() <= 1 and samples.min() >= 0.

        c = np.random.uniform(0., 1., num_vars)
        c *= 20/c.sum()
        w = np.zeros_like(c)
        w[0] = np.random.uniform(0., 1., 1)
        genz_function = GenzFunction()
        genz_function.set_coefficients(num_vars, 1, 'none')
        # genz_function._c, genz_function._w = c, w
        genz_name = 'oscillatory'
        fun = partial(genz_function, genz_name)
        values = fun(samples)
        
        # Ensure coef produce an interpolant
        coef = interpolate_fekete_samples(
            canonical_samples, values, data_structures)
        poly.set_coefficients(coef)

        assert np.allclose(poly(samples), values)

        # compare mean computed using quadrature and mean computed using
        # first coefficient of expansion. This is not testing that mean
        # is correct because rosenblatt transformation introduces large error
        # which makes it hard to compute accurate mean from pce or quadrature
        quad_w = get_quadrature_weights_from_fekete_samples(
            canonical_samples, data_structures)
        values_at_quad_x = values[:, 0]
        assert np.allclose(
            np.dot(values_at_quad_x, quad_w), poly.mean())


if __name__ == "__main__":
    polynomial_sampling_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestPolynomialSampling)
    unittest.TextTestRunner(verbosity=2).run(polynomial_sampling_test_suite)
