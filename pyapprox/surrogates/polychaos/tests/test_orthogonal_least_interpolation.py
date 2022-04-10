import unittest
from functools import partial
from scipy.stats import beta as beta, uniform
import numpy as np
from numpy.linalg import norm

from pyapprox.surrogates.polychaos.orthogonal_least_interpolation import (
    get_block_diagonal_matrix_num_cols, get_block_diagonal_matrix_num_rows,
    pre_multiply_block_diagonal_matrix,  LeastInterpolationSolver
)
from pyapprox.util.utilities import cartesian_product
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_level_indices, compute_hyperbolic_indices,
    compute_tensor_product_level_indices, get_total_degree
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_pts_wts_1D, gauss_hermite_pts_wts_1D
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
)
from pyapprox.util.utilities import (
    remove_common_rows, allclose_unsorted_matrix_rows
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.variables.density import tensor_product_pdf
from pyapprox.surrogates.orthopoly.leja_sequences import christoffel_function
from pyapprox.util.linalg import truncated_pivoted_lu_factorization


class TestBlockDiagonalOperations(unittest.TestCase):

    def test_block_diagonal_matrix_pre_multiply(self):
        A = np.asarray([0., 2., 2., 0.]).reshape((2, 2), order='F')
        B = np.asarray([1., 4., 2., 5., 3., 6.]).reshape((2, 3), order='F')
        C = np.asarray([1., 3., 5., 2., 4., 6.]).reshape((3, 2), order='F')
        D = np.asarray([5., 0., 5., 0., 5., 0., 5., 0., 5.]
                       ).reshape((3, 3), order='F')

        # ----------------------------------------- #
        # Square rectangular block matrix           #
        # ----------------------------------------- #
        matrix_blocks = [A, B, C, D]
        assert get_block_diagonal_matrix_num_rows(matrix_blocks) == 10
        assert get_block_diagonal_matrix_num_cols(matrix_blocks) == 10

        matrix = np.empty((10, 10), dtype=float)
        for ii in range(matrix.shape[0]):
            matrix[ii, :] = np.arange(matrix.shape[1])*10+ii+1

        result = pre_multiply_block_diagonal_matrix(
            matrix, matrix_blocks, False)

        exact_result = np.asarray(
            [4, 2, 26, 62, 20, 46, 72, 90, 45, 90, 24, 22, 86, 212,
             50, 116, 182, 190, 95, 190, 44, 42, 146, 362, 80, 186, 292, 290,
             145, 290, 64, 62, 206, 512, 110, 256, 402, 390, 195, 390, 84, 82,
             266, 662, 140, 326, 512, 490, 245, 490, 104, 102, 326, 812, 170,
             396, 622, 590, 295, 590, 124, 122, 386, 962, 200, 466, 732, 690,
             345, 690, 144, 142, 446, 1112, 230, 536, 842, 790, 395, 790, 164,
             162, 506, 1262, 260, 606, 952, 890, 445, 890, 184, 182, 566, 1412,
             290, 676, 1062, 990, 495, 990]).reshape((10, 10), order='F')
        assert np.allclose(result, exact_result)

        result = pre_multiply_block_diagonal_matrix(
            matrix, matrix_blocks, True)
        exact_result = np.asarray([
            4, 2,  19,  26,  33,  58,  76,  90,  45,  90,  24,  22,  69,  96,
            123, 148, 196, 190,  95, 190,  44,  42, 119, 166, 213, 238, 316, 290,
            145, 290,  64,  62, 169, 236, 303, 328, 436, 390, 195, 390,  84,  82,
            219, 306, 393, 418, 556, 490, 245, 490, 104, 102, 269, 376, 483, 508,
            676, 590, 295, 590, 124, 122, 319, 446, 573, 598, 796, 690, 345, 690,
            144, 142, 369, 516, 663, 688, 916, 790, 395, 790, 164, 162, 419, 586,
            753, 778, 1036, 890, 445, 890, 184, 182, 469, 656, 843, 868, 1156, 990,
            495, 990]).reshape((10, 10), order='F')
        assert np.allclose(result, exact_result)

        # ----------------------------------------- #
        # Under-determined rectangular block matrix #
        # ----------------------------------------- #

        OO = np.asarray([[1.]])
        matrix_blocks = [OO, A, B]

        matrix3 = matrix[:6, :6]

        # Test BlockMatrix*E
        result = pre_multiply_block_diagonal_matrix(
            matrix3, matrix_blocks, False)

        exact_result = np.asarray([
            1,  6,  4, 32, 77, 11, 26, 24, 92, 227, 21, 46, 44, 152, 377, 31,
            66, 64, 212, 527, 41, 86, 84, 272, 677, 51, 106, 104, 332, 827]
        ).reshape((5, 6), order='F')
        assert np.allclose(exact_result, result)

        # Test BlockMatrix'*E
        matrix4 = matrix[:5, :6]
        result = pre_multiply_block_diagonal_matrix(
            matrix4, matrix_blocks, True)

        exact_result = np.asarray(
            [1,  6,  4, 24, 33, 42, 11, 26, 24, 74, 103, 132, 21, 46, 44, 124,
             173, 222, 31, 66, 64, 174, 243, 312, 41, 86, 84, 224, 313, 402,
             51, 106, 104, 274, 383, 492]).reshape((6, 6), order='F')
        assert np.allclose(exact_result, result)

        # ----------------------------------------- #
        # Over-determined rectangular block matrix #
        # ----------------------------------------- #

        matrix_blocks = [OO, A, C]
        # Test BlockMatrix*E
        matrix5 = matrix[:5, :6]
        result = pre_multiply_block_diagonal_matrix(
            matrix5, matrix_blocks, False)

        exact_result = np.asarray(
            [1,  6,  4, 14, 32, 50, 11, 26, 24, 44, 102, 160, 21, 46, 44, 74,
             172, 270, 31, 66, 64, 104, 242, 380, 41, 86, 84, 134, 312, 490,
             51, 106, 104, 164, 382, 600]).reshape((6, 6), order='F')
        assert np.allclose(exact_result, result)

        # Test BlockMatrix'*E
        matrix6 = matrix[:6, :5]
        result = pre_multiply_block_diagonal_matrix(
            matrix6, matrix_blocks, True)

        exact_result = np.asarray(
            [1,  6,  4, 49, 64, 11, 26, 24, 139, 184, 21, 46, 44, 229, 304,
             31, 66, 64, 319, 424, 41, 86, 84, 409, 544]).reshape(
                 (5, 5), order='F')
        assert np.allclose(exact_result, result)


def get_tensor_product_points(level, var_trans, quad_type):
    abscissa_1d = []
    num_vars = var_trans.num_vars()
    if quad_type == 'CC':
        x, w = clenshaw_curtis_pts_wts_1D(level)
    elif quad_type == 'GH':
        x, w = gauss_hermite_pts_wts_1D(level)
    for dd in range(num_vars):
        abscissa_1d.append(x)
    pts = cartesian_product(abscissa_1d, 1)
    pts = var_trans.map_from_canonical(pts)
    return pts

# do not have test in the name or nose will try to test this function
# and throw and error


def helper_least_factorization(pts, model, var_trans, pce_opts, oli_opts,
                               basis_generator,
                               max_num_pts=None, initial_pts=None,
                               pce_degree=None,
                               preconditioning_function=None,
                               verbose=False,
                               points_non_degenerate=False,
                               exact_mean=None):

    num_vars = pts.shape[0]

    pce = PolynomialChaosExpansion()
    pce.configure(pce_opts)

    oli_solver = LeastInterpolationSolver()
    oli_solver.configure(oli_opts)
    oli_solver.set_pce(pce)

    if preconditioning_function is not None:
        oli_solver.set_preconditioning_function(preconditioning_function)

    oli_solver.set_basis_generator(basis_generator)

    if max_num_pts is None:
        max_num_pts = pts.shape[1]

    if initial_pts is not None:
        # find unique set of points and separate initial pts from pts
        # this allows for cases when
        # (1) pts intersect initial_pts = empty
        # (2) pts intersect initial_pts = initial pts
        # (3) 0 < #(pts intersect initial_pts) < #initial_pts
        pts = remove_common_rows([pts.T, initial_pts.T]).T

    oli_solver.factorize(
        pts, initial_pts,
        num_selected_pts=max_num_pts)

    permuted_pts = oli_solver.get_current_points()

    permuted_vals = model(permuted_pts)
    pce = oli_solver.get_current_interpolant(
        permuted_pts, permuted_vals)

    assert permuted_pts.shape[1] == max_num_pts

    # Ensure pce interpolates the training data
    pce_vals = pce.value(permuted_pts)
    assert np.allclose(permuted_vals, pce_vals)

    # Ensure pce exactly approximates the polynomial test function (model)
    test_pts = generate_independent_random_samples(
        var_trans.variable, num_samples=10)
    test_vals = model(test_pts)
    # print 'p',test_pts.T
    pce_vals = pce.value(test_pts)
    L, U, H = oli_solver.get_current_LUH_factors()
    # print L
    # print U
    # print test_vals
    # print pce_vals
    # print 'coeff',pce.get_coefficients()
    # print oli_solver.selected_basis_indices
    assert np.allclose(test_vals, pce_vals)

    if initial_pts is not None:
        temp = remove_common_rows([permuted_pts.T, initial_pts.T]).T
        assert temp.shape[1] == max_num_pts-initial_pts.shape[1]
        if oli_solver.enforce_ordering_of_initial_points:
            assert np.allclose(
                initial_pts, permuted_pts[:, :initial_pts.shape[0]])
        elif not oli_solver.get_initial_points_degenerate():
            assert allclose_unsorted_matrix_rows(
                initial_pts.T, permuted_pts[:, :initial_pts.shape[1]].T)
        else:
            # make sure that oli tried again to add missing initial
            # points after they were found to be degenerate
            # often adding one new point will remove degeneracy
            assert oli_solver.get_num_initial_points_selected() ==\
                initial_pts.shape[1]
            P = oli_solver.get_current_permutation()
            I = np.where(P < initial_pts.shape[1])[0]
            assert_allclose_unsorted_matrix_cols(
                initial_pts, permuted_pts[:, I])

    basis_generator = oli_solver.get_basis_generator()
    max_degree = oli_solver.get_current_degree()
    basis_cardinality = oli_solver.get_basis_cardinality()
    num_terms = 0
    for degree in range(max_degree):
        __, indices = basis_generator(num_vars, degree)
        num_terms += indices.shape[1]
        assert num_terms == basis_cardinality[degree]

    if points_non_degenerate:
        degree_list = oli_solver.get_points_to_degree_map()
        num_terms = 1
        degree = 0
        num_pts = permuted_pts.shape[1]
        for i in range(num_pts):
            # test assumes non-degeneracy
            if i >= num_terms:
                degree += 1
                indices = PolyIndexVector()
                basis_generator.get_degree_basis_indices(
                    num_vars, degree, indices)
                num_terms += indices.size()
            assert degree_list[i] == degree

    if exact_mean is not None:
        mean = pce.get_coefficients()[0, 0]
        assert np.allclose(mean, exact_mean)


class TestOrthogonalLeastInterpolationFactorization(unittest.TestCase):

    def setUp(self):
        # np.set_printoptions(linewidth=200)
        # np.set_printoptions(precision=5)
        pass

    def test_uniform_2d_canonical_domain(self):
        # ----------------------------------------------------- #
        # x in U[-1,1]^2                                        #
        # no intial pts, no candidate basis no preconditioning, #
        # no pivot weight, no return subset of points           #
        # degenerate points                                     #
        # ----------------------------------------------------- #

        # Set PCE options
        num_vars = 2
        var_trans = define_iid_random_variable_transformation(
            uniform(-1, 2), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}
        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # define target function
        def model(x): return np.asarray([x[0]**2 + x[1]**2 + x[0]*x[1]]).T

        # define points to interpolate
        pts = get_tensor_product_points(2, var_trans, 'CC')
        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts,
            basis_generator, exact_mean=2./3.)

    def test_uniform_2d_user_domain(self):
        # ----------------------------------------------------- #
        # x in U[0,1]^2                                         #
        # no intial pts, no candidate basis no preconditioning, #
        # no pivot weights, no return subset of points          #
        # ----------------------------------------------------- #

        # Set PCE options
        num_vars = 2
        var_trans = define_iid_random_variable_transformation(
            uniform(0, 1), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}

        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # define target function
        def model(x): return np.asarray([x[0]**2 + x[1]**2 + x[0]*x[1]]).T

        # define points to interpolate
        pts = get_tensor_product_points(1, var_trans, 'CC')
        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts,
            basis_generator, exact_mean=11./12.)

    def test_uniform_3d_user_domain(self):
        # ----------------------------------------------------- #
        # x in U[0,1]^3                                         #
        # no intial pts, no candidate basis no preconditioning, #
        # no pivot weights, no return subset of points          #
        # ----------------------------------------------------- #

        # Set PCE options
        num_vars = 3
        var_trans = define_iid_random_variable_transformation(
            uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}

        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # define target function
        def model(x): return np.array(
            [np.sum(x**2, axis=0)+x[0]*x[1]+x[1]*x[2]+x[0]*x[1]*x[2]]).T

        # define points to interpolate
        pts = get_tensor_product_points(2, var_trans, 'CC')
        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts,
            basis_generator, exact_mean=13./8.)

    def test_uniform_2d_subset_of_points(self):
        # ----------------------------------------------------- #
        # x in U[0,1]^2                                         #
        # no intial pts, no candidate basis no preconditioning, #
        # no pivot weights, YES return subset of points         #
        # ----------------------------------------------------- #

        num_vars = 2
        var_trans = define_iid_random_variable_transformation(
            uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}
        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # define target function
        def model(x): return np.asarray([x[0]**2 + x[1]**2 + x[0]*x[1]]).T

        # define points to interpolate
        pts = get_tensor_product_points(1, var_trans, 'CC')
        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts, basis_generator,
            max_num_pts=6, exact_mean=11./12.)

    def test_uniform_2d_initial_and_subset_points(self):
        """
        Interpolate a set of points, by first selecting all initial points
        which are NOT degenerate then adding a subset of the remaining points.

        CHECK: Orthogonal least interpolation produces an interpolant but does
        not approximate the function exactly.

        x in U[0,1]^2
        """

        num_vars = 2
        var_trans = define_iid_random_variable_transformation(
            uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False,
                    'enforce_all_initial_points_used': False,
                    'enforce_ordering_of_initial_points': False}

        def basis_generator(num_vars, degree): return (
            degree+1, compute_tensor_product_level_indices(num_vars, degree))

        # define target function
        def model(x): return np.asarray(
            [0.5*(3*x[0]**2-1) + 0.5*(3*x[1]**2-1) + x[0]*x[1]]).T

        # define points to interpolate
        pts = get_tensor_product_points(2, var_trans, 'CC')
        initial_pts = get_tensor_product_points(1, var_trans, 'CC')
        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts,
            basis_generator, initial_pts=initial_pts, max_num_pts=12)

    def test_uniform_2d_degenerate_initial_and_subset_points(self):
        """
        Interpolate a set of points, by first selecting all initial points
        which are degenerate then adding a subset of the remaining points.

        CHECK: Orthogonal least interpolation produces an interpolant but does
        not approximate the function exactly.

        x in U[0,1]^2
        """

        num_vars = 2
        var_trans = define_iid_random_variable_transformation(
            uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False,
                    'enforce_all_initial_points_used': True,
                    'enforce_ordering_of_initial_points': True}

        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # define target function
        def model(x): return np.asarray(
            [0.5*(3*x[0]**2-1) + 0.5*(3*x[1]**2-1) + x[0]*x[1]]).T

        # define points to interpolate
        pts = get_tensor_product_points(2, var_trans, 'CC')
        initial_pts = get_tensor_product_points(1, var_trans, 'CC')
        self.assertRaises(Exception,
                          helper_least_factorization,
                          pts, model, var_trans, pce_opts, oli_opts,
                          basis_generator,
                          initial_pts=initial_pts, max_num_pts=12,
                          use_preconditioning=1)

    def test_beta_2d_preconditioning(self):
        """
        Interpolate a set of points using preconditioing. First select
        all initial points then adding a subset of the remaining points.

        x in Beta(2,5)[0,1]^2
        """

        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        var_trans = define_iid_random_variable_transformation(
            beta(alpha_stat, beta_stat, -1, 2), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}

        basis_generator = \
            lambda num_vars, degree: (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        # from scipy.special import beta as beta_fn
        # def beta_pdf(x,alpha_poly,beta_poly):
        #     values = (1.-x)**(alpha_poly) * (1.+x)**(beta_poly)
        #     values /= 2.**(beta_poly+alpha_poly+1)*beta_fn(
        #         beta_poly+1,alpha_poly+1)
        #     return values
        # univariate_pdf = partial(beta_pdf,alpha_poly=beta_stat-1,beta_poly=alpha_stat-1)

        univariate_beta_pdf = partial(beta.pdf, a=alpha_stat, b=beta_stat)
        def univariate_pdf(x): return univariate_beta_pdf((x+1.)/2.)/2.
        preconditioning_function = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        # define target function
        def model(x): return np.asarray(
            [(x[0]**2-1) + (x[1]**2-1) + x[0]*x[1]]).T

        # define points to interpolate
        pts = generate_independent_random_samples(var_trans.variable, 12)
        initial_pts = np.array([pts[:, 0]]).T

        helper_least_factorization(
            pts, model, var_trans, pce_opts, oli_opts, basis_generator,
            initial_pts=initial_pts, max_num_pts=12,
            preconditioning_function=preconditioning_function)

    def test_factorization_using_exact_algebra(self):

        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        var_trans = define_iid_random_variable_transformation(
            beta(alpha_stat, beta_stat, -2, 1), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)

        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}

        basis_generator = \
            lambda num_vars, degree: (
                degree+1, compute_hyperbolic_level_indices(
                    num_vars, degree, 1.0))

        oli_solver = LeastInterpolationSolver()
        oli_solver.configure(oli_opts)
        oli_solver.set_pce(pce)
        oli_solver.set_basis_generator(basis_generator)

        # Define 4 candidate points so no pivoting is necessary
        candidate_pts = np.array([[-1., 1./np.sqrt(2.), -1./np.sqrt(2.), 0.],
                                  [-1., -1./np.sqrt(2.), 0., 0.]])

        U = np.zeros((4, 4))

        factor_history = []

        # Build vandermonde matrix for all degrees ahead of time
        degree = 2
        indices = compute_hyperbolic_indices(num_vars, degree, 1.)
        pce.set_indices(indices)
        V = pce.basis_matrix(candidate_pts)

        ##--------------------- ##
        ## S=1                  ##
        ##--------------------- ##

        # print 'V\n',V

        # print '################################'
        U1 = np.array([[V[0, 1], V[0, 2]],
                    [V[1, 1]-V[0, 1], V[1, 2]-V[0, 2]],
                    [V[2, 1]-V[0, 1], V[2, 2]-V[0, 2]],
                    [V[3, 1]-V[0, 1], V[3, 2]-V[0, 2]]])

        norms = [np.sqrt((V[1, 1]-V[0, 1])**2+(V[1, 2]-V[0, 2])**2),
                 np.sqrt((V[2, 1]-V[0, 1])**2+(V[2, 2]-V[0, 2])**2),
                 np.sqrt((V[3, 1]-V[0, 1])**2+(V[3, 2]-V[0, 2])**2)]
        U1[1, :] /= norms[0]
        # print 'U1\n',U1

        # print 'norms\n', norms

        magic_row = np.array(
            [[(V[1, 1]-V[0, 1])/norms[0], (V[1, 2]-V[0, 2])/norms[0]]])
        # print 'magic_row\n',magic_row

        inner_products = np.array([(V[1, 1]-V[0, 1])*(V[1, 1]-V[0, 1])/norms[0] +
                                (V[1, 2]-V[0, 2])*(V[1, 2]-V[0, 2])/norms[0],
                                (V[2, 1]-V[0, 1])*(V[1, 1]-V[0, 1])/norms[0] +
                                (V[2, 2]-V[0, 2])*(V[1, 2]-V[0, 2])/norms[0],
                                (V[3, 1]-V[0, 1])*(V[1, 1]-V[0, 1])/norms[0] +
                                (V[3, 2]-V[0, 2])*(V[1, 2]-V[0, 2])/norms[0]])
        # print 'inner_products\n', inner_products

        v1 = inner_products
        L = np.array([[1, 0, 0, 0], [0, v1[0], v1[1], v1[2]]]).T
        # print 'L\n',L

        Z = np.array([[V[0, 1]*(V[1, 1]-V[0, 1])/norms[0] +
                    V[0, 2]*(V[1, 2]-V[0, 2])/norms[0]]])
        # print 'Z\n',Z

        U = np.array([[1, Z[0, 0]], [0, 1]])
        # print 'U\n',U

        factor_history.append((L, U))

        ##--------------------- ##
        ## S=2                  ##
        ##--------------------- ##

        # print '################################'
        U2 = np.array([[V[0, 1], V[0, 2]],
                       [(V[1, 1]-V[0, 1])/L[1, 1], (V[1, 2]-V[0, 2])/L[1, 1]],
                       [(V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1],
                        (V[2, 2]-V[0, 2])-L[2, 1]*(V[1, 2]-V[0, 2])/L[1, 1]],
                       [(V[3, 1]-V[0, 1])-L[3, 1]*(V[1, 1]-V[0, 1])/L[1, 1],
                        (V[3, 2]-V[0, 2])-L[3, 1]*(V[1, 2]-V[0, 2])/L[1, 1]]])

        # print 'U2\n',U2

        norms = [
            np.sqrt(((V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1])**2 +
                    ((V[2, 2]-V[0, 2])-L[2, 1]*(V[1, 2]-V[0, 2])/L[1, 1])**2),
            np.sqrt(((V[3, 1]-V[0, 1])-L[3, 1]*(V[1, 1]-V[0, 1])/L[1, 1])**2 +
                    ((V[3, 2]-V[0, 2])-L[3, 1]*(V[1, 2]-V[0, 2])/L[1, 1])**2)]
        U2[2, :] /= norms[0]
        # print 'U2\n',U2

        # print 'norms\n', norms

        magic_row = np.array(
            [(V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1],
             (V[2, 2]-V[0, 2])-L[2, 1]*(V[1, 2]-V[0, 2])/L[1, 1]])/norms[0]
        # print 'magic_row', magic_row

        inner_products = [norms[0], ((V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1])*((V[3, 1]-V[0, 1])-L[3, 1]*(V[1, 1]-V[0, 1])/L[1, 1])/norms[0]+(
            (V[2, 2]-V[0, 2])-L[2, 1]*(V[1, 2]-V[0, 2])/L[1, 1])*((V[3, 2]-V[0, 2])-L[3, 1]*(V[1, 2]-V[0, 2])/L[1, 1])/norms[0]]
        # print 'inner_products',inner_products

        v2 = inner_products
        L = np.array(
            [[1, 0, 0, 0], [0, v1[0], v1[1], v1[2]], [0, 0, v2[0], v2[1]]]).T
        # print 'L\n',L

        Z = [V[0, 1]/norms[0]*((V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1]) +
             V[0, 2]/norms[0]*((V[2, 2]-V[0, 2])-L[2, 1] *
                               (V[1, 2]-V[0, 2])/L[1, 1]),
             (V[1, 1]-V[0, 1])/(L[1, 1]*norms[0])*((V[2, 1]-V[0, 1])-L[2, 1]*(V[1, 1]-V[0, 1])/L[1, 1]) +
             (V[1, 2]-V[0, 2])/(L[1, 1]*norms[0])*((V[2, 2]-V[0, 2])-L[2, 1]*(V[1, 2]-V[0, 2])/L[1, 1])]
        # print 'Z\n',Z

        U_prev = U.copy()
        U = np.zeros((3, 3))
        U[:2, :2] = U_prev
        U[:2, 2] = Z
        U[2, 2] = 1
        # print 'U\n', U

        factor_history.append((L, U))

        ##--------------------- ##
        ## S=3                  ##
        ##--------------------- ##

        # print '################################'
        U3 = np.array([[V[0, 3], V[0, 4], V[0, 5]],
                    [(V[1, 3]-V[0, 3])/L[1, 1], (V[1, 4]-V[0, 4]) /
                     L[1, 1], (V[1, 5]-V[0, 5])/L[1, 1]],
                    [((V[2, 3]-V[0, 3])-L[2, 1]*(V[1, 3]-V[0, 3])/L[1, 1])/L[2, 2], ((V[2, 4]-V[0, 4])-L[2, 1] *
                                                                                     (V[1, 4]-V[0, 4])/L[1, 1])/L[2, 2], ((V[2, 5]-V[0, 5])-L[2, 1]*(V[1, 5]-V[0, 5])/L[1, 1])/L[2, 2]],
                    [(V[3, 3]-V[0, 3])-L[3, 1]*(V[1, 3]-V[0, 3])/L[1, 1]-L[3, 2]/L[2, 2]*(V[2, 3]-V[0, 3]-L[2, 1]/L[1, 1]*(V[1, 3]-V[0, 3])), (V[3, 4]-V[0, 4])-L[3, 1]*(V[1, 4]-V[0, 4])/L[1, 1]-L[3, 2]/L[2, 2]*(V[2, 4]-V[0, 4]-L[2, 1]/L[1, 1]*(V[1, 4]-V[0, 4])), (V[3, 5]-V[0, 5])-L[3, 1]*(V[1, 5]-V[0, 5])/L[1, 1]-L[3, 2]/L[2, 2]*(V[2, 5]-V[0, 5]-L[2, 1]/L[1, 1]*(V[1, 5]-V[0, 5]))]])

        norms = [norm(U3[3, :])]

        U3[3, :] /= norms[0]
        # print 'U3\n', U3

        # print 'norms\n', norms

        magic_row = np.array([U3[3, :]])
        # print 'magic_row', magic_row

        inner_products = [norms[0]]
        # print 'inner_products\n', inner_products

        L_prev = L.copy()
        L = np.zeros((4, 4))
        L[:, :3] = L_prev
        L[3, 3] = inner_products[0]
        # print 'L\n', L

        Z = np.dot(U3[:3, :3], magic_row.T)
        # print 'Z\n',Z

        U_prev = U.copy()
        U = np.zeros((4, 4))
        U[:3, :3] = U_prev
        U[:3, 3] = Z.squeeze()
        U[3, 3] = 1
        # print 'U\n',U
        # assert False

        factor_history.append((L, U))

        candidate_pts = np.array([[-1., 1./np.sqrt(2.), -1./np.sqrt(2.), 0.],
                                  [-1., -1./np.sqrt(2.), 0., 0.]])

        # define target function
        def model(x): return np.asarray([x[0]**2 + x[1]**2 + x[0]*x[1]]).T

        # num_starting_pts = 5
        num_starting_pts = 1
        initial_pts = None
        oli_solver.factorize(
            candidate_pts, initial_pts, num_selected_pts=num_starting_pts)

        L, U, H = oli_solver.get_current_LUH_factors()
        # print 'L\n',L
        # print 'U\n',U
        # print 'H\n',H
        it = 0
        np.allclose(L[:1, :1], factor_history[it][0])
        np.allclose(U[:1, :1], factor_history[it][0])

        current_pts = oli_solver.get_current_points()
        current_vals = model(current_pts)

        num_pts = current_pts.shape[1]
        num_pts_prev = current_pts.shape[1]
        max_num_pts = candidate_pts.shape[1]
        finalize = False
        while not finalize:
            if ((num_pts == max_num_pts-1) or
                    (num_pts == candidate_pts.shape[1])):
                finalize = True

            oli_solver.update_factorization(1)

            L, U, H = oli_solver.get_current_LUH_factors()
            # print '###########'
            # print 'L\n',L
            # print 'U\n',U
            # print 'H\n',H
            np.allclose(L,
                        factor_history[it][0][:L.shape[0], :L.shape[1]])
            np.allclose(U,
                        factor_history[it][1][:U.shape[0], :U.shape[1]])
            it += 1

            num_pts_prev = num_pts
            num_pts = oli_solver.num_points_added()
            if (num_pts > num_pts_prev):
                # print 'number of points', num_pts
                current_pt = oli_solver.get_last_point_added()
                current_val = model(current_pt)
                current_pts = np.hstack(
                    (current_pts, current_pt.reshape(current_pt.shape[0], 1)))
                current_vals = np.vstack((current_vals, current_val))
                pce = oli_solver.get_current_interpolant(
                    current_pts, current_vals)
                current_pce_vals = pce.value(current_pts)
                assert np.allclose(current_pce_vals, current_vals)

    def test_least_interpolation_lu_equivalence_in_1d(self):
        num_vars = 1
        alpha_stat = 2
        beta_stat = 5
        max_num_pts = 100

        var_trans = define_iid_random_variable_transformation(
            beta(alpha_stat, beta_stat), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        # Set oli options
        oli_opts = {'verbosity': 0,
                    'assume_non_degeneracy': False}

        def basis_generator(num_vars, degree):
            return (degree+1, compute_hyperbolic_level_indices(
                num_vars, degree, 1.0))

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)

        oli_solver = LeastInterpolationSolver()
        oli_solver.configure(oli_opts)
        oli_solver.set_pce(pce)

        # univariate_beta_pdf = partial(beta.pdf,a=alpha_stat,b=beta_stat)
        # univariate_pdf = lambda x: univariate_beta_pdf(x)
        # preconditioning_function = partial(
        #     tensor_product_pdf,univariate_pdfs=univariate_pdf)
        max_degree = get_total_degree(num_vars, max_num_pts)
        indices = compute_hyperbolic_indices(num_vars, max_degree, 1.)
        pce.set_indices(indices)

        def preconditioning_function(samples): return 1./christoffel_function(
            samples, pce.basis_matrix)

        oli_solver.set_preconditioning_function(preconditioning_function)
        oli_solver.set_basis_generator(basis_generator)

        initial_pts = None
        candidate_samples = np.linspace(0., 1., 1000)[np.newaxis, :]

        oli_solver.factorize(
            candidate_samples, initial_pts,
            num_selected_pts=max_num_pts)

        oli_samples = oli_solver.get_current_points()

        pce.set_indices(oli_solver.selected_basis_indices)
        basis_matrix = pce.basis_matrix(candidate_samples)
        weights = np.sqrt(preconditioning_function(candidate_samples))
        basis_matrix = np.dot(np.diag(weights), basis_matrix)
        L, U, p = truncated_pivoted_lu_factorization(
            basis_matrix, max_num_pts)
        assert p.shape[0] == max_num_pts
        lu_samples = candidate_samples[:, p]

        assert np.allclose(lu_samples, oli_samples)

        L1, U1, H1 = oli_solver.get_current_LUH_factors()

        true_permuted_matrix = (pce.basis_matrix(lu_samples).T*weights[p]).T
        assert np.allclose(np.dot(L, U), true_permuted_matrix)
        assert np.allclose(np.dot(L1, np.dot(U1, H1)), true_permuted_matrix)


if __name__ == "__main__":
    block_matrix_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBlockDiagonalOperations)
    unittest.TextTestRunner(verbosity=2).run(block_matrix_test_suite)
    oli_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestOrthogonalLeastInterpolationFactorization)
    unittest.TextTestRunner(verbosity=2).run(oli_test_suite)
