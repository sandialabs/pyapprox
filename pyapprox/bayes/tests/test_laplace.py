import unittest
from scipy.linalg import cholesky
import numpy as np
import os
import glob
import tempfile

from pyapprox.variables.density import NormalDensity, ObsDataDensity
from pyapprox.util.linalg import get_low_rank_matrix
from pyapprox.util.randomized_svd import (
    randomized_svd, MatVecOperator, adjust_sign_svd, svd_using_orthogonal_basis
)
from pyapprox.variables.tests.test_density import check_gradient
from pyapprox.variables.gaussian import MultivariateGaussian,\
    CholeskySqrtCovarianceOperator, CovarianceOperator, get_operator_diagonal
from pyapprox.interface.wrappers import evaluate_1darray_function_on_2d_array
from pyapprox.bayes.laplace import (
    get_laplace_covariance_sqrt_operator,
    get_pointwise_laplace_variance_using_prior_variance,
    sample_from_laplace_posterior,
    laplace_posterior_approximation_for_linear_models,
    MisfitHessianVecOperator, PriorConditionedHessianMatVecOperator,
    directional_derivatives, find_map_point,
    push_forward_gaussian_though_linear_model,
    generate_and_save_laplace_posterior, LaplaceSqrtMatVecOperator,
    generate_and_save_pointwise_variance,
    compute_posterior_mean_covar_optimal_for_prediction
)
from pyapprox.util.visualization import plot_multiple_2d_gaussian_slices


class QuadraticMisfitModel(object):
    def __init__(self, num_vars, rank, num_qoi,
                 obs=None, noise_covariance=None, Amatrix=None):
        self.num_vars = num_vars
        self.rank = rank
        self.num_qoi = num_qoi
        if Amatrix is None:
            self.Amatrix = get_low_rank_matrix(num_qoi, num_vars, rank)
        else:
            self.Amatrix = Amatrix

        if obs is None:
            self.obs = np.zeros(num_qoi)
        else:
            self.obs = obs
        if noise_covariance is None:
            self.noise_covariance = np.eye(num_qoi)
        else:
            self.noise_covariance = noise_covariance

        self.noise_covariance_inv = np.linalg.inv(self.noise_covariance)

    def value(self, sample):
        assert sample.ndim == 1
        residual = np.dot(self.Amatrix, sample)-self.obs
        return np.asarray(
            [0.5*np.dot(residual.T, np.dot(
                self.noise_covariance_inv, residual))])

    def gradient(self, sample):
        assert sample.ndim == 1
        grad = np.dot(self.Amatrix.T,
                      np.dot(self.noise_covariance_inv,
                             np.dot(self.Amatrix, sample)-self.obs))
        return grad

    def gradient_set(self, samples):
        assert samples.ndim == 2
        num_vars, num_samples = samples.shape
        gradients = np.empty((num_vars, num_samples), dtype=float)
        for i in range(num_samples):
            gradients[:, i] = self.gradient(samples[:, i])
        return gradients

    def hessian(self, sample):
        assert sample.ndim == 1 or sample.shape[1] == 1
        return np.dot(
            np.dot(self.Amatrix.T, self.noise_covariance_inv), self.Amatrix)

    def __call__(self, samples, opts=dict()):
        eval_type = opts.get('eval_type', 'value')
        if eval_type == 'value':
            return evaluate_1darray_function_on_2d_array(
                self.value, samples, opts)
        elif eval_type == 'value_grad':
            vals = evaluate_1darray_function_on_2d_array(
                self.value, samples, opts)
            return np.hstack((vals, self.gradient_set(samples).T))
        elif eval_type == 'grad':
            return self.gradient_set(samples).T
        else:
            raise Exception('%s is not a valid eval_type' % eval_type)


class LogUnormalizedPosterior(object):

    def __init__(self, misfit, misfit_gradient, prior_pdf, prior_log_pdf,
                 prior_log_pdf_gradient):
        """
        Initialize the object.

        Parameters
        ----------
        """
        self.misfit = misfit
        self.misfit_gradient = misfit_gradient
        self.prior_pdf = prior_pdf
        self.prior_log_pdf = prior_log_pdf
        self.prior_log_pdf_gradient = prior_log_pdf_gradient

    def gradient(self, samples):
        """
        Evaluate the gradient of the logarithm of the unnormalized posterior
           likelihood(x)*posterior(x)
        at a sample x

        Parameters
        ----------
        samples : (num_vars,num_samples) vector
            The location at which to evalute the unnormalized posterior

        Returns
        -------
        val : (1x1) vector
            The logarithm of the unnormalized posterior
        """
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        grad = -self.misfit_gradient(samples) + \
            self.prior_log_pdf_gradient(samples)
        return grad

    def __call__(self, samples, opts=dict()):
        """
        Evaluate the logarithm of the unnormalized posterior
           likelihood(x)*posterior(x)
        at samples x

        Parameters
        ----------
        sampels : np.ndarray (num_vars, num_samples)
            The samples at which to evalute the unnormalized posterior

        Returns
        -------
        values : np.ndarray (num_samples,1)
            The logarithm of the unnormalized posterior
        """
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        eval_type = opts.get('eval_type', 'value')
        if eval_type == 'value':
            values = -self.misfit(samples)+self.prior_log_pdf(samples)
            assert values.ndim == 2

        elif eval_type == 'grad':
            values = self.gradient(samples).T

        elif eval_type == 'value_grad':
            values = -self.misfit(samples)+self.prior.log_pdf(samples)
            grad = self.gradient(samples)
            values = np.hstack((values, grad))
        else:
            raise Exception()

        return values


def assert_ndarray_allclose(matrix1, matrix2, atol=1e-8, rtol=1e-5, msg=None):
    """
    A more useful function for testing equivalence of numpy arrays.
    Print norms used by np.allclose function to determine equivalence.

    Matrix1 is considered the truth
    """
    if not np.allclose(matrix1, matrix2, atol=atol, rtol=rtol):
        if msg is not None:
            print(msg)
        diff = np.absolute(matrix1-matrix2)
        abs_error = diff.max()
        rel_error = (diff/np.absolute(matrix1)).max()
        print('abs error:', abs_error)
        print('rel error:', rel_error)
        print('atol:', atol)
        print('rtol:', rtol)
        print('matrix1 shape', matrix1.shape)
        print('matrix2 shape', matrix2.shape)
        assert False, 'matrices are not equivalent'


def setup_quadratic_misfit_problem(prior, rank, noise_sigma2=1):
    # Define observations
    num_qoi = 2*rank
    # assert num_qoi>=rank

    noise_covariance = np.eye(num_qoi)*noise_sigma2
    noise_covariance_inv = np.linalg.inv(noise_covariance)
    # In high dimensions computing cholesky factor is too expensive.
    # That is why we use PDE based operator
    noise_covariance_chol_factor = np.linalg.cholesky(noise_covariance)
    truth_sample = prior.generate_samples(1)[:, 0]
    num_vars = truth_sample.shape[0]
    Amatrix = get_low_rank_matrix(num_qoi, num_vars, rank)
    noise = np.dot(noise_covariance_chol_factor,
                   np.random.normal(0., noise_sigma2, num_qoi))
    obs = np.dot(Amatrix, truth_sample)+noise

    # Define mistit model
    misfit_model = QuadraticMisfitModel(num_vars, rank, num_qoi, Amatrix)

    return misfit_model, noise_covariance_inv, obs


def posterior_covariance_helper(prior, rank, comparison_tol,
                                test_sampling=False, plot=False):
    """
    Test that the Laplace posterior approximation can be obtained using
    the action of the sqrt prior covariance computed using a PDE solve

    Parameters
    ----------
    prior : MultivariateGaussian object
        The model which must be able to compute the action of the sqrt of the
        prior covariance (and its tranpose) on a set of vectors

    rank : integer
        The rank of the linear model used to generate the observations

    comparision_tol :
        tolerances for each of the internal comparisons. This allows different
        accuracy for PDE based operators
    """

    # Define prior sqrt covariance and covariance operators
    L_op = prior.sqrt_covariance_operator

    # Extract prior information required for computing exact posterior
    # mean and covariance
    num_vars = prior.num_vars()
    prior_mean = np.zeros((num_vars), float)
    L = L_op(np.eye(num_vars), False)
    L_T = L_op(np.eye(num_vars), True)
    assert_ndarray_allclose(L.T, L_T, rtol=comparison_tol, atol=0,
                            msg='Comparing prior sqrt and transpose')

    prior_covariance = np.dot(L, L_T)
    prior_pointwise_variance = prior.pointwise_variance()
    assert_ndarray_allclose(
        np.diag(prior_covariance), prior_pointwise_variance, rtol=1e-14,
        atol=0, msg='Comparing prior pointwise variance')

    misfit_model, noise_covariance_inv, obs = setup_quadratic_misfit_problem(
        prior, rank, noise_sigma2=1)

    # Get analytical mean and covariance
    prior_hessian = np.linalg.inv(prior_covariance)
    exact_laplace_mean, exact_laplace_covariance = \
        laplace_posterior_approximation_for_linear_models(
            misfit_model.Amatrix, prior.mean, prior_hessian,
            noise_covariance_inv, obs)

    # Define prior conditioned misfit operator
    sample = np.zeros(num_vars)
    misfit_hessian_operator = MisfitHessianVecOperator(
        misfit_model, sample, fd_eps=None)
    LHL_op = PriorConditionedHessianMatVecOperator(
        L_op, misfit_hessian_operator)

    # For testing purposes build entire L*H*L matrix using operator
    # and compare to result based upon explicit matrix mutiplication
    LHL_op = LHL_op.apply(np.eye(num_vars), transpose=False)
    H = misfit_model.hessian(sample)
    assert np.allclose(H, np.dot(np.dot(
        misfit_model.Amatrix.T, noise_covariance_inv), misfit_model.Amatrix))
    LHL_mat = np.dot(L_T, np.dot(H, L))
    assert_ndarray_allclose(
        LHL_mat, LHL_op, rtol=comparison_tol,
        msg='Comparing prior matrix and operator based LHL')

    # Test singular values obtained by randomized svd using operator
    # are the same as those obtained using singular decomposition
    Utrue, Strue, Vtrue = np.linalg.svd(LHL_mat)
    Utrue, Vtrue = adjust_sign_svd(Utrue, Vtrue)
    standard_svd_opts = {
        'num_singular_values': rank, 'num_extra_samples': 10}
    svd_opts = {'single_pass': True, 'standard_opts': standard_svd_opts}
    L_post_op = get_laplace_covariance_sqrt_operator(
        L_op, misfit_hessian_operator, svd_opts, weights=None,
        min_singular_value=0.0)
    # print np.max((Strue[:rank]-L_post_op.e_r)/Strue[0])
    max_error = np.max(Strue[:rank]-L_post_op.e_r)
    assert max_error/Strue[0] < comparison_tol, max_error/Strue[0]
    assert_ndarray_allclose(Vtrue.T[:, :rank], L_post_op.V_r, rtol=1e-6,
                            msg='Comparing eigenvectors')

    L_post_op.V_r = Vtrue.T[:, :rank]

    # Test posterior sqrt covariance operator transpose is the same as
    # explicit matrix transpose of matrix obtained by prior sqrt
    # covariance operator
    L_post = L_post_op.apply(np.eye(num_vars), transpose=False)
    L_post_T = L_post_op.apply(np.eye(num_vars), transpose=True)
    assert_ndarray_allclose(L_post.T, L_post_T, rtol=comparison_tol,
                            msg='Comparing posterior sqrt and transpose')

    # Test posterior covariance operator produced matrix is the same
    # as the exact posterior covariance obtained using analytical formula
    if rank == num_vars:
        # this test only makes sense if entire set of directions is found
        # if low rank approx is used then this will ofcourse induce errors
        post_covariance = np.dot(L_post, L_post_T)
        assert_ndarray_allclose(
            exact_laplace_covariance, post_covariance, rtol=comparison_tol,
            atol=0.,
            msg='Comparing matrix and operator based posterior covariance')

    # Test pointwise covariance of posterior
    post_pointwise_variance, prior_pointwise_variance =\
        get_pointwise_laplace_variance_using_prior_variance(
            prior, L_post_op, prior_pointwise_variance)
    assert_ndarray_allclose(
        np.diag(exact_laplace_covariance), post_pointwise_variance,
        rtol=comparison_tol, atol=0., msg='Comparing pointwise variance')

    if not test_sampling:
        return
    num_samples = int(2e5)
    posterior_samples = sample_from_laplace_posterior(
        exact_laplace_mean, L_post_op, num_vars, num_samples, weights=None)
    assert_ndarray_allclose(
        exact_laplace_covariance, np.cov(posterior_samples),
        atol=1e-2*exact_laplace_covariance.max(), rtol=0.,
        msg='Comparing posterior samples covariance')
    assert_ndarray_allclose(
        exact_laplace_mean.squeeze(),
        np.mean(posterior_samples, axis=1), atol=2e-2, rtol=0.,
        msg='Comparing posterior samples mean')

    if plot:
        # plot marginals of posterior using orginal ordering
        texfilename = 'slices.tex'
        plot_multiple_2d_gaussian_slices(
            exact_laplace_mean[:10], np.diag(exact_laplace_covariance)[:10],
            texfilename, reference_gaussian_data=(0., 1.), show=False)

        # plot marginals of posterior in rotated coordinates
        # from most to least important.
        # The following is not feasiable in practice as we cannot compute
        # entire covariance matrix in full space. But we have
        # C_r = V_r*L*V_r*D*V_r.T*L.T*V_r.T
        # is we compute matrix products from right to left we only have to
        # compute at most (d x r) matrices. And if only want first 20 say
        # variances then can apply C_r to vectors e_i i=1,...,20
        # then we need at most (dx20 matrices)
        texfilename = 'rotated-slices.tex'
        V_r = L_post_op.V_r
        plot_multiple_2d_gaussian_slices(
            np.dot(V_r.T, exact_laplace_mean[:10]),
            np.diag(np.dot(V_r.T, np.dot(exact_laplace_covariance, V_r)))[:10],
            texfilename, reference_gaussian_data=(0., 1.), show=True)


class TestLaplace(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    @unittest.skip(reason="only shows how to plot")
    def test_plot_multiple_2d_gaussian_slices(self):

        mean = np.array([0, 1, -1])
        covariance = np.diag(np.array([1, 0.5, 0.025]))

        texfilename = 'slices.tex'
        plot_multiple_2d_gaussian_slices(
            mean[:10], np.diag(covariance)[:10], texfilename,
            reference_gaussian_data=(0., 1.), show=False)
        filenames = glob.glob(texfilename[:-4]+'*')
        for filename in filenames:
            os.remove(filename)

    def test_operator_diagonal(self):
        num_vars = 4
        eval_concurrency = 2
        randn = np.random.normal(0., 1., (num_vars, num_vars))
        prior_covariance = np.dot(randn.T, randn)
        sqrt_covar_op = CholeskySqrtCovarianceOperator(
            prior_covariance, eval_concurrency)
        covariance_operator = CovarianceOperator(sqrt_covar_op)
        diagonal = get_operator_diagonal(
            covariance_operator, num_vars, eval_concurrency, transpose=None)
        assert np.allclose(diagonal, np.diag(prior_covariance))

    def test_posterior_dense_matrix_covariance_operator(self):
        num_vars = 121
        rank = 10
        eval_concurrency = 20
        #randn = np.random.normal(0.,1.,(num_vars,num_vars))
        #prior_covariance = np.dot(randn.T,randn)
        prior_covariance = np.eye(num_vars)
        prior_sqrt_covariance_op = CholeskySqrtCovarianceOperator(
            prior_covariance, eval_concurrency)
        prior = MultivariateGaussian(prior_sqrt_covariance_op)
        comparison_tol = 6e-7
        posterior_covariance_helper(
            prior, rank, comparison_tol, test_sampling=True)

    def test_log_unnormalized_posterior(self):
        num_dims = 4
        rank = 3
        num_qoi = 3
        obs = np.random.normal(0., 1., (num_qoi))
        prior_mean = np.zeros((num_dims), float)
        prior_covariance = np.eye(num_dims)*0.25
        # prior_covariance_chol_factor = np.linalg.cholesky(prior_covariance)
        noise_covariance = np.eye(num_qoi)*0.1
        # noise_covariance_inv = np.linalg.inv(noise_covariance)
        misfit_model = QuadraticMisfitModel(
            num_dims, rank, num_qoi, obs, noise_covariance=noise_covariance)

        prior_density = NormalDensity(prior_mean, covariance=prior_covariance)
        objective = LogUnormalizedPosterior(
            misfit_model, misfit_model.gradient_set, prior_density.pdf,
            prior_density.log_pdf, prior_density.log_pdf_gradient)

        samples = prior_density.generate_samples(2)
        exact_log_unnormalized_posterior_vals = np.log(
            np.exp(-misfit_model(samples)) *
            prior_density.pdf(samples))
        log_unnormalized_posterior_vals = objective(samples)
        assert np.allclose(exact_log_unnormalized_posterior_vals,
                           log_unnormalized_posterior_vals)

        exact_log_unnormalized_posterior_grads = \
            -misfit_model.gradient_set(samples) +\
            prior_density.log_pdf_gradient(samples)
        log_unnormalized_posterior_grads = objective(
            samples, {'eval_type': 'grad'})
        assert np.allclose(exact_log_unnormalized_posterior_grads.T,
                           log_unnormalized_posterior_grads)

    def test_get_map_point(self):
        num_dims = 4
        rank = 3
        num_qoi = 3
        obs = np.random.normal(0., 1., (num_qoi))
        prior_mean = np.zeros((num_dims), float)
        prior_covariance = np.eye(num_dims)*0.25
        prior_covariance_chol_factor = np.linalg.cholesky(prior_covariance)
        noise_covariance = np.eye(num_qoi)*0.1
        noise_covariance_inv = np.linalg.inv(noise_covariance)
        misfit_model = QuadraticMisfitModel(
            num_dims, rank, num_qoi, obs, noise_covariance=noise_covariance)
        # exact map point should be mean of Gaussian posterior
        prior_hessian = np.linalg.inv(prior_covariance)
        exact_map_point = \
            laplace_posterior_approximation_for_linear_models(
                misfit_model.Amatrix, prior_mean, prior_hessian,
                noise_covariance_inv, obs)[0]

        prior_density = NormalDensity(prior_mean, covariance=prior_covariance)
        objective = LogUnormalizedPosterior(
            misfit_model, misfit_model.gradient_set, prior_density.pdf,
            prior_density.log_pdf, prior_density.log_pdf_gradient)
        initial_point = prior_mean
        map_point, obj_min = find_map_point(objective, initial_point)

        assert np.allclose(exact_map_point.squeeze(), map_point)

        assert np.allclose(
            objective.gradient(map_point), objective.gradient(exact_map_point))
        assert np.allclose(objective.gradient(map_point), np.zeros(num_dims))

    def test_push_forward_gaussian_though_linear_model(self):
        num_qoi = 1
        num_dims = 2
        A = np.random.normal(0., 1., (num_qoi, num_dims))
        b = np.random.normal(0., 1., (num_qoi))

        mean = np.ones((num_dims), float)
        covariance = 0.1*np.eye(num_dims)
        covariance_chol_factor = cholesky(covariance)

        push_forward_mean, push_forward_covariance =\
            push_forward_gaussian_though_linear_model(A, b, mean, covariance)

        # Generate samples from original density and push forward through model
        # and approximate density using KDE
        num_samples = 1000000
        samples = np.dot(covariance_chol_factor,
                         np.random.normal(0., 1., (num_dims, num_samples))) +\
            np.tile(mean.reshape(num_dims, 1), num_samples)
        push_forward_samples = np.dot(A, samples)+b

        kde_density = ObsDataDensity(push_forward_samples)
        push_forward_density = NormalDensity(
            push_forward_mean, covariance=push_forward_covariance)

        test_samples = np.linspace(
            push_forward_samples.min(),
            push_forward_samples.max(), 100).reshape(1, 100)
        kde_values = kde_density.pdf(test_samples)
        normal_values = push_forward_density.pdf(test_samples)

        assert np.linalg.norm(kde_values-normal_values[:, 0]) < 4e-2

        # plt = kde_density.plot_density(1000,show=False)
        # import pylab
        # pylab.setp(plt, linewidth=2, color='r')
        # push_forward_density.plot_density(100,show=True)

    def test_quadratic_misfit_model(self):
        num_dims = 10
        rank = 3
        num_qoi = 3
        obs = np.random.normal(0., 1., (num_qoi))
        model = QuadraticMisfitModel(num_dims, rank, num_qoi, obs)

        sample = np.random.normal(0., 1., (num_dims))
        check_gradient(model.value, model.gradient, sample)

    def test_neg_log_posterior(self):
        num_dims = 10
        rank = 3
        num_qoi = 3
        obs = np.random.normal(0., 1., (num_qoi))
        noise_covariance = np.eye(num_qoi)*0.1
        misfit_model = QuadraticMisfitModel(
            num_dims, rank, num_qoi, obs, noise_covariance=noise_covariance)

        # prior_mean = np.ones((num_dims), float)
        # prior_covariance = np.eye(num_dims)*0.25
        # prior_density = NormalDensity(prior_mean, covariance=prior_covariance)
        # objective = LogUnormalizedPosterior(
        #     misfit_model, misfit_model.gradient_set, prior_density.pdf,
        #     prior_density.log_pdf, prior_density.log_pdf_gradient)

        sample = np.random.normal(0., 1., (num_dims))
        check_gradient(misfit_model.value, misfit_model.gradient, sample)

    def test_directional_derivative_using_finite_difference(self):

        num_dims = 10
        rank = 3
        num_qoi = 3
        model = QuadraticMisfitModel(num_dims, rank, num_qoi)
        directions = np.random.normal(0., 1., (num_dims, 2))
        directions /= np.linalg.norm(directions, axis=0)
        # derivatives of function values
        sample = np.random.normal(0., 1., (num_dims, 1))
        opts = {'eval_type': 'value_grad'}
        result = model(sample, opts)[0, :]
        # result is num_samples x num_qoi. There is only one sample so take
        # first row of result above
        value_at_sample = result[0:1]  # must be a vector
        gradient = result[1:]
        #gradient = model.gradient(sample)
        assert np.allclose(
            np.dot(gradient, directions).squeeze(),
            directional_derivatives(
                model, sample, value_at_sample, directions, 1e-7).squeeze())
        # derivatives of gradients
        sample = np.random.normal(0., 1., (num_dims, 1))
        opts = {'eval_type': 'grad'}
        gradient_at_sample = model(sample, opts)[0, :]
        hessian = model.hessian(sample)
        # function passed to directional_derivatives function must return
        # np.ndarray with shape (num_samples,num_vars)
        # each gradient entry is considered a qoi of a function
        def grad_func(x): return model.gradient_set(x).T
        assert np.allclose(
            np.dot(hessian, directions).squeeze(),
            directional_derivatives(
                grad_func, sample, gradient_at_sample, directions, 1e-7,
                use_central_finite_difference=False).T.squeeze())

        # If model has a __call__ which takes options to return values,
        # grads, hessians or combinations. It should also have
        #  value(), gradient(), and hessian() functions which invoke __call__
        # and reshape to have meaningful  shape, e.g. hessians as list of hessians
        # gradients (num_dims x num_samples) array instead of
        # (num_samples x num_dims returned buy __call__

    def test_laplace_posterior_hessian_vec_operator(self):
        num_dims = 10
        rank = 3
        num_qoi = 3
        model = QuadraticMisfitModel(num_dims, rank, num_qoi)
        map_point = np.zeros(num_dims)
        # map_point_misfit_gradient = model.gradient(map_point)
        operator = MisfitHessianVecOperator(
            model, map_point)
        vectors = np.random.normal(0., 1., (num_dims, 2))
        hess_vec_prods = operator.apply(vectors)
        true_hess_vec_prods = np.dot(model.hessian(map_point), vectors)
        assert np.allclose(true_hess_vec_prods, hess_vec_prods)

    def test_hessian_vector_multiply_operator_with_randomized_svd(self):
        num_dims = 100
        rank = 21
        num_qoi = 30
        concurrency = 10

        model = QuadraticMisfitModel(num_dims, rank, num_qoi)

        # --------------------------------- #
        # compute with exact misfit Hessian #
        # --------------------------------- #
        np.random.seed(2)
        Amatrix = model.Amatrix
        operator = MatVecOperator(np.dot(Amatrix.T, Amatrix))
        adaptive_svd_opts = {'tolerance': 1e-8, 'num_extra_samples': 10,
                             'max_num_iter_error_increase': 20,
                             'verbosity': 0, 'max_num_samples': 100,
                             'concurrency': concurrency}
        svd_opts = {'adaptive_opts': adaptive_svd_opts}
        U, S, V = randomized_svd(operator, svd_opts)
        Utrue, Strue, Vtrue = np.linalg.svd(
            np.dot(Amatrix.T, Amatrix), full_matrices=False)
        Utrue, Vtrue = adjust_sign_svd(Utrue, Vtrue)
        J = np.where(Strue > 1e-9)[0]
        assert J.shape[0] == rank
        assert np.allclose(Strue[J], S[J])
        assert np.allclose(Utrue[:, J], U[:, J])
        assert np.allclose(Vtrue[J, :], V[J, :])

        # -------------------------------------------------------------- #
        # compute with finite difference approximation of Hessian action #
        # -------------------------------------------------------------- #
        np.random.seed(2)
        map_point = np.zeros(num_dims)
        operator = MisfitHessianVecOperator(
            model, map_point, fd_eps=1e-7)
        U, S, V = randomized_svd(operator, svd_opts)
        Utrue, Strue, Vtrue = np.linalg.svd(
            np.dot(Amatrix.T, Amatrix), full_matrices=False)
        Utrue, Vtrue = adjust_sign_svd(Utrue, Vtrue)
        J = np.where(Strue > 1e-9)[0]
        assert J.shape[0] == rank
        assert np.allclose(Strue[J], S[J])
        assert np.allclose(Utrue[:, J], U[:, J])
        assert np.allclose(Vtrue[J, :], V[J, :])

    def test_prior_conditioned_misfit_covariance_operator(self):
        num_dims = 3
        rank = 2
        num_qoi = 2

        # define prior
        prior_mean = np.zeros((num_dims), float)
        prior_covariance = np.eye(num_dims)
        prior_hessian = np.eye(num_dims)
        prior_density = NormalDensity(prior_mean, covariance=prior_covariance)

        # define observations
        noise_sigma2 = 0.5
        noise_covariance = np.eye(num_qoi)*noise_sigma2
        noise_covariance_inv = np.linalg.inv(noise_covariance)
        noise_covariance_chol_factor = np.linalg.cholesky(noise_covariance)
        truth_sample = prior_density.generate_samples(1)[:, 0]
        Amatrix = get_low_rank_matrix(num_qoi, num_dims, rank)
        noise = np.dot(
            noise_covariance_chol_factor, np.random.normal(0., 1., num_qoi))
        obs = np.dot(Amatrix, truth_sample)+noise

        # define mistit model
        misfit_model = QuadraticMisfitModel(
            num_dims, rank, num_qoi, obs, noise_covariance, Amatrix)

        # Get analytical mean and covariance
        exact_laplace_mean, exact_laplace_covariance = \
            laplace_posterior_approximation_for_linear_models(
                misfit_model.Amatrix, prior_mean, prior_hessian,
                noise_covariance_inv, obs)

        objective = LogUnormalizedPosterior(
            misfit_model, misfit_model.gradient_set, prior_density.pdf,
            prior_density.log_pdf, prior_density.log_pdf_gradient)
        map_point, obj_min = find_map_point(objective, prior_mean)
        sample = map_point
        assert np.allclose(objective.gradient(sample), np.zeros(num_dims))
        prior_covariance_sqrt_operator = CholeskySqrtCovarianceOperator(
            prior_density.covariance)
        misfit_hessian_operator = MisfitHessianVecOperator(
            misfit_model, sample, fd_eps=1e-7)
        standard_svd_opts = {
            'num_singular_values': rank, 'num_extra_samples': 10}
        svd_opts = {'single_pass': False, 'standard_opts': standard_svd_opts}
        laplace_covariance_sqrt = get_laplace_covariance_sqrt_operator(
            prior_covariance_sqrt_operator, misfit_hessian_operator,
            svd_opts)

        identity = np.eye(num_dims)
        laplace_covariance_chol_factor = laplace_covariance_sqrt.apply(
            identity)
        laplace_covariance = np.dot(
            laplace_covariance_chol_factor, laplace_covariance_chol_factor.T)
        # print laplace_covariance
        # print exact_laplace_covariance
        assert np.allclose(laplace_covariance, exact_laplace_covariance)

    @unittest.skip("I need to implement this test using my new kle class")
    def test_prior_covariance_operator_based_kle(self):
        mesh = np.linspace(0., 1., 10)
        C = correlation_function(mesh[np.newaxis, :], 0.1, 'exp')
        #U,S,V = np.linalg.svd(C)
        #Csqrt = np.dot(U,np.dot(np.diag(np.sqrt(S)),V))
        #assert np.allclose(np.dot(Csqrt,Csqrt),C)
        sqrt_covariance_operator = CholeskySqrtCovarianceOperator(C, 10)
        covariance_operator = CovarianceOperator(
            sqrt_covariance_operator)
        kle = KLE()
        standard_svd_opts = {'num_singular_values': mesh.shape[0],
                             'num_extra_samples': 10}
        svd_opts = {'single_pass': False, 'standard_opts': standard_svd_opts}
        kle.solve_eigenvalue_problem(
            covariance_operator, mesh.shape[0], 0., 'randomized-svd', svd_opts)
        rand_svd_eig_vals = kle.eig_vals.copy()
        kle.solve_eigenvalue_problem(C, mesh.shape[0], 0., 'eig')
        eig_eig_vals = kle.eig_vals.copy()
        assert np.allclose(rand_svd_eig_vals, eig_eig_vals)

    def help_generate_and_save_laplace_posterior(
            self, fault_precentage):
        svd_history_filename = 'svd-history.npz'
        Lpost_op_filename = 'laplace_sqrt_operator.npz'
        prior_variance_filename = 'prior-pointwise-variance.npz'
        posterior_variance_filename = 'posterior-pointwise-variance.npz'
        if os.path.exists(svd_history_filename):
            os.remove(svd_history_filename)
        if os.path.exists(Lpost_op_filename):
            os.remove(Lpost_op_filename)
        if os.path.exists(prior_variance_filename):
            os.remove(prior_variance_filename)
        if os.path.exists(posterior_variance_filename):
            os.remove(posterior_variance_filename)

        num_vars = 121
        rank = 20
        eval_concurrency = 2
        prior_covariance = np.eye(num_vars)
        prior_sqrt_covariance_op = CholeskySqrtCovarianceOperator(
            prior_covariance, eval_concurrency, fault_percentage=2)
        prior = MultivariateGaussian(prior_sqrt_covariance_op)

        misfit_model, noise_covariance_inv, obs = \
            setup_quadratic_misfit_problem(prior, rank, noise_sigma2=1)

        # Get analytical mean and covariance
        prior_hessian = np.linalg.inv(prior_covariance)
        exact_laplace_mean, exact_laplace_covariance = \
            laplace_posterior_approximation_for_linear_models(
                misfit_model.Amatrix, prior.mean, prior_hessian,
                noise_covariance_inv, obs)

        # instead of using optimization to find map point just use exact
        # map
        misfit_model.map_point = lambda: exact_laplace_mean

        num_singular_values = prior.num_vars()
        try:
            L_post_op = generate_and_save_laplace_posterior(
                prior, misfit_model, num_singular_values,
                svd_history_filename=svd_history_filename,
                Lpost_op_filename=Lpost_op_filename,
                num_extra_svd_samples=30, fd_eps=None)
        except:
            recovered_svd_data = np.load('randomized_svd_recovery_data.npz')
            X = recovered_svd_data['X']
            Y = recovered_svd_data['Y']
            I = np.where(np.all(np.isfinite(Y), axis=0))[0]
            X = X[:, I]
            Y = Y[:, I]
            Q, R = np.linalg.qr(Y)
            U, S, V = svd_using_orthogonal_basis(None, Q, X, Y, True)
            U = U[:, :num_singular_values]
            S = S[:num_singular_values]
            L_post_op = LaplaceSqrtMatVecOperator(
                prior_sqrt_covariance_op, e_r=S, V_r=U)
            L_post_op.save(Lpost_op_filename)

        L_post_op_from_file = LaplaceSqrtMatVecOperator(
            prior_sqrt_covariance_op, filename=Lpost_op_filename)

        # turn off faults in prior to allow easy comparison
        prior_sqrt_covariance_op.fault_percentage = 0
        L_post = L_post_op_from_file.apply(np.eye(num_vars), transpose=False)
        L_post_T = L_post_op_from_file.apply(np.eye(num_vars), transpose=True)
        assert_ndarray_allclose(L_post.T, L_post_T, rtol=1e-12)

        # Test posterior covariance operator produced matrix is the same
        # as the exact posterior covariance obtained using analytical formula
        post_covariance = np.dot(L_post, L_post_T)
        assert_ndarray_allclose(
            exact_laplace_covariance, post_covariance, rtol=5e-7,
            atol=0.)

        prior_pointwise_variance, posterior_pointwise_variance = \
            generate_and_save_pointwise_variance(
                prior, L_post_op_from_file,
                prior_variance_filename=prior_variance_filename,
                posterior_variance_filename=posterior_variance_filename)
        assert_ndarray_allclose(
            np.diag(exact_laplace_covariance), posterior_pointwise_variance,
            rtol=3e-11, atol=0.)

        # prior_pointwise_variance =\
        #     np.load(prior_variance_filename)['prior_pointwise_variance']
        posterior_pointwise_variance =\
            np.load(posterior_variance_filename)[
                'posterior_pointwise_variance']
        assert_ndarray_allclose(
            np.diag(exact_laplace_covariance), posterior_pointwise_variance,
            rtol=3e-11, atol=0.)

    def test_generate_and_save_laplace_posterior(self):
        # files will be created so move to temporary directory
        curdir = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dirname:
            os.chdir(temp_dirname)

            fault_precentage = 0.
            self.help_generate_and_save_laplace_posterior(
                fault_precentage)

            fault_precentage = 2.
            self.help_generate_and_save_laplace_posterior(
                fault_precentage)

            os.chdir(curdir)


class TestGoalOrientedInference(unittest.TestCase):

    def help_compute_posterior_mean_covar_optimal_for_prediction(
            self, num_dims, num_obs_qoi, num_pred_qoi):
        # print num_dims,num_obs_qoi,num_pred_qoi
        obs_matrix = np.random.normal(0., 1., (num_obs_qoi, num_dims))
        # prior_mean = np.zeros((num_dims),float)
        prior_mean = np.ones((num_dims), float)
        prior_cov = np.eye(num_dims)
        noise_cov = np.eye(num_obs_qoi)
        truth_sample = np.random.normal(0., 1., num_dims)
        obs = np.dot(obs_matrix, truth_sample) +\
            np.random.normal(0., 1., num_obs_qoi)
        pred_matrix = np.random.normal(0., 1., (num_pred_qoi, num_dims))
        pred_offset = 0.  # *np.random.normal(0.,1.,(num_pred_qoi))

        # prior_chol_factor = np.linalg.cholesky(prior_cov)
        prior_hessian = np.linalg.inv(prior_cov)
        noise_cov_inv = np.linalg.inv(noise_cov)

        # Compute true posterior
        exact_post_mean, exact_post_cov = \
            laplace_posterior_approximation_for_linear_models(
                obs_matrix, prior_mean, prior_hessian, noise_cov_inv, obs)

        # Compute true posterior push-forward
        exact_pf_mean, exact_pf_cov = \
            push_forward_gaussian_though_linear_model(
                pred_matrix, pred_offset, exact_post_mean, exact_post_cov)

        # Compute optimal push forward of the posterior
        opt_pf_mean, opt_pf_cov, posterior_mean, opt_post_cov = \
            compute_posterior_mean_covar_optimal_for_prediction(
                obs, obs_matrix, prior_mean, prior_cov, noise_cov,
                pred_matrix, economical=False)

        # print opt_pf_mean, exact_pf_mean

        # check prediction mean is exact
        assert np.allclose(opt_pf_mean, exact_pf_mean.squeeze())
        # check posterior mean pushed through prediction model is exact
        assert np.allclose(
            np.dot(pred_matrix, posterior_mean),
            np.dot(pred_matrix, exact_post_mean.squeeze()))
        # check prediction covariance is exact
        # print opt_pf_cov
        # print exact_pf_cov
        assert np.allclose(opt_pf_cov, exact_pf_cov)
        # check covariance pushed through prediction model is exact
        assert np.allclose(np.dot(pred_matrix, np.dot(
            opt_post_cov, pred_matrix.T)), exact_pf_cov)

    def test_goal_oriented_inference(self):
        num_dims = 10
        num_obs_qoi = 5
        num_pred_qoi = 3
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            num_dims, num_obs_qoi, num_pred_qoi)

        num_dims = 10
        num_obs_qoi = 15
        num_pred_qoi = 2
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            num_dims, num_obs_qoi, num_pred_qoi)

        num_dims = 10
        num_obs_qoi = 2
        num_pred_qoi = 5
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            num_dims, num_obs_qoi, num_pred_qoi)

        num_dims = 2
        num_obs_qoi = 3
        num_pred_qoi = 2
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            num_dims, num_obs_qoi, num_pred_qoi)

        num_dims = 3
        num_obs_qoi = 2
        num_pred_qoi = 3
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            num_dims, num_obs_qoi, num_pred_qoi)


if __name__ == '__main__':
    unittest.main()

    # suite = unittest.TestSuite()
    # suite.addTest( TestLaplace("test_log_unnormalized_posterior"))
    # runner = unittest.TextTestRunner()
    # runner.run( suite )
