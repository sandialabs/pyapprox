import unittest
from scipy.linalg import cholesky
import numpy as np
import os
import glob
import tempfile


from pyapprox.util.linalg import get_low_rank_matrix
from pyapprox.util.randomized_svd import (
    randomized_svd,
    MatVecOperator,
    adjust_sign_svd,
    svd_using_orthogonal_basis,
)
from pyapprox.variables.joint import MultivariateGaussian
from pyapprox.variables.gaussian import (
    DenseCholeskySqrtCovarianceOperator,
    CovarianceOperator,
)
from pyapprox.interface.model import DenseMatrixLinearModel, Model
from pyapprox.bayes.laplace import (
    get_laplace_covariance_sqrt_operator,
    get_pointwise_laplace_variance_using_prior_variance,
    GaussianPushForward,
    DenseMatrixLaplacePosteriorApproximation,
    DenseMatrixLaplaceApproximationForPrediction,
    PriorConditionedHessianMatVecOperator,
    LaplaceSqrtMatVecOperator,
)
from pyapprox.util.visualization import plot_multiple_2d_gaussian_slices
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def assert_ndarray_allclose(matrix1, matrix2, atol=1e-8, rtol=1e-5, msg=None):
    """
    A more useful function for testing equivalence of numpy arrays.
    Print norms used by np.allclose function to determine equivalence.

    Matrix1 is considered the truth
    """
    if not np.allclose(matrix1, matrix2, atol=atol, rtol=rtol):
        if msg is not None:
            print(msg)
        diff = np.absolute(matrix1 - matrix2)
        abs_error = diff.max()
        rel_error = (diff / np.absolute(matrix1)).max()
        print("abs error:", abs_error)
        print("rel error:", rel_error)
        print("atol:", atol)
        print("rtol:", rtol)
        print("matrix1 shape", matrix1.shape)
        print("matrix2 shape", matrix2.shape)
        assert False, "matrices are not equivalent"


def setup_quadratic_misfit_problem(prior, rank, noise_sigma2=1):
    # Define observations
    nqoi = 2 * rank
    # assert nqoi>=rank

    noise_covariance = np.eye(nqoi) * noise_sigma2
    noise_covariance_inv = np.linalg.inv(noise_covariance)
    # In high dimensions computing cholesky factor is too expensive.
    # That is why we use PDE based operator
    noise_covariance_chol_factor = np.linalg.cholesky(noise_covariance)
    truth_sample = prior.generate_samples(1)[:, 0]
    nvars = truth_sample.shape[0]
    Amatrix = get_low_rank_matrix(nqoi, nvars, rank)
    noise = np.dot(
        noise_covariance_chol_factor,
        np.random.normal(0.0, noise_sigma2, nqoi),
    )
    obs = np.dot(Amatrix, truth_sample) + noise

    # Define mistit model
    misfit_model = QuadraticMisfitModel(nvars, rank, nqoi, Amatrix)

    return misfit_model, noise_covariance_inv, obs


def posterior_covariance_helper(
    prior, rank, comparison_tol, test_sampling=False, plot=False
):
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
    nvars = prior.nvars()
    prior_mean = np.zeros((nvars), float)
    L = L_op(np.eye(nvars), False)
    L_T = L_op(np.eye(nvars), True)
    assert_ndarray_allclose(
        L.T,
        L_T,
        rtol=comparison_tol,
        atol=0,
        msg="Comparing prior sqrt and transpose",
    )

    prior_covariance = np.dot(L, L_T)
    prior_pointwise_variance = prior.pointwise_variance()
    assert_ndarray_allclose(
        np.diag(prior_covariance),
        prior_pointwise_variance,
        rtol=1e-14,
        atol=0,
        msg="Comparing prior pointwise variance",
    )

    misfit_model, noise_covariance_inv, obs = setup_quadratic_misfit_problem(
        prior, rank, noise_sigma2=1
    )

    # Get analytical mean and covariance
    prior_hessian = np.linalg.inv(prior_covariance)
    exact_laplace_mean, exact_laplace_covariance = (
        laplace_posterior_approximation_for_linear_models(
            misfit_model.Amatrix,
            prior.mean,
            prior_hessian,
            noise_covariance_inv,
            obs,
        )
    )

    # Define prior conditioned misfit operator
    sample = np.zeros(nvars)
    misfit_hessian_operator = MisfitHessianVecOperator(
        misfit_model, sample, fd_eps=None
    )
    LHL_op = PriorConditionedHessianMatVecOperator(
        L_op, misfit_hessian_operator
    )

    # For testing purposes build entire L*H*L matrix using operator
    # and compare to result based upon explicit matrix mutiplication
    LHL_op = LHL_op.apply(np.eye(nvars), transpose=False)
    H = misfit_model.hessian(sample)
    assert np.allclose(
        H,
        np.dot(
            np.dot(misfit_model.Amatrix.T, noise_covariance_inv),
            misfit_model.Amatrix,
        ),
    )
    LHL_mat = np.dot(L_T, np.dot(H, L))
    assert_ndarray_allclose(
        LHL_mat,
        LHL_op,
        rtol=comparison_tol,
        msg="Comparing prior matrix and operator based LHL",
    )

    # Test singular values obtained by randomized svd using operator
    # are the same as those obtained using singular decomposition
    Utrue, Strue, Vtrue = np.linalg.svd(LHL_mat)
    Utrue, Vtrue = adjust_sign_svd(Utrue, Vtrue)
    standard_svd_opts = {"nsingular_values": rank, "nextra_samples": 10}
    svd_opts = {"single_pass": True, "standard_opts": standard_svd_opts}
    L_post_op = get_laplace_covariance_sqrt_operator(
        L_op,
        misfit_hessian_operator,
        svd_opts,
        weights=None,
        min_singular_value=0.0,
    )
    # print np.max((Strue[:rank]-L_post_op.e_r)/Strue[0])
    max_error = np.max(Strue[:rank] - L_post_op.e_r)
    assert max_error / Strue[0] < comparison_tol, max_error / Strue[0]
    assert_ndarray_allclose(
        Vtrue.T[:, :rank],
        L_post_op.V_r,
        rtol=1e-6,
        msg="Comparing eigenvectors",
    )

    L_post_op.V_r = Vtrue.T[:, :rank]

    # Test posterior sqrt covariance operator transpose is the same as
    # explicit matrix transpose of matrix obtained by prior sqrt
    # covariance operator
    L_post = L_post_op.apply(np.eye(nvars), transpose=False)
    L_post_T = L_post_op.apply(np.eye(nvars), transpose=True)
    assert_ndarray_allclose(
        L_post.T,
        L_post_T,
        rtol=comparison_tol,
        msg="Comparing posterior sqrt and transpose",
    )

    # Test posterior covariance operator produced matrix is the same
    # as the exact posterior covariance obtained using analytical formula
    if rank == nvars:
        # this test only makes sense if entire set of directions is found
        # if low rank approx is used then this will ofcourse induce errors
        post_covariance = np.dot(L_post, L_post_T)
        assert_ndarray_allclose(
            exact_laplace_covariance,
            post_covariance,
            rtol=comparison_tol,
            atol=0.0,
            msg="Comparing matrix and operator based posterior covariance",
        )

    # Test pointwise covariance of posterior
    post_pointwise_variance, prior_pointwise_variance = (
        get_pointwise_laplace_variance_using_prior_variance(
            prior, L_post_op, prior_pointwise_variance
        )
    )
    assert_ndarray_allclose(
        np.diag(exact_laplace_covariance),
        post_pointwise_variance,
        rtol=comparison_tol,
        atol=0.0,
        msg="Comparing pointwise variance",
    )

    if not test_sampling:
        return
    nsamples = int(2e5)
    posterior_samples = sample_from_laplace_posterior(
        exact_laplace_mean, L_post_op, nvars, nsamples, weights=None
    )
    assert_ndarray_allclose(
        exact_laplace_covariance,
        np.cov(posterior_samples),
        atol=1e-2 * exact_laplace_covariance.max(),
        rtol=0.0,
        msg="Comparing posterior samples covariance",
    )
    assert_ndarray_allclose(
        exact_laplace_mean.squeeze(),
        np.mean(posterior_samples, axis=1),
        atol=2e-2,
        rtol=0.0,
        msg="Comparing posterior samples mean",
    )

    if plot:
        # plot marginals of posterior using orginal ordering
        texfilename = "slices.tex"
        plot_multiple_2d_gaussian_slices(
            exact_laplace_mean[:10],
            np.diag(exact_laplace_covariance)[:10],
            texfilename,
            reference_gaussian_data=(0.0, 1.0),
            show=False,
        )

        # plot marginals of posterior in rotated coordinates
        # from most to least important.
        # The following is not feasiable in practice as we cannot compute
        # entire covariance matrix in full space. But we have
        # C_r = V_r*L*V_r*D*V_r.T*L.T*V_r.T
        # is we compute matrix products from right to left we only have to
        # compute at most (d x r) matrices. And if only want first 20 say
        # variances then can apply C_r to vectors e_i i=1,...,20
        # then we need at most (dx20 matrices)
        texfilename = "rotated-slices.tex"
        V_r = L_post_op.V_r
        plot_multiple_2d_gaussian_slices(
            np.dot(V_r.T, exact_laplace_mean[:10]),
            np.diag(np.dot(V_r.T, np.dot(exact_laplace_covariance, V_r)))[:10],
            texfilename,
            reference_gaussian_data=(0.0, 1.0),
            show=True,
        )


class TestLaplace:

    def setUp(self):
        np.random.seed(2)

    @unittest.skip(reason="only shows how to plot")
    def test_plot_multiple_2d_gaussian_slices(self):

        mean = np.array([0, 1, -1])
        covariance = np.diag(np.array([1, 0.5, 0.025]))

        texfilename = "slices.tex"
        plot_multiple_2d_gaussian_slices(
            mean[:10],
            np.diag(covariance)[:10],
            texfilename,
            reference_gaussian_data=(0.0, 1.0),
            show=False,
        )
        filenames = glob.glob(texfilename[:-4] + "*")
        for filename in filenames:
            os.remove(filename)

    def test_operator_diagonal(self):
        bkd = self.get_backend()
        nvars = 4
        randn = bkd.asarray(np.random.normal(0.0, 1.0, (nvars, nvars)))
        prior_covariance = randn.T @ randn
        sqrt_covar_op = DenseCholeskySqrtCovarianceOperator(
            prior_covariance, backend=bkd
        )

        batch_size = 3
        cov_op = CovarianceOperator(sqrt_covar_op)
        diagonal = cov_op.diagonal(batch_size)
        assert bkd.allclose(diagonal, bkd.diag(prior_covariance))

    def test_push_forward_gaussian_though_linear_model(self):
        bkd = self.get_backend()
        nqoi = 1
        nvars = 2
        mean = bkd.ones((nvars, 1))
        covariance = 0.1 * bkd.eye(nvars)
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nqoi, nvars)))
        b = bkd.asarray(np.random.normal(0.0, 1.0, (nqoi, 1)))
        prior = MultivariateGaussian(mean, covariance, backend=bkd)
        push_forward = GaussianPushForward(
            A, prior.mean(), prior.covariance(), b, backend=bkd
        )

        # Generate samples from original density and push forward through model
        # and approximate density using KDE
        nsamples = 1000000
        samples = prior.rvs(nsamples)
        model = DenseMatrixLinearModel(A, b, backend=bkd)
        values = model(samples)
        assert bkd.allclose(
            push_forward.mean(), np.mean(values, axis=0), rtol=1e-2
        )
        assert bkd.allclose(
            push_forward.covariance(),
            bkd.cov(values, rowvar=False, ddof=1),
            rtol=1e-2,
        )

    def test_posterior_push_forward_gaussian_though_linear_model(self):
        bkd = self.get_backend()
        nqoi = 1
        nvars = 2
        nobs = 3
        mean = np.ones((nvars, 1))
        covariance = 0.1 * np.eye(nvars)
        noise_std = 0.01
        noise_cov = noise_std**2 * np.eye(nobs)
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        pred_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nqoi, nvars)))
        prior = MultivariateGaussian(mean, covariance, backend=bkd)
        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        true_sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        noise = bkd.asarray(np.random.normal(0, noise_std, (nobs, 1)))
        obs = obs_mat @ true_sample + noise
        laplace.compute(obs)
        posterior_push_forward = GaussianPushForward(
            pred_mat,
            laplace.posterior_mean(),
            laplace.posterior_covariance(),
            backend=bkd,
        )
        laplace4pred = DenseMatrixLaplaceApproximationForPrediction(
            obs_mat,
            pred_mat,
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        laplace4pred.compute(obs)
        assert bkd.allclose(laplace4pred.mean(), posterior_push_forward.mean())
        assert bkd.allclose(
            laplace4pred.covariance(), posterior_push_forward.covariance()
        )

    def test_posterior_dense_matrix_covariance_operator(self):
        nvars = 121
        rank = 10
        eval_concurrency = 20
        # randn = np.random.normal(0.,1.,(nvars,nvars))
        # prior_covariance = np.dot(randn.T,randn)
        prior_covariance = np.eye(nvars)
        prior_sqrt_covariance_op = CholeskySqrtCovarianceOperator(
            prior_covariance, eval_concurrency, backend=bkd
        )
        prior = MultivariateGaussian(prior_sqrt_covariance_op)
        comparison_tol = 6e-7
        posterior_covariance_helper(
            prior, rank, comparison_tol, test_sampling=True
        )

    def test_hessian_vector_multiply_operator_with_randomized_svd(self):
        nvars = 100
        rank = 21
        nqoi = 30
        concurrency = 10

        model = QuadraticMisfitModel(nvars, rank, nqoi)

        # --------------------------------- #
        # compute with exact misfit Hessian #
        # --------------------------------- #
        np.random.seed(2)
        Amatrix = model.Amatrix
        operator = MatVecOperator(np.dot(Amatrix.T, Amatrix))
        adaptive_svd_opts = {
            "tolerance": 1e-8,
            "nextra_samples": 10,
            "max_niter_error_increase": 20,
            "verbosity": 0,
            "max_nsamples": 100,
            "concurrency": concurrency,
        }
        svd_opts = {"adaptive_opts": adaptive_svd_opts}
        U, S, V = randomized_svd(operator, svd_opts)
        Utrue, Strue, Vtrue = np.linalg.svd(
            np.dot(Amatrix.T, Amatrix), full_matrices=False
        )
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
        map_point = np.zeros(nvars)
        operator = MisfitHessianVecOperator(model, map_point, fd_eps=1e-7)
        U, S, V = randomized_svd(operator, svd_opts)
        Utrue, Strue, Vtrue = np.linalg.svd(
            np.dot(Amatrix.T, Amatrix), full_matrices=False
        )
        Utrue, Vtrue = adjust_sign_svd(Utrue, Vtrue)
        J = np.where(Strue > 1e-9)[0]
        assert J.shape[0] == rank
        assert np.allclose(Strue[J], S[J])
        assert np.allclose(Utrue[:, J], U[:, J])
        assert np.allclose(Vtrue[J, :], V[J, :])

    def test_prior_conditioned_misfit_covariance_operator(self):
        nvars = 3
        rank = 2
        nqoi = 2

        # define prior
        prior_mean = np.zeros((nvars), float)
        prior_covariance = np.eye(nvars)
        prior_hessian = np.eye(nvars)
        prior_density = NormalDensity(prior_mean, covariance=prior_covariance)

        # define observations
        noise_sigma2 = 0.5
        noise_covariance = np.eye(nqoi) * noise_sigma2
        noise_covariance_inv = np.linalg.inv(noise_covariance)
        noise_covariance_chol_factor = np.linalg.cholesky(noise_covariance)
        truth_sample = prior_density.generate_samples(1)[:, 0]
        Amatrix = get_low_rank_matrix(nqoi, nvars, rank)
        noise = np.dot(
            noise_covariance_chol_factor, np.random.normal(0.0, 1.0, nqoi)
        )
        obs = np.dot(Amatrix, truth_sample) + noise

        # define mistit model
        misfit_model = QuadraticMisfitModel(
            nvars, rank, nqoi, obs, noise_covariance, Amatrix
        )

        # Get analytical mean and covariance
        exact_laplace_mean, exact_laplace_covariance = (
            laplace_posterior_approximation_for_linear_models(
                misfit_model.Amatrix,
                prior_mean,
                prior_hessian,
                noise_covariance_inv,
                obs,
            )
        )

        objective = LogUnormalizedPosterior(
            misfit_model,
            misfit_model.gradient_set,
            prior_density.pdf,
            prior_density.log_pdf,
            prior_density.log_pdf_gradient,
        )
        map_point, obj_min = find_map_point(objective, prior_mean)
        sample = map_point
        assert np.allclose(objective.gradient(sample), np.zeros(nvars))
        prior_covariance_sqrt_operator = CholeskySqrtCovarianceOperator(
            prior_density.covariance
        )
        misfit_hessian_operator = MisfitHessianVecOperator(
            misfit_model, sample, fd_eps=1e-7
        )
        standard_svd_opts = {
            "nsingular_values": rank,
            "nextra_samples": 10,
        }
        svd_opts = {"single_pass": False, "standard_opts": standard_svd_opts}
        laplace_covariance_sqrt = get_laplace_covariance_sqrt_operator(
            prior_covariance_sqrt_operator, misfit_hessian_operator, svd_opts
        )

        identity = np.eye(nvars)
        laplace_covariance_chol_factor = laplace_covariance_sqrt.apply(
            identity
        )
        laplace_covariance = np.dot(
            laplace_covariance_chol_factor, laplace_covariance_chol_factor.T
        )
        # print laplace_covariance
        # print exact_laplace_covariance
        assert np.allclose(laplace_covariance, exact_laplace_covariance)

    @unittest.skip("I need to implement this test using my new kle class")
    def test_prior_covariance_operator_based_kle(self):
        mesh = np.linspace(0.0, 1.0, 10)
        C = correlation_function(mesh[np.newaxis, :], 0.1, "exp")
        # U,S,V = np.linalg.svd(C)
        # Csqrt = np.dot(U,np.dot(np.diag(np.sqrt(S)),V))
        # assert np.allclose(np.dot(Csqrt,Csqrt),C)
        sqrt_covariance_operator = CholeskySqrtCovarianceOperator(C, 10)
        covariance_operator = CovarianceOperator(sqrt_covariance_operator)
        kle = KLE()
        standard_svd_opts = {
            "nsingular_values": mesh.shape[0],
            "nextra_samples": 10,
        }
        svd_opts = {"single_pass": False, "standard_opts": standard_svd_opts}
        kle.solve_eigenvalue_problem(
            covariance_operator, mesh.shape[0], 0.0, "randomized-svd", svd_opts
        )
        rand_svd_eig_vals = kle.eig_vals.copy()
        kle.solve_eigenvalue_problem(C, mesh.shape[0], 0.0, "eig")
        eig_eig_vals = kle.eig_vals.copy()
        assert np.allclose(rand_svd_eig_vals, eig_eig_vals)

    def help_generate_and_save_laplace_posterior(self, fault_precentage):
        svd_history_filename = "svd-history.npz"
        Lpost_op_filename = "laplace_sqrt_operator.npz"
        prior_variance_filename = "prior-pointwise-variance.npz"
        posterior_variance_filename = "posterior-pointwise-variance.npz"
        if os.path.exists(svd_history_filename):
            os.remove(svd_history_filename)
        if os.path.exists(Lpost_op_filename):
            os.remove(Lpost_op_filename)
        if os.path.exists(prior_variance_filename):
            os.remove(prior_variance_filename)
        if os.path.exists(posterior_variance_filename):
            os.remove(posterior_variance_filename)

        nvars = 121
        rank = 20
        eval_concurrency = 2
        prior_covariance = np.eye(nvars)
        prior_sqrt_covariance_op = CholeskySqrtCovarianceOperator(
            prior_covariance, eval_concurrency, fault_percentage=2
        )
        prior = MultivariateGaussian(prior_sqrt_covariance_op)

        misfit_model, noise_covariance_inv, obs = (
            setup_quadratic_misfit_problem(prior, rank, noise_sigma2=1)
        )

        # Get analytical mean and covariance
        prior_hessian = np.linalg.inv(prior_covariance)
        exact_laplace_mean, exact_laplace_covariance = (
            laplace_posterior_approximation_for_linear_models(
                misfit_model.Amatrix,
                prior.mean,
                prior_hessian,
                noise_covariance_inv,
                obs,
            )
        )

        # instead of using optimization to find map point just use exact
        # map
        misfit_model.map_point = lambda: exact_laplace_mean

        nsingular_values = prior.nvars()
        try:
            L_post_op = generate_and_save_laplace_posterior(
                prior,
                misfit_model,
                nsingular_values,
                svd_history_filename=svd_history_filename,
                Lpost_op_filename=Lpost_op_filename,
                nextra_svd_samples=30,
                fd_eps=None,
            )
        except:
            recovered_svd_data = np.load("randomized_svd_recovery_data.npz")
            X = recovered_svd_data["X"]
            Y = recovered_svd_data["Y"]
            I = np.where(np.all(np.isfinite(Y), axis=0))[0]
            X = X[:, I]
            Y = Y[:, I]
            Q, R = np.linalg.qr(Y)
            U, S, V = svd_using_orthogonal_basis(None, Q, X, Y, True)
            U = U[:, :nsingular_values]
            S = S[:nsingular_values]
            L_post_op = LaplaceSqrtMatVecOperator(
                prior_sqrt_covariance_op, e_r=S, V_r=U
            )
            L_post_op.save(Lpost_op_filename)

        L_post_op_from_file = LaplaceSqrtMatVecOperator(
            prior_sqrt_covariance_op, filename=Lpost_op_filename
        )

        # turn off faults in prior to allow easy comparison
        prior_sqrt_covariance_op.fault_percentage = 0
        L_post = L_post_op_from_file.apply(np.eye(nvars), transpose=False)
        L_post_T = L_post_op_from_file.apply(np.eye(nvars), transpose=True)
        assert_ndarray_allclose(L_post.T, L_post_T, rtol=1e-12)

        # Test posterior covariance operator produced matrix is the same
        # as the exact posterior covariance obtained using analytical formula
        post_covariance = np.dot(L_post, L_post_T)
        assert_ndarray_allclose(
            exact_laplace_covariance, post_covariance, rtol=5e-7, atol=0.0
        )

        prior_pointwise_variance, posterior_pointwise_variance = (
            generate_and_save_pointwise_variance(
                prior,
                L_post_op_from_file,
                prior_variance_filename=prior_variance_filename,
                posterior_variance_filename=posterior_variance_filename,
            )
        )
        assert_ndarray_allclose(
            np.diag(exact_laplace_covariance),
            posterior_pointwise_variance,
            rtol=3e-11,
            atol=0.0,
        )

        # prior_pointwise_variance =\
        #     np.load(prior_variance_filename)['prior_pointwise_variance']
        posterior_pointwise_variance = np.load(posterior_variance_filename)[
            "posterior_pointwise_variance"
        ]
        assert_ndarray_allclose(
            np.diag(exact_laplace_covariance),
            posterior_pointwise_variance,
            rtol=3e-11,
            atol=0.0,
        )

    def test_generate_and_save_laplace_posterior(self):
        # files will be created so move to temporary directory
        curdir = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dirname:
            os.chdir(temp_dirname)

            fault_precentage = 0.0
            self.help_generate_and_save_laplace_posterior(fault_precentage)

            fault_precentage = 2.0
            self.help_generate_and_save_laplace_posterior(fault_precentage)

            os.chdir(curdir)


class TestGoalOrientedInference(unittest.TestCase):

    def help_compute_posterior_mean_covar_optimal_for_prediction(
        self, nvars, nobs_qoi, npred_qoi
    ):
        # print nvars,nobs_qoi,npred_qoi
        obs_matrix = np.random.normal(0.0, 1.0, (nobs_qoi, nvars))
        # prior_mean = np.zeros((nvars),float)
        prior_mean = np.ones((nvars), float)
        prior_cov = np.eye(nvars)
        noise_cov = np.eye(nobs_qoi)
        truth_sample = np.random.normal(0.0, 1.0, nvars)
        obs = np.dot(obs_matrix, truth_sample) + np.random.normal(
            0.0, 1.0, nobs_qoi
        )
        pred_matrix = np.random.normal(0.0, 1.0, (npred_qoi, nvars))
        pred_offset = 0.0  # *np.random.normal(0.,1.,(npred_qoi))

        # prior_chol_factor = np.linalg.cholesky(prior_cov)
        prior_hessian = np.linalg.inv(prior_cov)
        noise_cov_inv = np.linalg.inv(noise_cov)

        # Compute true posterior
        exact_post_mean, exact_post_cov = (
            laplace_posterior_approximation_for_linear_models(
                obs_matrix, prior_mean, prior_hessian, noise_cov_inv, obs
            )
        )

        # Compute true posterior push-forward
        exact_pf_mean, exact_pf_cov = (
            push_forward_gaussian_though_linear_model(
                pred_matrix, pred_offset, exact_post_mean, exact_post_cov
            )
        )

        # Compute optimal push forward of the posterior
        opt_pf_mean, opt_pf_cov, posterior_mean, opt_post_cov = (
            compute_posterior_mean_covar_optimal_for_prediction(
                obs,
                obs_matrix,
                prior_mean,
                prior_cov,
                noise_cov,
                pred_matrix,
                economical=False,
            )
        )

        # print opt_pf_mean, exact_pf_mean

        # check prediction mean is exact
        assert np.allclose(opt_pf_mean, exact_pf_mean.squeeze())
        # check posterior mean pushed through prediction model is exact
        assert np.allclose(
            np.dot(pred_matrix, posterior_mean),
            np.dot(pred_matrix, exact_post_mean.squeeze()),
        )
        # check prediction covariance is exact
        # print opt_pf_cov
        # print exact_pf_cov
        assert np.allclose(opt_pf_cov, exact_pf_cov)
        # check covariance pushed through prediction model is exact
        assert np.allclose(
            np.dot(pred_matrix, np.dot(opt_post_cov, pred_matrix.T)),
            exact_pf_cov,
        )

    def test_goal_oriented_inference(self):
        nvars = 10
        nobs_qoi = 5
        npred_qoi = 3
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            nvars, nobs_qoi, npred_qoi
        )

        nvars = 10
        nobs_qoi = 15
        npred_qoi = 2
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            nvars, nobs_qoi, npred_qoi
        )

        nvars = 10
        nobs_qoi = 2
        npred_qoi = 5
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            nvars, nobs_qoi, npred_qoi
        )

        nvars = 2
        nobs_qoi = 3
        npred_qoi = 2
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            nvars, nobs_qoi, npred_qoi
        )

        nvars = 3
        nobs_qoi = 2
        npred_qoi = 3
        self.help_compute_posterior_mean_covar_optimal_for_prediction(
            nvars, nobs_qoi, npred_qoi
        )


class TestNumpyLaplace(TestLaplace, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main()
