import unittest
import os
import glob

import numpy as np

# from pyapprox.util.randomized_svd import (
#     randomized_svd,
#     MatVecOperator,
#     adjust_sign_svd,
#     svd_using_orthogonal_basis,
# )
from pyapprox.variables.gaussian import (
    DenseCholeskySqrtCovarianceOperator,
    CovarianceOperator,
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.interface.model import (
    DenseMatrixLinearModel,
    ChangeModelSignWrapper,
)
from pyapprox.bayes.laplace import (
    GaussianPushForward,
    DenseMatrixLaplacePosteriorApproximation,
    DenseMatrixLaplaceApproximationForPrediction,
    PriorConditionedHessianMatVecOperator,
    LaplacePosteriorLowRankApproximation,
    DenseMatrixLaplacePosteriorLowRankApproximation,
    ApplyNegLogLikelihoodHessian,
)
from pyapprox.bayes.likelihood import (
    LogUnNormalizedPosterior,
    ModelBasedGaussianLogLikelihood,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.util.visualization import plot_multiple_2d_gaussian_slices
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


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
        prior = DenseCholeskyMultivariateGaussian(mean, covariance, backend=bkd)
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
        prior = DenseCholeskyMultivariateGaussian(mean, covariance, backend=bkd)
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

    def test_low_rank_laplace_posterior_approximation(self):
        bkd = self.get_backend()
        nvars = 2
        nobs = 3
        mean = np.ones((nvars, 1))
        covariance = 0.2 * np.eye(nvars)
        noise_std = 0.01
        noise_cov = noise_std**2 * np.eye(nobs)
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        prior = DenseCholeskyMultivariateGaussian(mean, covariance, backend=bkd)
        full_laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        true_sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        noise = bkd.asarray(np.random.normal(0, noise_std, (nobs, 1)))
        obs = obs_mat @ true_sample + noise
        full_laplace.compute(obs)

        hess_mat = obs_mat.T @ bkd.inv(noise_cov) @ obs_mat
        # make low-rank laplace full rank and compare to full rank
        # laplace approx
        lr_laplace = DenseMatrixLaplacePosteriorLowRankApproximation(
            prior, hess_mat, rank=prior.nvars()
        )
        lr_laplace.compute()
        assert bkd.allclose(
            lr_laplace.posterior_covariance(),
            full_laplace.posterior_covariance(),
        )
        assert bkd.allclose(
            lr_laplace.covariance_diagonal(),
            bkd.diag(full_laplace.posterior_covariance()),
        )

    def test_laplace_with_randomized_svd(self):
        bkd = self.get_backend()
        nvars = 3
        nobs = 2

        # define prior
        prior_mean = bkd.full((nvars, 1), 0.5)
        prior_covariance = bkd.eye(nvars)
        prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_covariance)

        # define observations
        noise_sigma2 = 0.5
        noise_cov = bkd.eye(nobs) * noise_sigma2
        true_sample = prior.rvs(1)
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))

        model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(model, noise_cov)
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
        neg_log_unormalized_post = ChangeModelSignWrapper(
            LogUnNormalizedPosterior(loglike, prior)
        )

        # check gradients of unnormalized posterior
        errors = neg_log_unormalized_post.check_apply_jacobian(prior.mean())
        assert errors.min() / errors.max() < 1e-6
        errors = neg_log_unormalized_post.check_apply_hessian(prior.mean())
        assert errors.min() / errors.max() < 1e-6

        # find map point
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(gtol=1e-8, maxiter=100)
        # optimizer.set_verbosity(3)
        optimizer.set_objective_function(neg_log_unormalized_post)
        result = optimizer.minimize(prior.mean())
        map_point = result.x
        sample = map_point
        print(map_point, true_sample, "s")
        assert bkd.allclose(
            neg_log_unormalized_post.jacobian(sample),
            bkd.zeros((1, nvars)),
            atol=1e-7,
        )

        # Get analytical mean and covariance
        model_jac_at_map = model.jacobian(map_point)
        laplace = DenseMatrixLaplacePosteriorApproximation(
            model_jac_at_map,
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        laplace.compute(obs)

        rank = prior.nvars()
        prior_sqrt = DenseCholeskySqrtCovarianceOperator(prior.covariance())
        apply_hess = ApplyNegLogLikelihoodHessian(loglike, map_point)
        # only true for DenseMatrixLinearModel
        # when using nonlinear model the data misfit hessian
        # will be different to the misfit hessian obtained by computing
        # the data misfit hessian from a linearized model.
        # E.g. apply_hess(identity) !=
        # @ (model_jac_at_map.T @ bkd.inv(noise_cov) @ model_jac_at_map)
        assert bkd.allclose(map_point, laplace.posterior_mean())
        assert np.allclose(
            apply_hess(bkd.eye(nvars)),
            (model_jac_at_map.T @ bkd.inv(noise_cov) @ model_jac_at_map),
        )
        hessian_mat_vec_op = PriorConditionedHessianMatVecOperator(
            prior_sqrt, apply_hess
        )
        lr_laplace = LaplacePosteriorLowRankApproximation(
            prior, hessian_mat_vec_op, rank
        )
        lr_laplace.compute(noversampling=1000)
        assert bkd.allclose(
            lr_laplace.posterior_covariance(), laplace.posterior_covariance()
        )


class TestNumpyLaplace(TestLaplace, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main()
