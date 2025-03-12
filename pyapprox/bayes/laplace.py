import os
import numpy as np
from scipy.linalg import eigh as generalized_eigevalue_decomp

from pyapprox.util.linalg import (
    SymmetricMatrixDoublePassRandomizedSVD,
    DenseSymmetricMatVecOperator,
    SymmetricMatVecOperator,
)
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.variables.gaussian import (
    DenseMatrixMultivariateGaussian,
    GaussianSqrtCovarianceOperator,
)


class DenseMatrixLaplacePosteriorApproximation:
    def __init__(
        self,
        matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_cov: Array,
        vec: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        r"""
        Compute the mean and covariance of the Laplace posterior of a
        linear (or linearized) model with a Gaussian prior and noise model.

        Given some data d and a linear forward model, A(x) = Ax+b,
        and a Gaussian likelihood and a Gaussian prior, the resulting posterior
        is always Gaussian.

        Parameters
        ----------
        matrix : Array (num_qoi, nvars)
            The matrix reprsenting the linear forward model.

        prior_mean : Array (nvars, 1)
            The mean of the Gaussian prior

        prior_covariance: Array (nvars, nvars)
            The covarianceof the Gaussian prior

        noise_covariancev : Array (num_qoi, num_qoi)
            The covariance of the observational noise

        obs : Array (num_qoi, 1)
            The observations

        vec : Array (num_qoi, 1)
            The deterministic shift of the linear model
        """
        self._bkd = backend
        self._nobs, self._nvars = matrix.shape
        self._matrix = matrix
        if prior_mean.shape != (self.nvars(), 1):
            raise ValueError("prior_mean has the wrong shape")
        self._prior_mean = prior_mean
        if prior_cov.shape != (self.nvars(), self.nvars()):
            raise ValueError("prior_cov has the wrong shape")
        self._prior_cov = prior_cov
        if noise_cov.shape != (self.nobs(), self.nobs()):
            raise ValueError("noise_cov has the wrong shape")
        self._noise_cov = noise_cov
        if vec is None:
            vec = self._bkd.zeros((self._nobs, 1))
        if vec.shape != (self.nobs(), 1):
            raise ValueError("vec has the wrong shape")
        self._vec = vec

        self._noise_cov_inv = self._bkd.inv(self._noise_cov)
        self._prior_hessian = self._bkd.inv(self._prior_cov)
        model = DenseMatrixLinearModel(
            self._matrix, self._vec, backend=self._bkd
        )
        self._loglike = ModelBasedGaussianLogLikelihood(model, self._noise_cov)
        self._prior = DenseMatrixMultivariateGaussian(
            self._prior_mean, self._prior_cov, self._bkd
        )

    def _set_observations(self, obs: Array):
        if obs.shape != (self.nobs(), 1):
            raise ValueError("obs has the wrong shape")
        self._obs = obs

    def nvars(self) -> int:
        return self._nvars

    def nobs(self) -> int:
        return self._nobs

    def compute(self, obs: Array):
        self._set_observations(obs)
        misfit_hessian = self._matrix.T @ self._noise_cov_inv @ self._matrix
        self._posterior_cov = self._bkd.inv(
            misfit_hessian + self._prior_hessian
        )
        residual = self._obs - self._matrix @ self._prior_mean - self._vec
        temp = self._matrix.T @ (self._noise_cov_inv @ residual)
        self._posterior_mean = self._prior_mean + self._posterior_cov @ temp
        self._compute_evidence()
        self._compute_expected_posterior_statistics()
        self._compute_expected_kl_divergence()

    def _compute_evidence(self) -> Array:
        """
        References
        ----------
        Ryan, K. (2003). Estimating Expected Information Gains for Experimental
        Designs with Application to the Random Fatigue-Limit Model. Journal of
        Computational and Graphical Statistics, 12(3), 585-603.
        http://www.jstor.org/stable/1391040

        Friel, N. and Wyse, J. (2012), Estimating the evidence – a review.
        Statistica Neerlandica, 66: 288-308.
        https://doi.org/10.1111/j.1467-9574.2011.00515.x
        """
        self._loglike.set_observations(self._obs)
        lval = self._bkd.exp(self._loglike(self._posterior_mean))[:, 0]
        prior_val = self._prior.pdf(self._posterior_mean)
        assert lval.ndim == 1
        assert prior_val.ndim == 2
        self._evidence = (
            (2 * np.pi) ** (self._nvars / 2)
            * self._bkd.sqrt(self._bkd.det(self.posterior_covariance()))
            * lval[0]
            * prior_val[0, 0]
        )

    def posterior_mean(self) -> Array:
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_mean

    def posterior_covariance(self) -> Array:
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_cov

    def evidence(self) -> Array:
        return self._evidence

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def posterior_variable(self) -> DenseMatrixMultivariateGaussian:
        return DenseMatrixMultivariateGaussian(
            self.posterior_mean(),
            self.posterior_covariance(),
            backend=self._bkd,
        )

    def _compute_expected_posterior_statistics(self):
        """
        Compute the mean and variance of the posterior mean with respect to
        uncertainty in the observation data. The posterior mean is a
        Gaussian variable
        """
        Rmat = self._bkd.multidot(
            (self.posterior_covariance(), self._matrix.T, self._noise_cov_inv)
        )
        ROmat = Rmat @ self._matrix
        self._nu_vec = (ROmat @ self._prior_mean) + self._bkd.multidot(
            (
                self.posterior_covariance(),
                self._prior_hessian,
                self._prior_mean,
            )
        )
        self._Cmat = self._bkd.multidot(
            (ROmat, self._prior_cov, ROmat.T)
        ) + self._bkd.multidot((Rmat, self._noise_cov, Rmat.T))

    def _compute_expected_kl_divergence(self):
        """
        Compute the expected KL divergence between a Gaussian posterior
        and prior, where average is taken with respect to the data
        """
        kl_div = (
            self._bkd.trace(self._prior_hessian @ self.posterior_covariance())
            - self.nvars()
        )
        kl_div += self._bkd.log(
            self._bkd.det(self._prior_cov)
            / self._bkd.det(self.posterior_covariance())
        )
        kl_div += self._bkd.trace(self._prior_hessian @ self._Cmat)
        xi = self._prior_mean - self._nu_vec
        kl_div += self._bkd.multidot((xi.T, self._prior_hessian, xi))[0, 0]
        kl_div *= 0.5
        self._kl_div = kl_div

    def expected_kl_divergence(self) -> float:
        return self._kl_div


class PriorConditionedHessianMatVecOperator(SymmetricMatVecOperator):
    r"""
    Compute the action of prior conditioned misfit Hessian on a vector.

    E.g. for a arbitrary vector w, the Cholesky factor L of the prior
    and the misfit Hessian H compute
        L*H*L'*w
    """

    def __init__(
        self,
        prior_sqrt: GaussianSqrtCovarianceOperator,
        apply_hessian: callable,
    ):
        self._bkd = prior_sqrt._bkd
        self._prior_sqrt = prior_sqrt
        self._apply_hessian = apply_hessian

    def apply(self, vecs: Array) -> Array:
        Lv = self._prior_sqrt.apply(vecs)
        HLv = self._apply_hessian(Lv)
        return self._prior_sqrt.apply_transpose(HLv)

    def apply_transpose(self, vecs: Array) -> Array:
        return self.apply(vecs)

    def nrows(self) -> int:
        return self._prior_sqrt.nvars()


class LaplacePosteriorLowRankApproximation:
    def __init__(
        self,
        prior_conditioned_hessian: PriorConditionedHessianMatVecOperator,
        rank: int,
    ):
        if not isinstance(
            prior_conditioned_hessian, PriorConditionedHessianMatVecOperator
        ):
            raise ValueError(
                "prior_conditioned_hessian must be instance of"
                "PriorConditionedHessianMatVecOperator"
            )
        self._prior_condition_hess_op = prior_conditioned_hessian
        self._bkd = prior_conditioned_hessian._bkd
        if rank > self.nvars():
            raise ValueError("rank requested was to high")
        self._rank = rank

    def nvars(self) -> int:
        return self._prior.nvars()

    def __repr__(self) -> str:
        return "{0}(nvars={1})".format(self.__class__.__name__, self.nvars())

    def compute(self):
        svd_solver = SymmetricMatrixDoublePassRandomizedSVD(
            self._prior_condition_hess_op
        )
        self._Ur, self._Sr = svd_solver.compute(self._rank)[:2]
        P = 1 / self._bkd.sqrt(self._Sr + 1)
        self._post_cov_sqrt = self._prior._cov_sqrt.apply(
            self._Ur @ (P[:, None] * self._Ur.T)
        )

    def rvs(self, nsamples: int) -> Array:
        std_normal_samples = self._bkd.asarray(
            np.random.normal(0, 1, (self.nvars(), nsamples))
        )
        return self._post_cov_sqrt @ std_normal_samples + self._mean

    def covariance_diagonal(self) -> Array:
        # compute L*V_r
        tmp1 = self._prior._cov_sqrt.apply(self._Ur)
        # compute D*(L*V_r)**2
        tmp2 = self._Sr / (1.0 + self._Sr)
        tmp3 = self._bkd.sum(tmp1**2 * tmp2, axis=1)
        return self._prior.covariance_diagonal() - tmp3

    def posterior_covariance(self) -> Array:
        return self._post_cov_sqrt @ self._post_cov_sqrt.T


class DenseMatrixLaplacePosteriorLowRankApproximation(
    LaplacePosteriorLowRankApproximation
):
    def __init__(
        self,
        prior: DenseMatrixMultivariateGaussian,
        hess_mat: Array,
        rank: int,
    ):
        # mainly useful for testing
        if not isinstance(prior, DenseMatrixMultivariateGaussian):
            raise ValueError(
                "prior must be an instance of DenseMatrixMultivariateGaussian"
            )
        self._prior = prior
        self._hess_mat = hess_mat
        super().__init__(
            PriorConditionedHessianMatVecOperator(
                self._prior._cov_sqrt, self._apply_hessian
            ),
            rank,
        )

    def _apply_hessian(self, vecs: Array) -> Array:
        return self._hess_mat @ vecs


class GaussianPushForward:
    def __init__(
        self,
        matrix: Array,
        mean: Array,
        cov: Array,
        vec: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        r"""
        Find the mean and covariance of a gaussian distribution when it
        is push forward through a linear model. A linear transformation
        applied to a Gaussian is still a Gaussian.

        Original Gaussian with mean x and covariance \Sigma
        z~N(x,\Sigma)

        Transformation with b is a constant vector, e.g has no variance
        y = Az + b

        Distribution of resulting gaussian
        y~N(Ax+b,A\Sigma A^T)
        """
        self._bkd = backend
        self._nqoi, self._nvars = matrix.shape
        self._mat = matrix
        if mean.shape != (self.nvars(), 1):
            raise ValueError("mean has the wrong shape")
        self._mean = mean
        if cov.shape != (self.nvars(), self.nvars()):
            raise ValueError("cov has the wrong shape")
        self._cov = cov
        if vec is None:
            vec = self._bkd.zeros((self.nqoi(), 1))
        if vec.shape != (self.nqoi(), 1):
            raise ValueError("vec has the wrong shape")
        self._vec = vec
        self._compute()

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def _compute(self) -> Array:
        self._pushforward_mean = self._mat @ self._mean + self._vec
        self._pushforward_cov = self._mat @ self._cov @ self._mat.T

    def mean(self) -> Array:
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_mean

    def covariance(self) -> Array:
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> DenseMatrixMultivariateGaussian:
        return DenseMatrixMultivariateGaussian(self.mean(), self.covariance())


class DenseMatrixLaplaceApproximationForPrediction:
    def __init__(
        self,
        obs_matrix: Array,
        pred_matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        obs_noise_cov: Array,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self._obs_matrix = obs_matrix
        self._pred_matrix = pred_matrix
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._obs_noise_cov = obs_noise_cov

    def compute(self, obs: Array):
        # step 1
        OP = self._pred_matrix @ self._prior_cov
        # step 2
        C = OP @ self._obs_matrix.T
        # step 3
        Pz = OP @ self._pred_matrix.T
        # step 4
        Pz_inv = self._bkd.inv(Pz)
        # step 5
        A = C.T @ Pz_inv @ C
        # step 6
        data_cov = (
            self._obs_matrix @ self._prior_cov @ self._obs_matrix.T
            + self._obs_noise_cov
        )
        # step 7
        # print 'TODO replace generalized_eigevalue_decomp by my
        # subspace iteration'
        evals, evecs = generalized_eigevalue_decomp(A, data_cov)
        evecs = evecs[:, ::-1]
        evals = evals[::-1]
        rank = min(self._pred_matrix.shape[0], self._obs_matrix.shape[0])
        evecs = evecs[:, :rank]
        evals = evals[:rank]
        # step 8
        ppf_cov_evecs = C @ evecs

        residual = obs - self._obs_matrix @ self._prior_mean
        self._opt_pf_cov = Pz - ppf_cov_evecs @ ppf_cov_evecs.T
        self._opt_pf_mean = (ppf_cov_evecs @ (evecs.T @ residual)) + (
            self._pred_matrix @ self._prior_mean
        )

    def mean(self) -> Array:
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_mean

    def covariance(self) -> Array:
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> DenseMatrixMultivariateGaussian:
        return DenseMatrixMultivariateGaussian(self.mean(), self.covariance())
