"""
Clases for Computing Posterriors with Conjugate Priors

This module contains implementations of various posterior with conjugate
priors.
When the prior and posterior belong to the same parametric family,
the prior is said to be conjugate for the likelihood.
The classes below compute the posterior distributions analytically from
observations.

Purpose
-------
These classes are designed for useful for testing Bayesian inference
algorithms, where posterior distributions are computed based on prior
distributions and observed data.

Additionally, the Laplace approximation is for computing the posterior
distribution using a conjugate Gaussian prior and noise model when using
linear(ized) observation models.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import eigh as generalized_eigevalue_decomp

from pyapprox.util.linalg import (
    SymmetricMatrixDoublePassRandomizedSVD,
    SymmetricMatVecOperator,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    GaussianSqrtCovarianceOperator,
    MultivariateGaussian,
)
from pyapprox.bayes.likelihood import GaussianLogLikelihood
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    DirichletVariable,
)
from pyapprox.variables.marginals import BetaMarginal, log_beta_function


class DenseMatrixLaplacePosteriorApproximation:
    def __init__(
        self,
        matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_cov: Array,
        vec: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        r"""
        Compute the mean and covariance of the Laplace posterior of a
        linear (or linearized) model with a Gaussian prior and noise model.

        Given some data `d` and a linear forward model, `A(x) = Ax+b`,
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

        noise_covariance : Array (num_qoi, num_qoi)
            The covariance of the observational noise

        obs : Array (num_qoi, 1)
            The observations

        vec : Array (num_qoi, 1)
            The deterministic shift of the linear model

        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.

        Raises
        ------
        ValueError
            If the shapes of `prior_mean`, `prior_cov`, `noise_cov`,
            or `vec` are invalid.
        """
        self._bkd = backend
        self._nobs, self._nvars = matrix.shape
        self._matrix = matrix
        if prior_mean.shape != (self.nvars(), 1):
            raise ValueError(
                "prior_mean has the wrong shape {0} must be {1}".format(
                    prior_mean.shape, (self.nvars(), 1)
                )
            )
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
        self._prior = DenseCholeskyMultivariateGaussian(
            self._prior_mean, self._prior_cov, self._bkd
        )

    def _set_observations(self, obs: Array):
        if obs.ndim != 2 or obs.shape[0] != self.nobs():
            raise ValueError("obs has the wrong shape")
        self._obs = obs

    def nvars(self) -> int:
        """
        Returns the number of variables in the model.

        Returns
        -------
        int
            Number of variables in the model.
        """
        return self._nvars

    def nobs(self) -> int:
        """
        Returns the number of observations in a single experiment.

        Returns
        -------
        int
            Number of observations.
        """
        return self._nobs

    def compute(self, obs: Array):
        """
        Compute the posterior mean, covariance, evidence, and expected KL
        divergence given the observations.

        Parameters
        ----------
        obs : Array of shape (num_qoi, num_experiments)
            Observations used to compute the posterior.

        Raises
        ------
        ValueError
            If the shape of `obs` is invalid.
        """
        self._set_observations(obs)
        misfit_hessian = self._matrix.T @ self._noise_cov_inv @ self._matrix
        # raise NotImplementedError("need to modify for multiple experiments")
        self._posterior_cov = self._bkd.inv(
            obs.shape[1] * misfit_hessian + self._prior_hessian
        )
        obs_sample_sum = self._bkd.sum(obs, axis=1)[:, None]
        residual = obs_sample_sum - self._matrix @ self._prior_mean - self._vec
        temp = self._matrix.T @ (self._noise_cov_inv @ residual)
        self._posterior_mean = self._prior_mean + self._posterior_cov @ temp
        self._compute_evidence()
        self._compute_expected_posterior_statistics()
        self._compute_expected_kl_divergence()

    def _compute_evidence(self) -> Array:
        """
        References
        ----------
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
        """
        Returns the posterior mean.

        Returns
        -------
        mean : Array (nvars, 1)
            Posterior mean.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_mean

    def posterior_covariance(self) -> Array:
        """
        Returns the posterior covariance.

        Returns
        -------
        cov : Array (nvars, nvars)
            Posterior covariance.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_cov

    def evidence(self) -> Array:
        """
        Returns the evidence of the model.

        Returns
        -------
        evidence : float
            Evidence of the model.
        """
        return self._evidence

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def posterior_variable(self) -> DenseCholeskyMultivariateGaussian:
        """
        Returns the posterior as a Gaussian variable.

        Returns
        -------
        posterior : DenseCholeskyMultivariateGaussian
            The Gaussian posterior variable.
        """
        return DenseCholeskyMultivariateGaussian(
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
        """
        Returns the expected KL divergence between the posterior and prior.

        Returns
        -------
        expected_kl : float
            Expected KL divergence.
        """
        return self._kl_div


class ApplyNegLogLikelihoodHessian:
    def __init__(self, loglike: GaussianLogLikelihood, sample: Array):
        self._loglike = loglike
        self._bkd = loglike._bkd
        if not loglike.apply_hessian_implemented():
            # TODO allow computation of directional finite differences of jac
            raise ValueError(
                "loglike.apply_hessian_implemented() must be True"
            )
        self.set_sample(sample)

    def set_sample(self, sample: Array):
        self._sample = sample

    def __call__(self, vecs: Array) -> Array:
        return -self._loglike.apply_hessian(self._sample, vecs)


class PriorConditionedHessianMatVecOperator(SymmetricMatVecOperator):
    r"""
    This class computes the action of the prior-conditioned misfit Hessian
    on a vector.
    Specifically, for an arbitrary vector `w`, the Cholesky factor `L` of the
    prior  and the misfit Hessian `H`, it computes:

    .. math::
        L \cdot H \cdot L^T \cdot w
    Compute the action of prior conditioned misfit Hessian on a vector.
    """

    def __init__(
        self,
        prior_sqrt: GaussianSqrtCovarianceOperator,
        apply_hessian: callable,
    ):
        """
        Initialize the PriorConditionedHessianMatVecOperator class.

        Parameters
        ----------
        prior_sqrt : GaussianSqrtCovarianceOperator
            The Cholesky factor `L` of the prior covariance matrix.

        apply_hessian : callable
            A function that applies the misfit Hessian `H` to a vector.
        """
        self._bkd = prior_sqrt._bkd
        self._prior_sqrt = prior_sqrt
        self._apply_hessian = apply_hessian

    def nvars(self) -> int:
        """
        Returns the number of variables of the prior.

        Returns
        -------
        int
            Number of variables in the model.
        """
        return self._prior_sqrt.nvars()

    def apply(self, vecs: Array) -> Array:
        r"""
        Applies the prior-conditioned misfit Hessian to a vector.

        For a given vector `vecs`, this method computes:

        .. math::
            L \cdot H \cdot L^T \cdot vecs

        Where:
        - `L` is the Cholesky factor of the prior covariance matrix.
        - `H` is the misfit Hessian.

        Parameters
        ----------
        vecs : Array
            Input vector(s) to which the operator is applied.

        Returns
        -------
        Array
            Result of applying the prior-conditioned misfit Hessian to the
            input vector(s).
        """
        Lv = self._prior_sqrt.apply(vecs)
        HLv = self._apply_hessian(Lv)
        return self._prior_sqrt.apply_transpose(HLv)

    def apply_transpose(self, vecs: Array) -> Array:
        """
        Applies the transpose of the prior-conditioned misfit Hessian to a
        vector.

        For symmetric operators, this is equivalent to `apply()`.

        Parameters
        ----------
        vecs : Array
            Input vector(s) to which the transpose operator is applied.

        Returns
        -------
        Array
            Result of applying the transpose of the prior-conditioned misfit
            Hessian to the input vector(s).
        """
        return self.apply(vecs)

    def nrows(self) -> int:
        """
        Returns the number of rows in the operator.

        For this operator, the number of rows is equal to the number of
        variables.

        Returns
        -------
        int
            Number of rows in the operator.
        """
        return self._prior_sqrt.nvars()


class LaplacePosteriorLowRankApproximation:
    def __init__(
        self,
        prior: MultivariateGaussian,
        prior_conditioned_hessian: PriorConditionedHessianMatVecOperator,
        rank: int,
    ):
        if not isinstance(prior, MultivariateGaussian):
            raise ValueError("prior must be instance of MultivariateGaussian")
        self._prior = prior
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
        return self._prior_condition_hess_op.nvars()

    def __repr__(self) -> str:
        return "{0}(nvars={1})".format(self.__class__.__name__, self.nvars())

    def compute(self, noversampling: int = 10, npower_iters: int = 1):
        svd_solver = SymmetricMatrixDoublePassRandomizedSVD(
            self._prior_condition_hess_op, noversampling, npower_iters
        )
        self._Ur, self._Sr = svd_solver.compute(self._rank)[:2]
        P = 1 / self._bkd.sqrt(self._Sr + 1)
        self._post_cov_sqrt = self._prior_condition_hess_op._prior_sqrt.apply(
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
    """
    Compute a low-rank approximation of a Gaussian posterior.

    This class computes a low-rank approximation of the posterior covariance
    matrix for a Gaussian posterior using the Laplace approximation.
    It is particularly useful for scenarios where the posterior covariance
    matrix is large and dense, but only a small number of directions in
    parameter space are informed by data, which can be leveraged to produce
    a low-rank representation for efficient computation.
    """

    def __init__(
        self,
        prior: DenseCholeskyMultivariateGaussian,
        hess_mat: Array,
        rank: int,
    ):
        """
        Initialize the DenseMatrixLaplacePosteriorLowRankApproximation class.

        Parameters
        ----------
        prior : DenseCholeskyMultivariateGaussian
            The prior distribution, represented as a dense Gaussian with
            Cholesky decomposition.
        hess_mat : Array
            The Hessian matrix of the log-likelihood, used to approximate the
            posterior covariance.
        rank : int
            The rank of the low-rank approximation.

        Raises
        ------
        ValueError
            If `prior` is not an instance of
            `DenseCholeskyMultivariateGaussian`.
        """
        # mainly useful for testing
        if not isinstance(prior, DenseCholeskyMultivariateGaussian):
            raise ValueError(
                "prior must be an instance of DenseCholeskyMultivariateGaussian"
            )
        self._hess_mat = hess_mat
        super().__init__(
            prior,
            PriorConditionedHessianMatVecOperator(
                prior._cov_sqrt, self._apply_hessian
            ),
            rank,
        )

    def _apply_hessian(self, vecs: Array) -> Array:
        return self._hess_mat @ vecs


class GaussianPushForward:
    r"""
    Compute the mean and covariance of a Gaussian distribution when it is
    pushed forward through a linear model.

    A linear transformation applied to a Gaussian distribution results in
    another Gaussian distribution.
    This class computes the mean and covariance of the resulting Gaussian
    distribution after applying a linear transformation.

    Notes
    -----
    - The transformation applied to the original Gaussian is:
      .. math::
          y = A z + b
      where:
      - `z ~ N(x, \Sigma)` is the original Gaussian distribution.
      - `A` is the transformation matrix.
      - `b` is a constant vector.
    - The resulting Gaussian distribution is:
      .. math::
          y ~ N(Ax + b, A \Sigma A^T)
    """

    def __init__(
        self,
        matrix: Array,
        mean: Array,
        cov: Array,
        vec: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the GaussianPushForward class.

        Parameters
        ----------
        matrix : Array of shape (nqoi, nvars)
            The matrix representing the linear transformation.
        mean : Array of shape (nvars, 1)
            The mean of the original Gaussian distribution.
        cov : Array of shape (nvars, nvars)
            The covariance of the original Gaussian distribution.
        vec : Array of shape (nqoi, 1), optional
            A constant vector added to the linear transformation.
            Defaults to a zero vector.
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.

        Raises
        ------
        ValueError
            If the shapes of `mean`, `cov`, or `vec` are invalid.
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
        """
        Returns the number of quantities of interest (nqoi).

        Returns
        -------
        nqoi : int
            Number of quantities of interest (nqoi).
        """
        return self._nqoi

    def nvars(self) -> int:
        """
        Returns the number of variables in the original Gaussian distribution.

        Returns
        -------
        int
            Number of variables in the original Gaussian distribution.
        """
        return self._nvars

    def _compute(self) -> Array:
        r"""
        Compute the mean and covariance of the resulting Gaussian distribution.

        Notes
        -----
        - The mean is computed as:
          .. math::
              \text{mean} = A x + b
        - The covariance is computed as:
          .. math::
              \text{covariance} = A \Sigma A^T
        """
        self._pushforward_mean = self._mat @ self._mean + self._vec
        self._pushforward_cov = self._mat @ self._cov @ self._mat.T

    def mean(self) -> Array:
        """
        Returns the mean of the resulting Gaussian distribution.

        Returns
        -------
        mean : Array
            Mean of the resulting Gaussian distribution.

        Raises
        ------
        RuntimeError
            If `_compute()` has not been called.
        """
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_mean

    def covariance(self) -> Array:
        """
        Returns the covariance of the resulting Gaussian distribution.

        Returns
        -------
        cov : Array
            Covariance of the resulting Gaussian distribution.

        Raises
        ------
        RuntimeError
            If `_compute()` has not been called.
        """
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> DenseCholeskyMultivariateGaussian:
        """
        Returns the resulting Gaussian distribution as a
        `DenseCholeskyMultivariateGaussian` object.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            The resulting Gaussian distribution.
        """
        return DenseCholeskyMultivariateGaussian(
            self.mean(), self.covariance()
        )


class DenseMatrixLaplaceApproximationForPrediction:
    """
    Compute the Gaussian posterior, using a Gaussian conujgate prior, and
    the posterior pushforward through a linear model by solving a
    generalized eigenvalue problem.
    """

    def __init__(
        self,
        obs_matrix: Array,
        pred_matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        obs_noise_cov: Array,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the DenseMatrixLaplaceApproximationForPrediction class.

        Parameters
        ----------
        obs_matrix : Array of shape (nobs, nvars)
            The matrix representing the linear transformation for observations.
        pred_matrix : Array of shape (npred, nvars)
            The matrix representing the linear transformation for predictions.
        prior_mean : Array of shape (nvars, 1)
            The mean of the Gaussian prior distribution.
        prior_cov : Array of shape (nvars, nvars)
            The covariance of the Gaussian prior distribution.
        obs_noise_cov : Array of shape (nobs, nobs)
            The covariance of the observational noise.
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.
        """
        self._bkd = backend
        self._obs_matrix = obs_matrix
        self._pred_matrix = pred_matrix
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._obs_noise_cov = obs_noise_cov

    def compute(self, obs: Array):
        """
        Compute the predictive mean and covariance given observations.

        Parameters
        ----------
        obs : Array of shape (nobs, 1)
            Observations used to compute the predictive mean and covariance.
        """
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
        # evecs = evecs[:, ::-1]
        # evals = evals[::-1]
        evecs = self._bkd.flip(self._bkd.asarray(evecs), axis=(1,))
        evals = self._bkd.flip(self._bkd.asarray(evals))
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
        """
        Returns the posterior pushforward mean.

        Returns
        -------
        Array
            Predictive mean.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_mean

    def covariance(self) -> Array:
        """
        Returns the posterior pushforward covariance.

        Returns
        -------
        Array
            Predictive covariance.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> DenseCholeskyMultivariateGaussian:
        """
        Returns the posterior pushforward Gaussian distribution as
        a `DenseCholeskyMultivariateGaussian` object.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Predictive Gaussian distribution.
        """
        return DenseCholeskyMultivariateGaussian(
            self.mean(), self.covariance()
        )


# TODO put laplace in this format
class ConjugatePriorPosterior(ABC):
    """
    Abstract base class for computing the posterior distribution of a
    conjugate prior model.

    When the prior and posterior belong to the same parametric family,
    the prior is said to be conjugate for the likelihood.
    The posterior distribution is computed analytically using the observations.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the ConjugatePriorPosterior class.

        Parameters
        ----------
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.
        """
        self._bkd = backend

    def _set_observations(self, obs: Array):
        """
        Set the observations for the model.

        Parameters
        ----------
        obs : Array
            Observations used to compute the posterior distribution.

        Raises
        ------
        ValueError
            If the shape of `obs` does not match the expected number of
            observations.
        """
        if obs.ndim != 2 or obs.shape[0] != self.nobs():
            raise ValueError("obs has the wrong shape")
        self._obs = obs

    def nvars(self) -> int:
        """
        Returns the number of variables in the model.

        Returns
        -------
        int
            Number of variables in the model.
        """
        return self._nvars

    def nobs(self) -> int:
        """
        Returns the number of observations.

        Returns
        -------
        int
            Number of observations.
        """
        return self._nobs

    @abstractmethod
    def _compute(self, obs: Array):
        """
        Abstract method for computing the posterior distribution.

        Parameters
        ----------
        obs : Array
            Observations used to compute the posterior distribution.

        """
        raise NotImplementedError

    def compute(self, obs: Array):
        """
        Computes the posterior distribution given observations.

        Parameters
        ----------
        obs : Array
            Observations used to compute the posterior distribution.
        """
        self._set_observations(obs)
        self._compute(obs)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nobs={2})".format(
            self.__class__.__name__, self.nvars(), self.nobs()
        )


class BetaConjugatePriorPosterior(ConjugatePriorPosterior):
    r"""
    Compute the posterior distribution for a Beta conjugate prior model.

    This class models a Beta conjugate prior for a Bernoulli likelihood.

    Likelihood
    ----------
    The likelihood is given by:
    .. math::
        p(y|x) = \Chi_{x \in \{0,1\}} x^y (1-x)^{1-y}
    where:
    - \Chi_{x \in \{0,1\}} is an indicator function equal to 1 if
      \(x \in \{0,1\}\), and 0 otherwise.
    - \(y\) is the observation (binary: 0 or 1).

    Prior
    -----
    The prior assigned to \(x\) is a Beta distribution:
    .. math::
        p(x) \sim \text{Beta}(\alpha, \beta)
    """

    def __init__(
        self,
        shape_args: Array,
        nexperiments: int,
        backend: BackendMixin = NumpyMixin,
    ):
        r"""
        Initialize the BetaConjugatePriorPosterior class.

        Parameters
        ----------
        shape_args : Array of shape (2, nvars)
            Shape parameters (\(\alpha\) and \(\beta\)) of the Beta prior
            distribution.
        nexperiments : int
            Number of experiments (observations).
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.

        Raises
        ------
        ValueError
            If `shape_args` is not a 2D array with two rows.
        ValueError
            If any shape parameter (\(\alpha\) or \(\beta\)) is less than 1.
        NotImplementedError
            If the number of variables is not equal to 1
            (only univariate posterior is supported).
        """
        super().__init__(backend)
        if shape_args.ndim != 2 or shape_args.shape[0] != 2:
            raise ValueError("shapes must be a 2D array with two rows")
        self._prior_shapes = shape_args
        if self._prior_shapes[0] < 1:
            raise ValueError("shape_args[0] must be >= 1")
        if self._prior_shapes[1] < 1:
            raise ValueError("shape_args[1] must be >= 1")
        self._nvars = shape_args.shape[0]
        if self._nvars == 1:
            raise NotImplementedError(
                "only univariate posterior is currently suported"
            )
        self._nobs = 1
        self._nexperiments = nexperiments

    def _compute(self, obs: Array):
        r"""
        Compute the posterior distribution given observations.

        Parameters
        ----------
        obs : Array
            Observations used to compute the posterior distribution.

        Raises
        ------
        ValueError
            If observations are not binary (0 or 1).

        Notes
        -----
        - The posterior shape parameters (\(\alpha'\) and \(\beta'\))
          are computed as:
          .. math::
              \alpha' = \alpha + \sum_i y_i
              \beta' = \beta + n_{\text{experiments}} - \sum y_i
        """
        if self._bkd.any((self._obs != 1) & (self._obs != 0.0)):
            raise ValueError("obs must be zero or one")

        self._posterior_shapes = self._bkd.stack(
            (
                self._prior_shapes[0] + self._bkd.sum(self._obs),
                self._prior_shapes[1]
                + self._nexperiments
                - self._bkd.sum(self._obs),
            ),
            axis=0,
        )

    def posterior_variable(self) -> IndependentMarginalsVariable:
        """
        Returns the posterior distribution as an
        `IndependentMarginalsVariable` object with BetaMarginal.

        Returns
        -------
        IndependentMarginalsVariable
            Posterior distribution.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_posterior_shapes"):
            raise RuntimeError("must call compute")
        marginals = [
            BetaMarginal(*shape, 0.0, 1.0, backend=self._bkd)
            for shape in self._posterior_shapes.T
        ]
        return IndependentMarginalsVariable(marginals, backend=self._bkd)

    def evidence(self) -> float:
        """
        Computes the evidence (marginal likelihood) of the model.

        Returns
        -------
        float
            Evidence (marginal likelihood).
        """
        shape = self._prior_shapes[:, 0]
        log_evidence = -log_beta_function(*shape, self._bkd)
        nzeros = self._bkd.where(self._obs[0] == 0.0)[0].shape[0]
        nones = self._bkd.where(self._obs[0] == 1.0)[0].shape[0]
        log_evidence += log_beta_function(
            shape[0] + nones, shape[1] + nzeros, self._bkd
        )
        return self._bkd.exp(log_evidence)


class DirichletConjugatePriorPosterior(ConjugatePriorPosterior):
    r"""
    Compute the posterior distribution for a Dirichlet conjugate prior model.

    This class models a Dirichlet conjugate prior for a multinomial likelihood.

    Likelihood
    ----------
    The likelihood is given by the multinomial distribution:
    .. math::
        p(x|\theta) = \frac{M!}{x_1!...x_K!} \prod_{k=1}^K \theta_k^{x_k}
    where:
    - \(x_k\) is the count of category \(k\) in the observation.
    - \(M = \sum_{k=1}^K x_k\) is the total number of trials.

    Prior
    -----
    The prior assigned to \(\theta\) is a Dirichlet distribution:
    .. math::
        p(\theta) \sim \text{Dirichlet}(\alpha_1, \alpha_2, ..., \alpha_K)

    Posterior
    ---------
    The posterior distribution is also a Dirichlet distribution:
    .. math::
        p(\theta|x) \sim \text{Dirichlet}(\alpha_1 + \sum_{n=1}^N x_{n,1}, ..., \alpha_K + \sum_{n=1}^N x_{n,K})
    """

    def __init__(
        self,
        shape_args: Array,
        nexperiments: int,
        ntrials: int,
        noptions: int,
        backend: BackendMixin = NumpyMixin,
    ):
        r"""
        Initialize the DirichletConjugatePriorPosterior class.

        Parameters
        ----------
        shape_args : Array of shape (noptions,)
            Shape parameters (\(\alpha_1, \alpha_2, ..., \alpha_K\)) of the
            Dirichlet prior distribution.
        nexperiments : int
            Number of experiments (observations).
        ntrials : int
            Number of trials per experiment.
        noptions : int
            Number of categories (options) in the multinomial distribution.
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to `NumpyMixin`.

        Raises
        ------
        ValueError
            If `shape_args` is not a 1D array.
        ValueError
            If any shape parameter (\(\alpha_k\)) is less than 1.
        NotImplementedError
            If the number of variables is less than 2
            (only multivariate posterior is supported).
        """
        super().__init__(backend)
        if shape_args.ndim != 1:
            raise ValueError("shapes must be a 1D array")
        self._prior_shapes = shape_args
        if self._bkd.any(self._prior_shapes[0] < 1):
            raise ValueError("shape_args[0] must be >= 1")
        self._nvars = shape_args.shape[0]
        if self._nvars < 2:
            raise NotImplementedError("nvars must be >= 2")
        self._nobs = noptions
        self._ntrials = self._bkd.array(ntrials)
        self._nexperiments = nexperiments

    def _set_observations(self, obs: Array):
        r"""
        Set the observations for the model.

        Parameters
        ----------
        obs : Array of shape (noptions, nexperiments)
            Observations used to compute the posterior distribution.

        Raises
        ------
        ValueError
            If the shape of `obs` does not match the expected dimensions.
        ValueError
            If any column of `obs` does not sum to the number of trials
            (\(M\)).
        """
        if obs.shape != (self._nobs, self._nexperiments):
            raise ValueError(
                "obs must be a 2D array with shape "
                "{0} but had shape{1}".format(
                    (self._nobs, self._nexperiments), obs.shape
                )
            )
        if self._bkd.any(obs.sum(axis=0) != self._ntrials):
            raise ValueError(
                "each column of obs must sum to {0}".format(self._ntrials)
            )
        self._obs = obs

    def _compute(self, obs: Array):
        # prior is
        # p(\theta) ~ Dirichlet(a1,\ldots, aK) K=noptions
        # Each obs x=(x_1,...,x_K) represents an experiment that extracts
        # M balls of K different colors from a bag. M=ntrials
        # x_k is the number of extraced balls of color k and
        # \sum_{k=1}^K x_k = M
        # likelihood is
        # p(x|theta) = M!/(x_1!...x_K!)\theta_1^{x_1}...\theta_K^{x_K}
        #            = \Gamma(\sum_k x_k+1)/(\prod_k \Gamma(x_k+1))\prod \theta_k^{x_k}
        # posterior is
        # p(\theta|x)~Dirichlet(a_1+\sum_{n=1}^N x_{n,1},\ldots,a_K+\sum_{n=1}^N x_{n,K})
        self._posterior_shapes = self._prior_shapes + self._bkd.sum(
            obs, axis=1
        )

    def posterior_variable(self) -> IndependentMarginalsVariable:
        """
        Returns the posterior distribution as an `IndependentMarginalsVariable` object.

        Returns
        -------
        DirichletVariable
            Posterior distribution.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called.
        """
        if not hasattr(self, "_posterior_shapes"):
            raise RuntimeError("must call compute")
        return DirichletVariable(self._posterior_shapes, self._bkd)

    def _log_multivariate_beta_function(self, shapes: Array) -> float:
        r"""
        Compute the logarithm of the multivariate Beta function.

        Parameters
        ----------
        shapes : Array
            Shape parameters (\(\alpha_1, \alpha_2, ..., \alpha_K\)).

        Returns
        -------
        float
            Logarithm of the multivariate Beta function.
        """
        alpha0 = self._bkd.sum(shapes)
        return self._bkd.sum(
            self._bkd.hstack([self._bkd.gammaln(shape) for shape in shapes])
        ) - self._bkd.gammaln(alpha0)

    def evidence(self) -> float:
        """
        Compute the evidence (marginal likelihood) of the model.

        Returns
        -------
        float
            Evidence (marginal likelihood).
        """
        # let \theta be parameters and x data
        # to compute evidence E find equate constants of
        # p(\theta)p(x|\theta) and p(\theta|x)
        # such that p(\theta)p(x|\theta)/E = p(\theta|x)
        # Let W_i = (x_{i1}!)*...*(x_{iK}!) where x_i is the ith observation
        # and a be shapes of prior and b shapes of post
        # and Q=\theta^{b_1-1}...\theta^{b_K-1}
        # p(x|\theta)p(\theta) = (M!)^nexps/(Beta[a]*prod_i(W_i))*Q
        # p(\theta|x)=Q/Beta[b]
        # E=p(\theta)p(x|\theta)/p(\theta|x)/
        # E=(M!)^nobs/(Beta[a]*prod_i(W_i))*Q/(Q/Beta[b])
        # E=((M!)^nobs*Beta[b]/(Beta[a]*prod_i(W_i))
        log_beta_prior = self._log_multivariate_beta_function(
            self._prior_shapes
        )
        log_beta_post = self._log_multivariate_beta_function(
            self._posterior_shapes
        )
        log_evidence = (
            # (M!)^nexps
            self._nexperiments * self._bkd.gammaln(self._ntrials + 1)
            # Beta[b]
            + log_beta_post
            # 1/Beta[a]
            - log_beta_prior
            # 1/prod_i(W_i)
            - sum(
                self._bkd.sum(self._bkd.gammaln(self._obs[:, ii] + 1))
                for ii in range(self._nexperiments)
            )
        )
        return self._bkd.exp(log_evidence)
