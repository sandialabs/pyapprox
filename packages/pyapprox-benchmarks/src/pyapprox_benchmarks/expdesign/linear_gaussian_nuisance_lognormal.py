r"""Linear-Gaussian OED benchmark with nuisances and a lognormal QoI.

The inputs :math:`\xi = (m, a, b)` stack a parameter :math:`m`, an
observation nuisance :math:`a` and a prediction nuisance :math:`b`, with
independent zero-mean Gaussian priors
:math:`m \sim N(0, \sigma_m^2 I)`, :math:`a \sim N(0, \sigma_a^2 I)` and
:math:`b \sim N(0, \sigma_b^2 I)`. Observations and QoI are

.. math::

    y = H\xi + e = Am + B_a a + e, \qquad e \sim N(0, \sigma_e^2 I),

    q = \exp(F\xi) = \exp(Rm + S_a a + S_b b),

so :math:`H = [A, B_a, 0]` and :math:`F = [R, S_a, S_b]`. :math:`B_a`
carries the observation nuisance into the data, :math:`S_a` carries it into
the QoI, and :math:`S_b` adds a prediction-only floor that no data can
reduce. Stacking the nuisances into :math:`\xi` marginalizes them exactly,
so every ground-truth method is a closed form.
"""

from typing import Generic

import numpy as np
from pyapprox.expdesign.analytical import (
    LogNormalMGBlocks,
    lognormal_goal_mg_blocks,
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
    relaxed_lognormal_expected_variance,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.problems.inverse import GaussianInferenceProblem
from pyapprox_benchmarks.problems.oed import PredictionOEDProblem


class LinearGaussianNuisanceLognormalBenchmark(Generic[Array]):
    """Prediction OED benchmark with nuisances and q = exp(F xi).

    Ground truth at any weights in [0, 1]^nobs comes from the relaxed
    linear-Gaussian closed forms. Weights enter through the blended
    observation with reference variances equal to the noise variances, so
    for binary weights the values are those of the selected sensors.

    Parameters
    ----------
    problem : PredictionOEDProblem[Array]
        The prediction OED problem over xi = (m, a, b).
    prior : DenseCholeskyMultivariateGaussian[Array]
        The Gaussian prior over xi, the one ``problem`` holds.
    obs_mat : Array
        ``H = [A, B_a, 0]``. Shape: (nobs, nvars)
    qoi_mat : Array
        ``F = [R, S_a, S_b]``. Shape: (nqoi, nvars)
    nparams : int
        Size of the parameter block ``m``.
    nobs_nuisance : int
        Size of the observation nuisance ``a``.
    npred_nuisance : int
        Size of the prediction nuisance ``b``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        problem: PredictionOEDProblem[Array],
        prior: DenseCholeskyMultivariateGaussian[Array],
        obs_mat: Array,
        qoi_mat: Array,
        nparams: int,
        nobs_nuisance: int,
        npred_nuisance: int,
        bkd: Backend[Array],
    ) -> None:
        nvars = nparams + nobs_nuisance + npred_nuisance
        if obs_mat.shape[1] != nvars or qoi_mat.shape[1] != nvars:
            raise ValueError(
                f"obs_mat and qoi_mat must have {nvars} columns, got "
                f"{obs_mat.shape[1]} and {qoi_mat.shape[1]}"
            )
        self._problem = problem
        self._prior = prior
        self._obs_mat = obs_mat
        self._qoi_mat = qoi_mat
        self._nparams = nparams
        self._nobs_nuisance = nobs_nuisance
        self._npred_nuisance = npred_nuisance
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> PredictionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    def obs_map(self) -> FunctionProtocol[Array]:
        """Map xi (nvars, nsamples) to noise-free data (nobs, nsamples)."""
        return self._problem.obs_map()

    def qoi_map(self) -> FunctionProtocol[Array]:
        """Map xi (nvars, nsamples) to the QoI (nqoi, nsamples)."""
        return self._problem.qoi_map()

    def prior(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """Gaussian prior over xi = (m, a, b)."""
        return self._prior

    def prior_mean(self) -> Array:
        """Prior mean of xi. Shape: (nvars, 1)"""
        return self.prior().mean()

    def prior_covariance(self) -> Array:
        """Prior covariance of xi. Shape: (nvars, nvars)"""
        return self.prior().covariance()

    def noise_covariance(self) -> Array:
        """Noise covariance. Shape: (nobs, nobs)"""
        return self._bkd.diag(self._problem.noise_variances())

    def obs_matrix(self) -> Array:
        """``H = [A, B_a, 0]``. Shape: (nobs, nvars)"""
        return self._obs_mat

    def qoi_matrix(self) -> Array:
        """``F = [R, S_a, S_b]``. Shape: (nqoi, nvars)"""
        return self._qoi_mat

    def nparams(self) -> int:
        """Size of the parameter block ``m``."""
        return self._nparams

    def nobs_nuisance(self) -> int:
        """Size of the observation nuisance ``a``."""
        return self._nobs_nuisance

    def npred_nuisance(self) -> int:
        """Size of the prediction nuisance ``b``."""
        return self._npred_nuisance

    def evaluate_both(self, samples: Array) -> tuple[Array, Array]:
        """Evaluate noise-free data and QoI from one set of samples.

        Parameters
        ----------
        samples : Array
            Samples of xi. Shape: (nvars, nsamples)

        Returns
        -------
        tuple[Array, Array]
            Data of shape (nobs, nsamples) and QoI of shape
            (nqoi, nsamples).
        """
        return self.obs_map()(samples), self.qoi_map()(samples)

    # --- Ground truth ---

    def exact_goal_expected_posterior_variance(self, weights: Array) -> Array:
        """Exact ``E_z[Var(q_i | z)]`` at the weights.

        Parameters
        ----------
        weights : Array
            Design weights in [0, 1]. Shape: (nobs, 1)

        Returns
        -------
        Array
            Expected posterior variance of each QoI. Shape: (nqoi, 1)
        """
        return relaxed_lognormal_expected_variance(
            self._qoi_mat,
            self._obs_mat,
            self.prior_mean(),
            self.prior_covariance(),
            self.noise_covariance(),
            weights,
            self._bkd,
        )

    def exact_goal_eig(self, weights: Array) -> Array:
        """Exact expected information gain about ``q`` at the weights.

        Equal to the EIG about ``log q = F xi``, which is linear-Gaussian.

        Parameters
        ----------
        weights : Array
            Design weights in [0, 1]. Shape: (nobs, 1)

        Returns
        -------
        Array
            The expected information gain. Shape: (1, 1)
        """
        return relaxed_linear_target_eig(
            self._qoi_mat,
            self._obs_mat,
            self.prior_covariance(),
            self.noise_covariance(),
            weights,
            self._bkd,
        )

    def exact_mg_blocks(self) -> LogNormalMGBlocks[Array]:
        """Exact means and covariances of ``(q, H xi)`` under the prior."""
        return lognormal_goal_mg_blocks(
            self._obs_mat,
            self._qoi_mat,
            self.prior_mean(),
            self.prior_covariance(),
            self._bkd,
        )

    def exact_param_posterior_covariance(self, weights: Array) -> Array:
        """Exact posterior covariance of the parameter ``m`` at the weights.

        The parameter target is linear-Gaussian, so the moment-Gaussian
        approximation of it is exact.

        Parameters
        ----------
        weights : Array
            Design weights in [0, 1]. Shape: (nobs, 1)

        Returns
        -------
        Array
            Shape: (nparams, nparams)
        """
        nvars = self._obs_mat.shape[1]
        param_mat = self._bkd.eye(nvars)[: self._nparams]
        return relaxed_linear_target_covariance(
            param_mat,
            self._obs_mat,
            self.prior_covariance(),
            self.noise_covariance(),
            weights,
            self._bkd,
        )

    # --- Misspecified variant ---

    def nuisance_free(self) -> "LinearGaussianNuisanceLognormalBenchmark[Array]":
        """The model that ignores the nuisances, fixing them at their means.

        Its inputs are ``m`` alone, with ``y = A m + e`` and
        ``q = exp(R m)``; the nuisance means are zero, so they add no
        offset. Its ground truth is what a designer ignoring the nuisances
        believes; scoring its designs with this benchmark's methods gives
        their true performance.
        """
        nm = self._nparams
        return _build_benchmark(
            self._obs_mat[:, :nm],
            self._qoi_mat[:, :nm],
            self.prior_covariance()[:nm, :nm],
            self._problem.noise_variances(),
            nm,
            0,
            0,
            self._bkd,
        )


def _build_benchmark(
    obs_mat: Array,
    qoi_mat: Array,
    prior_cov: Array,
    noise_variances: Array,
    nparams: int,
    nobs_nuisance: int,
    npred_nuisance: int,
    bkd: Backend[Array],
) -> LinearGaussianNuisanceLognormalBenchmark[Array]:
    """Assemble the benchmark from its matrices and zero-mean prior."""
    nobs, nvars = obs_mat.shape
    nqoi = qoi_mat.shape[0]

    def _obs_fun(samples: Array) -> Array:
        return bkd.dot(obs_mat, samples)

    def _qoi_fun(samples: Array) -> Array:
        return bkd.exp(bkd.dot(qoi_mat, samples))

    prior_mean = bkd.zeros((nvars, 1))
    prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, bkd)
    inference = GaussianInferenceProblem(
        obs_map=FunctionFromCallable(nobs, nvars, _obs_fun, bkd),
        prior=prior,
        noise_variances=noise_variances,
        bkd=bkd,
        prior_mean=prior_mean,
        prior_covariance=prior_cov,
    )
    problem = PredictionOEDProblem(
        inference,
        FunctionFromCallable(nqoi, nvars, _qoi_fun, bkd),
        bkd.linspace(0.0, nobs - 1.0, nobs),
        bkd,
    )
    return LinearGaussianNuisanceLognormalBenchmark(
        problem,
        prior,
        obs_mat,
        qoi_mat,
        nparams,
        nobs_nuisance,
        npred_nuisance,
        bkd,
    )


def build_linear_gaussian_nuisance_lognormal_benchmark(
    nobs: int,
    nparams: int,
    nobs_nuisance: int,
    npred_nuisance: int,
    nqoi: int,
    bkd: Backend[Array],
    noise_std: float = 0.2,
    prior_std: float = 0.5,
    obs_nuisance_std: float = 0.5,
    pred_nuisance_std: float = 0.25,
    seed: int = 0,
) -> LinearGaussianNuisanceLognormalBenchmark[Array]:
    """Build the benchmark with seeded random ``A, B_a, R, S_a, S_b``.

    Entries are standard normal, divided by the square root of the block
    width, so each block's contribution to a datum or log-QoI has variance
    of order its prior variance whatever the sizes.

    Parameters
    ----------
    nobs : int
        Number of candidate observations.
    nparams : int
        Size of the parameter ``m``.
    nobs_nuisance : int
        Size of the observation nuisance ``a``.
    npred_nuisance : int
        Size of the prediction nuisance ``b``.
    nqoi : int
        Number of QoIs.
    bkd : Backend[Array]
        Computational backend.
    noise_std : float
        Observation noise standard deviation ``sigma_e``.
    prior_std : float
        Prior standard deviation ``sigma_m`` of the parameter.
    obs_nuisance_std : float
        Prior standard deviation ``sigma_a``, positive. To remove the
        nuisance, set ``nobs_nuisance = 0``.
    pred_nuisance_std : float
        Prior standard deviation ``sigma_b``, positive. To remove the
        floor, set ``npred_nuisance = 0``.
    seed : int
        Seed for the random matrices.

    Returns
    -------
    LinearGaussianNuisanceLognormalBenchmark[Array]
        The configured benchmark.
    """
    stds = (noise_std, prior_std, obs_nuisance_std, pred_nuisance_std)
    if min(stds) <= 0.0:
        raise ValueError(f"standard deviations must be positive, got {stds}")
    rng = np.random.default_rng(seed)

    def block(nrows: int, ncols: int) -> np.ndarray:
        scale = 1.0 / float(np.sqrt(max(ncols, 1)))
        return scale * rng.standard_normal((nrows, ncols))

    amat = block(nobs, nparams)
    ba = block(nobs, nobs_nuisance)
    rmat = block(nqoi, nparams)
    sa = block(nqoi, nobs_nuisance)
    sb = block(nqoi, npred_nuisance)
    obs_mat = np.hstack([amat, ba, np.zeros((nobs, npred_nuisance))])
    qoi_mat = np.hstack([rmat, sa, sb])
    variances = np.concatenate(
        [
            np.full(nparams, prior_std**2),
            np.full(nobs_nuisance, obs_nuisance_std**2),
            np.full(npred_nuisance, pred_nuisance_std**2),
        ]
    )
    return _build_benchmark(
        bkd.asarray(obs_mat),
        bkd.asarray(qoi_mat),
        bkd.asarray(np.diag(variances)),
        bkd.full((nobs,), noise_std**2),
        nparams,
        nobs_nuisance,
        npred_nuisance,
        bkd,
    )
