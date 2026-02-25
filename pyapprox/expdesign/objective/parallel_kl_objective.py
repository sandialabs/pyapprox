"""
Parallel KL-OED objective function.

Provides a parallelized version of the KL-OED objective by using
ParallelGaussianOEDInnerLoopLikelihood for evidence computation.
"""

from typing import Generic, Optional

from pyapprox.expdesign.evidence import LogEvidence
from pyapprox.expdesign.likelihood import (
    ParallelGaussianOEDInnerLoopLikelihood,
)
from pyapprox.interface.parallel import ParallelConfig
from pyapprox.util.backends.protocols import Array, Backend


class ParallelKLOEDObjective(Generic[Array]):
    """
    Parallel KL-based OED objective (expected information gain).

    Same computation as KLOEDObjective but uses parallel likelihood
    evaluation for the inner loop.

    Parameters
    ----------
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    outer_shapes : Array
        Model outputs for outer samples. Shape: (nobs, nouter)
    latent_samples : Array
        Latent noise samples for reparameterization. Shape: (nobs, nouter)
    inner_shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    bkd : Backend[Array]
        Computational backend.
    parallel_config : ParallelConfig, optional
        Parallel execution configuration.
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)
    """

    def __init__(
        self,
        noise_variances: Array,
        outer_shapes: Array,
        latent_samples: Array,
        inner_shapes: Array,
        bkd: Backend[Array],
        parallel_config: Optional[ParallelConfig] = None,
        outer_quad_weights: Optional[Array] = None,
        inner_quad_weights: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._parallel_config = parallel_config or ParallelConfig(
            backend="sequential", n_jobs=1
        )

        self._nobs = noise_variances.shape[0]
        self._nouter = outer_shapes.shape[1]
        self._ninner = inner_shapes.shape[1]

        # Store shapes
        self._outer_shapes = outer_shapes
        self._latent_samples = latent_samples
        self._inner_shapes = inner_shapes
        self._noise_variances = noise_variances

        # Create parallel inner likelihood
        self._inner_loglike = ParallelGaussianOEDInnerLoopLikelihood(
            noise_variances, bkd, self._parallel_config
        )
        self._inner_loglike.set_shapes(inner_shapes)

        # Create outer likelihood (non-parallel, it's fast)
        self._outer_loglike = self._inner_loglike.create_outer_loop_likelihood()
        self._outer_loglike.set_shapes(outer_shapes)

        # Set quadrature weights
        if outer_quad_weights is None:
            outer_quad_weights = bkd.ones((self._nouter,)) / self._nouter
        if inner_quad_weights is None:
            inner_quad_weights = bkd.ones((self._ninner,)) / self._ninner

        self._outer_quad_weights = outer_quad_weights
        self._inner_quad_weights = inner_quad_weights

        self._log_evidence: Optional[LogEvidence[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of design variables (= nobs)."""
        return self._nobs

    def nqoi(self) -> int:
        """Number of outputs (= 1 for scalar objective)."""
        return 1

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        return self._ninner

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        return self._nouter

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Update parallel configuration."""
        self._parallel_config = config
        self._inner_loglike.set_parallel_config(config)

    def _generate_observations(self, design_weights: Array) -> Array:
        """Generate artificial observations using reparameterization trick."""
        base_var = self._noise_variances
        effective_std = self._bkd.sqrt(base_var[:, None] / design_weights)
        return self._outer_shapes + effective_std * self._latent_samples

    def _update_observations(self, design_weights: Array) -> None:
        """Update observations for the current design weights."""
        obs = self._generate_observations(design_weights)

        self._inner_loglike.set_observations(obs)
        self._outer_loglike.set_observations(obs)

        self._inner_loglike.set_latent_samples(self._latent_samples)
        self._outer_loglike.set_latent_samples(self._latent_samples)

        self._log_evidence = LogEvidence(
            self._inner_loglike, self._inner_quad_weights, self._bkd
        )

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate the KL-OED objective (parallel).

        Returns -EIG for minimization.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value (-EIG). Shape: (1, 1)
        """
        self._update_observations(design_weights)

        log_like_true = self._outer_loglike(design_weights)
        log_evidence = self._log_evidence(design_weights)

        diff = log_like_true - log_evidence
        eig = self._bkd.sum(self._outer_quad_weights * diff[0])

        return self._bkd.reshape(-eig, (1, 1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of objective w.r.t. design weights (parallel).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nobs)
        """
        self._update_observations(design_weights)

        jac_log_like_true = self._outer_loglike.jacobian(design_weights)
        jac_log_evidence = self._log_evidence.jacobian(design_weights)

        jac_diff = jac_log_like_true - jac_log_evidence

        jac_eig = self._bkd.sum(self._outer_quad_weights[:, None] * jac_diff, axis=0)

        return self._bkd.reshape(-jac_eig, (1, self._nobs))

    def expected_information_gain(self, design_weights: Array) -> float:
        """
        Compute expected information gain (positive value).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        float
            Expected information gain.
        """
        neg_eig = self(design_weights)
        return -float(self._bkd.to_numpy(neg_eig)[0, 0])
