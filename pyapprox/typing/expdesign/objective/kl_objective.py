"""
KL-OED objective function.

The KL-OED objective computes the expected information gain (EIG):
    EIG = E_obs[log p(obs | theta_true, design) - log p(obs | design)]
        = E_obs[log_likelihood_true - log_evidence]

This is the KL divergence between posterior and prior, averaged over data.
For minimization, we return -EIG.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
    GaussianOEDOuterLoopLikelihood,
)
from pyapprox.typing.expdesign.evidence import LogEvidence


class KLOEDObjective(Generic[Array]):
    """
    KL-based OED objective (expected information gain).

    Computes the expected information gain for Bayesian experimental design:
        EIG = E_outer[log p(obs | theta_true) - log p(obs | design)]

    The objective returns -EIG for minimization.

    Parameters
    ----------
    inner_likelihood : GaussianOEDInnerLoopLikelihood[Array]
        Inner loop likelihood for evidence computation.
    outer_shapes : Array
        Model outputs for outer samples (shapes/means). Shape: (nobs, nouter)
    latent_samples : Array
        Latent noise samples for reparameterization. Shape: (nobs, nouter)
    inner_shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        inner_likelihood: GaussianOEDInnerLoopLikelihood[Array],
        outer_shapes: Array,
        latent_samples: Array,
        inner_shapes: Array,
        outer_quad_weights: Optional[Array],
        inner_quad_weights: Optional[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._inner_loglike = inner_likelihood
        self._outer_loglike = inner_likelihood.create_outer_loop_likelihood()

        self._nobs = inner_likelihood.nobs()
        self._nouter = outer_shapes.shape[1]
        self._ninner = inner_shapes.shape[1]

        # Store shapes
        self._outer_shapes = outer_shapes
        self._latent_samples = latent_samples
        self._inner_shapes = inner_shapes

        # Set up inner likelihood with shapes (observations set later)
        self._inner_loglike.set_shapes(inner_shapes)

        # Set up outer likelihood with shapes
        self._outer_loglike.set_shapes(outer_shapes)

        # Set quadrature weights
        if outer_quad_weights is None:
            outer_quad_weights = bkd.ones((self._nouter,)) / self._nouter
        if inner_quad_weights is None:
            inner_quad_weights = bkd.ones((self._ninner,)) / self._ninner

        self._outer_quad_weights = outer_quad_weights
        self._inner_quad_weights = inner_quad_weights

        # LogEvidence will be created when observations are set
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

    def _generate_observations(self, design_weights: Array) -> Array:
        """Generate artificial observations using reparameterization trick.

        obs = shapes + sqrt(variance / weights) * latent_samples
        """
        # Get base variance from likelihood
        base_var = self._inner_loglike._base_variances
        # effective_std = sqrt(base_var / weights)
        effective_std = self._bkd.sqrt(base_var[:, None] / design_weights)
        return self._outer_shapes + effective_std * self._latent_samples

    def _update_observations(self, design_weights: Array) -> None:
        """Update observations for the current design weights."""
        obs = self._generate_observations(design_weights)

        # Set observations on both likelihoods
        self._inner_loglike.set_observations(obs)
        self._outer_loglike.set_observations(obs)

        # Set latent samples (must be after observations are set)
        self._inner_loglike.set_latent_samples(self._latent_samples)
        self._outer_loglike.set_latent_samples(self._latent_samples)

        # Create LogEvidence with current inner likelihood
        self._log_evidence = LogEvidence(
            self._inner_loglike, self._inner_quad_weights, self._bkd
        )

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate the KL-OED objective.

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

        # log p(obs | theta_true, design) for each outer sample
        # Shape: (1, nouter)
        log_like_true = self._outer_loglike(design_weights)

        # log p(obs | design) = log(evidence) for each outer sample
        # Shape: (1, nouter)
        log_evidence = self._log_evidence(design_weights)

        # EIG = E_outer[log_like_true - log_evidence]
        # Use quadrature weights for expectation
        diff = log_like_true - log_evidence  # (1, nouter)
        eig = self._bkd.sum(self._outer_quad_weights * diff[0])

        # Return -EIG for minimization
        return self._bkd.reshape(-eig, (1, 1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of objective w.r.t. design weights.

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

        # Jacobians of log_like_true and log_evidence
        # Shape: (nouter, nobs)
        jac_log_like_true = self._outer_loglike.jacobian(design_weights)
        jac_log_evidence = self._log_evidence.jacobian(design_weights)

        # d/dw EIG = E_outer[d/dw (log_like_true - log_evidence)]
        jac_diff = jac_log_like_true - jac_log_evidence  # (nouter, nobs)

        # Apply quadrature weights
        jac_eig = self._bkd.sum(
            self._outer_quad_weights[:, None] * jac_diff, axis=0
        )

        # Return -d/dw EIG for minimization
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

    def evaluate(self, design_weights: Array) -> Array:
        """
        Single-sample evaluation (alias for __call__).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        return self(design_weights)
