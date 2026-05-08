"""KL-OED diagnostics — pure functions of raw sample arrays.

Computes numerical EIG estimates from pre-generated samples.
Sampling is the caller's responsibility.
"""

from typing import Generic, Optional

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.util.backends.protocols import Array, Backend


class KLOEDDiagnostics(Generic[Array]):
    """Numerical EIG estimation from raw sample arrays.

    This class does NOT generate samples. The caller provides
    outer_shapes, latent_samples, and inner_shapes directly
    (e.g. from OEDQuadratureSampler + problem.obs_map()).

    Parameters
    ----------
    noise_variances : Array
        Per-observation noise variances. Shape: (nobs,).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
    ) -> None:
        self._noise_variances = noise_variances
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def compute_numerical_eig(
        self,
        design_weights: Array,
        outer_shapes: Array,
        latent_samples: Array,
        inner_shapes: Array,
        outer_quad_weights: Optional[Array] = None,
        inner_quad_weights: Optional[Array] = None,
    ) -> float:
        """Compute numerical EIG estimate from raw samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1).
        outer_shapes : Array
            obs_map(prior_samples_outer). Shape: (nobs, nouter).
        latent_samples : Array
            Standard normal latent samples. Shape: (nobs, nouter).
        inner_shapes : Array
            obs_map(prior_samples_inner). Shape: (nobs, ninner).
        outer_quad_weights : Array, optional
            Quadrature weights for outer expectation. Shape: (nouter,).
        inner_quad_weights : Array, optional
            Quadrature weights for inner expectation. Shape: (ninner,).

        Returns
        -------
        eig : float
            Numerical EIG estimate.
        """
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            outer_quad_weights,
            inner_quad_weights,
            self._bkd,
        )
        return objective.expected_information_gain(design_weights)
