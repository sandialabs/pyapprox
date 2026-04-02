"""Prediction OED problem: inference problem + qoi_map + design space.

Satisfies GaussianInferenceProblemProtocol via delegation.
"""

from typing import Generic, Optional

from pyapprox.expdesign.benchmarks.protocols import (
    GaussianInferenceProblemProtocol,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.util.backends.protocols import Array, Backend


class PredictionOEDProblem(Generic[Array]):
    """Prediction OED problem: inference problem + qoi_map + design space.

    Composes a GaussianInferenceProblemProtocol with a QoI map and
    design-space metadata. Satisfies GaussianInferenceProblemProtocol
    by delegating all inference methods.

    Parameters
    ----------
    inference_problem : GaussianInferenceProblemProtocol[Array]
        The underlying Gaussian inference problem.
    qoi_map : FunctionProtocol[Array]
        QoI model mapping parameters to predictions.
    design_conditions : Array
        Design conditions. Shape: (nobs, nconditions).
    bkd : Backend[Array]
        Computational backend.
    weight_bounds : Array or None
        Bounds on design weights. Shape: (nobs, 2). Default: [[0, 1], ...].
    """

    def __init__(
        self,
        inference_problem: GaussianInferenceProblemProtocol[Array],
        qoi_map: FunctionProtocol[Array],
        design_conditions: Array,
        bkd: Backend[Array],
        weight_bounds: Optional[Array] = None,
    ) -> None:
        self._inference_problem = inference_problem
        self._qoi_map = qoi_map
        self._design_conditions = design_conditions
        self._bkd = bkd
        if weight_bounds is None:
            nobs = inference_problem.nobs()
            zeros = bkd.zeros((nobs, 1))
            ones = bkd.ones((nobs, 1))
            weight_bounds = bkd.hstack([zeros, ones])
        self._weight_bounds = weight_bounds

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def inference_problem(
        self,
    ) -> GaussianInferenceProblemProtocol[Array]:
        """Get the underlying inference problem."""
        return self._inference_problem

    def qoi_map(self) -> FunctionProtocol[Array]:
        """Get the QoI map."""
        return self._qoi_map

    def npred(self) -> int:
        """Number of prediction QoI outputs."""
        return self._qoi_map.nqoi()

    def design_conditions(self) -> Array:
        """Get design conditions. Shape: (nobs, nconditions)."""
        return self._design_conditions

    def weight_bounds(self) -> Array:
        """Get weight bounds. Shape: (nobs, 2)."""
        return self._weight_bounds

    # --- Delegations to satisfy GaussianInferenceProblemProtocol ---

    def obs_map(self) -> FunctionProtocol[Array]:
        """Get the observation map (delegated)."""
        return self._inference_problem.obs_map()

    def prior(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """Get the prior distribution (delegated)."""
        return self._inference_problem.prior()

    def prior_mean(self) -> Array:
        """Get prior mean (delegated)."""
        return self._inference_problem.prior_mean()

    def prior_covariance(self) -> Array:
        """Get prior covariance (delegated)."""
        return self._inference_problem.prior_covariance()

    def noise_variances(self) -> Array:
        """Get noise variances (delegated)."""
        return self._inference_problem.noise_variances()

    def nobs(self) -> int:
        """Number of observations (delegated)."""
        return self._inference_problem.nobs()

    def nparams(self) -> int:
        """Number of parameters (delegated)."""
        return self._inference_problem.nparams()
