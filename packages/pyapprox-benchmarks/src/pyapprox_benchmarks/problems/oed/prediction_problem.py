"""Prediction OED problem: inference problem + qoi_map + design space.

Composition helper for building a PredictionOEDProblemProtocol from an
existing BayesianInferenceProblemProtocol.  Users with flat classes
(e.g. single PDE solve producing both obs and QoI) can implement the
protocol directly without using this class.

If the wrapped inference problem also satisfies
GaussianInferenceProblemProtocol, the composed result will too
(prior_mean / prior_covariance are forwarded).
"""

from typing import Generic, Optional

from pyapprox.expdesign.protocols.oed import (
    BayesianInferenceProblemProtocol,
    GaussianInferenceProblemProtocol,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class PredictionOEDProblem(Generic[Array]):
    """Prediction OED problem: inference problem + qoi_map + design space.

    Composes a BayesianInferenceProblemProtocol with a QoI map and
    design-space metadata. Satisfies PredictionOEDProblemProtocol
    by delegating all inference methods.

    If the wrapped inference problem is Gaussian (has prior_mean /
    prior_covariance), those methods are forwarded so the composed
    object also satisfies GaussianInferenceProblemProtocol.

    Parameters
    ----------
    inference_problem : BayesianInferenceProblemProtocol[Array]
        The underlying inference problem.
    qoi_map : FunctionProtocol[Array]
        QoI model mapping parameters to predictions.
    design_conditions : Array
        Design conditions. Shape: (ndim, nobs) or (nobs,).
    bkd : Backend[Array]
        Computational backend.
    weight_bounds : Array or None
        Bounds on design weights. Shape: (nobs, 2). Default: [[0, 1], ...].
    """

    def __init__(
        self,
        inference_problem: BayesianInferenceProblemProtocol[Array],
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

        # Forward Gaussian methods if available
        if isinstance(inference_problem, GaussianInferenceProblemProtocol):
            self.prior_mean = inference_problem.prior_mean
            self.prior_covariance = inference_problem.prior_covariance

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def qoi_map(self) -> FunctionProtocol[Array]:
        """Get the QoI map."""
        return self._qoi_map

    def npred(self) -> int:
        """Number of prediction QoI outputs."""
        return self._qoi_map.nqoi()

    def design_conditions(self) -> Array:
        """Get design conditions."""
        return self._design_conditions

    def weight_bounds(self) -> Array:
        """Get weight bounds. Shape: (nobs, 2)."""
        return self._weight_bounds

    # --- Delegations to satisfy BayesianInferenceProblemProtocol ---

    def obs_map(self) -> FunctionProtocol[Array]:
        """Get the observation map (delegated)."""
        return self._inference_problem.obs_map()

    def prior(self) -> DistributionProtocol[Array]:
        """Get the prior distribution (delegated)."""
        return self._inference_problem.prior()

    def noise_variances(self) -> Array:
        """Get noise variances (delegated)."""
        return self._inference_problem.noise_variances()

    def nobs(self) -> int:
        """Number of observations (delegated)."""
        return self._inference_problem.nobs()

    def nparams(self) -> int:
        """Number of parameters (delegated)."""
        return self._inference_problem.nparams()
