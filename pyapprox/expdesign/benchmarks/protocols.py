"""Protocols for OED benchmarks.

Defines the structural interfaces for inference problems, OED problems,
and OED benchmarks. Users can implement custom classes satisfying these
protocols without inheriting from any base class.
"""

from typing import Protocol, runtime_checkable

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend

from .ground_truth import OEDGroundTruth


@runtime_checkable
class BayesianInferenceProblemProtocol(Protocol[Array]):
    """Protocol for a general Bayesian inference problem.

    Defines the minimal interface: observation map, prior (any distribution
    satisfying DistributionProtocol), and noise model. Users can implement
    custom classes satisfying this protocol.
    """

    def bkd(self) -> Backend[Array]: ...
    def obs_map(self) -> FunctionProtocol[Array]: ...
    def prior(self) -> DistributionProtocol[Array]: ...
    def noise_variances(self) -> Array: ...
    def nobs(self) -> int: ...
    def nparams(self) -> int: ...


@runtime_checkable
class GaussianInferenceProblemProtocol(
    BayesianInferenceProblemProtocol[Array], Protocol[Array]
):
    """Protocol for a Gaussian inference problem.

    Extends BayesianInferenceProblemProtocol with prior_mean and
    prior_covariance for conjugate Gaussian analytics.
    """

    def prior(self) -> DenseCholeskyMultivariateGaussian[Array]: ...
    def prior_mean(self) -> Array: ...
    def prior_covariance(self) -> Array: ...


@runtime_checkable
class KLOEDProblemProtocol(
    GaussianInferenceProblemProtocol[Array], Protocol[Array]
):
    """Protocol for a KL-OED problem.

    Inherits GaussianInferenceProblemProtocol and adds design space metadata.
    """

    def inference_problem(
        self,
    ) -> GaussianInferenceProblemProtocol[Array]: ...
    def design_conditions(self) -> Array: ...
    def weight_bounds(self) -> Array: ...


@runtime_checkable
class PredictionOEDProblemProtocol(
    GaussianInferenceProblemProtocol[Array], Protocol[Array]
):
    """Protocol for a prediction OED problem.

    Inherits GaussianInferenceProblemProtocol and adds qoi_map + design space.
    """

    def inference_problem(
        self,
    ) -> GaussianInferenceProblemProtocol[Array]: ...
    def qoi_map(self) -> FunctionProtocol[Array]: ...
    def npred(self) -> int: ...
    def design_conditions(self) -> Array: ...
    def weight_bounds(self) -> Array: ...


@runtime_checkable
class KLOEDBenchmarkProtocol(Protocol[Array]):
    """Protocol for a KL-OED benchmark with ground truth."""

    def problem(self) -> KLOEDProblemProtocol[Array]: ...
    def ground_truth(self) -> OEDGroundTruth: ...
    def exact_eig(self, weights: Array) -> float: ...


@runtime_checkable
class PredictionOEDBenchmarkProtocol(Protocol[Array]):
    """Protocol for a prediction OED benchmark with ground truth."""

    def problem(self) -> PredictionOEDProblemProtocol[Array]: ...
    def ground_truth(self) -> OEDGroundTruth: ...
