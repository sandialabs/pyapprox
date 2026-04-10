"""Protocols for OED benchmarks.

Defines the structural interfaces for inference problems, OED problems,
and OED benchmarks. Users can implement custom classes satisfying these
protocols without inheriting from any base class.

Protocol hierarchy::

    BayesianInferenceProblemProtocol
      ├── GaussianInferenceProblemProtocol  (+prior_mean, +prior_covariance)
      ├── KLOEDProblemProtocol              (+design_conditions, +weight_bounds)
      └── PredictionOEDProblemProtocol      (+qoi_map, +npred, +design_conditions, +weight_bounds)

KL and Prediction OED protocols extend the *base* protocol, not the
Gaussian one.  A Gaussian prediction problem satisfies both
``PredictionOEDProblemProtocol`` AND ``GaussianInferenceProblemProtocol``
simultaneously (structural typing).  Consumers that need Gaussian
analytics should do an explicit ``isinstance`` check.
"""

from typing import Protocol, runtime_checkable

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend


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
    BayesianInferenceProblemProtocol[Array], Protocol[Array]
):
    """Protocol for a KL-OED problem.

    Extends BayesianInferenceProblemProtocol with design space metadata.
    """

    def design_conditions(self) -> Array: ...
    def weight_bounds(self) -> Array: ...


@runtime_checkable
class PredictionOEDProblemProtocol(
    BayesianInferenceProblemProtocol[Array], Protocol[Array]
):
    """Protocol for a prediction OED problem.

    Extends BayesianInferenceProblemProtocol with qoi_map and design space.
    """

    def qoi_map(self) -> FunctionProtocol[Array]: ...
    def npred(self) -> int: ...
    def design_conditions(self) -> Array: ...
    def weight_bounds(self) -> Array: ...
