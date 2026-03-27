"""
OED data generation utilities.

Provides convenience functions for generating quadrature data for the
inner and outer loops of the OED double-loop estimator, and for
constructing joint prior-noise distributions.
"""

from typing import Generic, Optional, Tuple, Union

import numpy as np

from pyapprox.expdesign.quadrature.strategies import get_sampler
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend

# Type alias for priors accepted by the data generator
PriorType = Union[
    IndependentJoint[Array],
    DenseCholeskyMultivariateGaussian[Array],
]


def _extract_marginals(
    prior: PriorType[Array],
    bkd: "Backend[Array]",
) -> list[Any]:
    """Extract independent Gaussian marginals from a prior distribution.

    For ``IndependentJoint``, delegates to ``marginals()``.  For
    ``DenseCholeskyMultivariateGaussian``, extracts marginals from the
    diagonal of the covariance matrix (assumes independent components).
    """
    if isinstance(prior, IndependentJoint):
        return list(prior.marginals())

    # DenseCholeskyMultivariateGaussian — extract from diagonal
    mean_np = bkd.to_numpy(prior.mean()).ravel()
    cov_np = bkd.to_numpy(prior.covariance())
    nvars = len(mean_np)
    return [
        GaussianMarginal(float(mean_np[i]), float(np.sqrt(cov_np[i, i])), bkd)
        for i in range(nvars)
    ]


class OEDDataGenerator(Generic[Array]):
    """
    Generates quadrature data for OED double-loop estimation.

    Wraps existing sampler strategies to provide a convenient interface
    for generating outer-loop and inner-loop samples and weights.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def joint_prior_noise_variable(
        self,
        prior: PriorType[Array],
        noise_cov_diag: Array,
    ) -> IndependentJoint[Array]:
        """
        Create joint distribution over prior parameters and noise.

        Concatenates the prior marginals with independent Gaussian noise
        marginals. This is used for outer-loop sampling where both
        parameter samples and noise samples are needed jointly.

        Parameters
        ----------
        prior : IndependentJoint or DenseCholeskyMultivariateGaussian
            Prior distribution over model parameters. If a multivariate
            Gaussian is given, marginals are extracted from the diagonal
            of its covariance matrix (assumes independent components).
        noise_cov_diag : Array
            Diagonal of the noise covariance matrix. Shape: (nobs,) or
            (nobs, 1). Each element is the variance of the noise at
            one observation.

        Returns
        -------
        IndependentJoint[Array]
            Joint distribution with prior marginals followed by noise
            marginals.
        """
        # Flatten noise_cov_diag if needed
        noise_var = self._bkd.to_numpy(noise_cov_diag).ravel()

        # Create noise marginals
        noise_marginals = [
            GaussianMarginal(0.0, float(np.sqrt(v)), self._bkd) for v in noise_var
        ]

        # Extract prior marginals (works for both prior types)
        prior_marginals = _extract_marginals(prior, self._bkd)

        return IndependentJoint(prior_marginals + noise_marginals, self._bkd)

    def setup_quadrature_data(
        self,
        quadrature_type: str,
        variable: PriorType[Array],
        nsamples: int,
        loop: str,
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate quadrature samples and weights for an OED loop.

        Parameters
        ----------
        quadrature_type : str
            Type of quadrature: "MC", "Halton", "Sobol", or "gauss".
            Case-insensitive.
        variable : IndependentJoint or DenseCholeskyMultivariateGaussian
            Distribution to sample from. For outer loop, this should be
            the joint prior-noise distribution. For inner loop, this
            should be the prior distribution only. If a multivariate
            Gaussian is given, it is converted to an IndependentJoint.
        nsamples : int
            Number of quadrature samples.
        loop : str
            Either "outer" or "inner". Currently informational only.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : Array
            Quadrature sample points. Shape: (nvars, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        # Normalize quadrature type name
        type_map = {
            "mc": "mc",
            "halton": "halton",
            "sobol": "sobol",
            "gauss": "gauss",
        }
        qt_lower = quadrature_type.lower()
        if qt_lower not in type_map:
            raise ValueError(
                f"Unknown quadrature_type: {quadrature_type}. "
                f"Expected one of: MC, Halton, Sobol, gauss"
            )

        # Convert multivariate Gaussian to IndependentJoint if needed
        if isinstance(variable, DenseCholeskyMultivariateGaussian):
            marginals = _extract_marginals(variable, self._bkd)
            variable = IndependentJoint(marginals, self._bkd)

        strategy_cls = get_sampler(type_map[qt_lower])
        strategy = strategy_cls()
        return strategy.sample(variable, nsamples, self._bkd, seed=seed)
