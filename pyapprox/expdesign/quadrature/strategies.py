"""
Sampler strategies for OED quadrature.

This module provides a registry-based system for different sampling strategies
used in OED expectation computation. Strategies include Gauss-Hermite quadrature,
Monte Carlo, and quasi-Monte Carlo (Halton/Sobol).

The registry pattern allows new strategies to be added without modifying
existing code.
"""

from typing import (
    Dict,
    Generic,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import numpy as np

from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SamplerStrategy(Protocol, Generic[Array]):
    """Protocol for sampler strategies.

    Sampler strategies generate samples and quadrature weights from a
    given distribution for use in OED expectation computation.
    """

    def sample(
        self,
        distribution: IndependentJoint[Array],
        n_approx: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Generate samples and weights from the distribution.

        Parameters
        ----------
        distribution : IndependentJoint[Array]
            The joint distribution to sample from.
        n_approx : int
            Approximate number of samples. For Gauss quadrature, this is
            used to compute the 1D level. For MC/QMC, this is the exact
            number of samples.
        bkd : Backend[Array]
            Computational backend.
        seed : int, optional
            Random seed for reproducibility. Ignored for deterministic
            quadrature rules (e.g., Gauss).

        Returns
        -------
        samples : Array
            Sample points. Shape: (nvars, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        ...


class GaussStrategy(Generic[Array]):
    """Gauss-Hermite quadrature strategy for Gaussian marginals.

    Uses tensor product Gauss-Hermite quadrature for distributions with
    all Gaussian marginals. Samples are generated on standard normal and
    then transformed via affine transformation: x = mean + std * z.

    This is the most accurate strategy for Gaussian distributions, but
    only works when all marginals are GaussianMarginal.

    Raises
    ------
    TypeError
        If any marginal is not a GaussianMarginal.
    """

    def sample(
        self,
        distribution: IndependentJoint[Array],
        n_approx: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Generate Gauss-Hermite quadrature samples.

        Parameters
        ----------
        distribution : IndependentJoint[Array]
            Joint distribution with all Gaussian marginals.
        n_approx : int
            Approximate total number of quadrature points. The actual
            number is npoints_1d^nvars where npoints_1d is computed as
            int(n_approx^(1/nvars)).
        bkd : Backend[Array]
            Computational backend.
        seed : int, optional
            Ignored (Gauss quadrature is deterministic).

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)

        Raises
        ------
        TypeError
            If any marginal is not a GaussianMarginal.
        """
        # Verify all marginals are Gaussian
        marginals = distribution.marginals()
        for i, m in enumerate(marginals):
            if not isinstance(m, GaussianMarginal):
                raise TypeError(
                    f"Gauss quadrature requires GaussianMarginal, "
                    f"marginal {i} is {type(m).__name__}"
                )

        nvars = distribution.nvars()
        npoints_1d = max(1, int(n_approx ** (1.0 / nvars)))

        # Import quadrature infrastructure
        from pyapprox.surrogates.affine.univariate.globalpoly.hermite import (
            HermitePolynomial1D,
        )
        from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
            GaussQuadratureRule,
        )
        from pyapprox.surrogates.quadrature.tensor_product import (
            TensorProductQuadratureRule,
        )

        # Create Hermite polynomial for standard normal
        hermite = HermitePolynomial1D(bkd, rho=0.0, prob_meas=True)
        gauss_rule = GaussQuadratureRule(hermite)

        # Wrapper to match TensorProductQuadratureRule interface
        # GaussQuadratureRule returns (points, weights) with shapes
        # (1, npoints) and (npoints, 1), but TensorProductQuadratureRule
        # expects 1D arrays
        def gauss_rule_1d(npts: int) -> Tuple[Array, Array]:
            pts, wts = gauss_rule(npts)
            return bkd.flatten(pts), bkd.flatten(wts)

        # Build tensor product rule
        tp_rule = TensorProductQuadratureRule(
            bkd,
            [gauss_rule_1d] * nvars,
            [npoints_1d] * nvars,
        )
        std_samples, weights = tp_rule()

        # Transform from standard normal to actual distribution
        # x_i = mean_i + std_i * z_i for each marginal
        samples = bkd.zeros_like(std_samples)
        for i, marginal in enumerate(marginals):
            mean = marginal.mean_value()
            std = marginal.std()
            samples[i, :] = mean + std * std_samples[i, :]

        return samples, weights


class MCStrategy(Generic[Array]):
    """Monte Carlo sampling strategy.

    Uses the distribution's rvs() method to generate random samples
    with uniform weights (1/nsamples).
    """

    def sample(
        self,
        distribution: IndependentJoint[Array],
        n_approx: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Generate Monte Carlo samples.

        Parameters
        ----------
        distribution : IndependentJoint[Array]
            Joint distribution to sample from.
        n_approx : int
            Number of samples to generate.
        bkd : Backend[Array]
            Computational backend.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : Array
            Random samples. Shape: (nvars, nsamples)
        weights : Array
            Uniform weights (1/nsamples). Shape: (nsamples,)
        """
        if seed is not None:
            np.random.seed(seed)
        samples = distribution.rvs(n_approx)
        weights = bkd.ones((n_approx,)) / n_approx
        return samples, weights


class HaltonStrategy(Generic[Array]):
    """Halton quasi-Monte Carlo sampling strategy.

    Uses Halton sequence with inverse CDF transformation for the
    distribution. Provides better coverage than random sampling.
    """

    def sample(
        self,
        distribution: IndependentJoint[Array],
        n_approx: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Generate Halton sequence samples.

        Parameters
        ----------
        distribution : IndependentJoint[Array]
            Joint distribution to transform samples to.
        n_approx : int
            Number of samples to generate.
        bkd : Backend[Array]
            Computational backend.
        seed : int, optional
            Random seed for scrambling reproducibility.

        Returns
        -------
        samples : Array
            Halton sequence samples. Shape: (nvars, nsamples)
        weights : Array
            Uniform weights (1/nsamples). Shape: (nsamples,)
        """
        from pyapprox.expdesign.quadrature.halton import HaltonSampler

        sampler = HaltonSampler(
            distribution.nvars(),
            bkd,
            distribution=distribution,
            seed=seed,
        )
        return sampler.sample(n_approx)


class SobolStrategy(Generic[Array]):
    """Sobol quasi-Monte Carlo sampling strategy.

    Uses Sobol sequence with inverse CDF transformation for the
    distribution. Provides better coverage than random sampling.
    """

    def sample(
        self,
        distribution: IndependentJoint[Array],
        n_approx: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Generate Sobol sequence samples.

        Parameters
        ----------
        distribution : IndependentJoint[Array]
            Joint distribution to transform samples to.
        n_approx : int
            Number of samples to generate.
        bkd : Backend[Array]
            Computational backend.
        seed : int, optional
            Random seed for scrambling reproducibility.

        Returns
        -------
        samples : Array
            Sobol sequence samples. Shape: (nvars, nsamples)
        weights : Array
            Uniform weights (1/nsamples). Shape: (nsamples,)
        """
        from pyapprox.expdesign.quadrature.sobol import SobolSampler

        sampler = SobolSampler(
            distribution.nvars(),
            bkd,
            distribution=distribution,
            seed=seed,
        )
        return sampler.sample(n_approx)


# Registry for sampler strategies
_SAMPLER_STRATEGIES: Dict[str, Type[SamplerStrategy[Array]]] = {}


def register_sampler(name: str, strategy: Type[SamplerStrategy[Array]]) -> None:
    """Register a sampler strategy.

    Parameters
    ----------
    name : str
        Name to register the strategy under.
    strategy : Type[SamplerStrategy]
        Strategy class to register.
    """
    _SAMPLER_STRATEGIES[name] = strategy


def get_sampler(name: str) -> Type[SamplerStrategy[Array]]:
    """Get a registered sampler strategy.

    Parameters
    ----------
    name : str
        Name of the strategy to retrieve.

    Returns
    -------
    Type[SamplerStrategy]
        The registered strategy class.

    Raises
    ------
    ValueError
        If the strategy name is not registered.
    """
    if name not in _SAMPLER_STRATEGIES:
        available = list(_SAMPLER_STRATEGIES.keys())
        raise ValueError(f"Unknown sampler: '{name}'. Available: {available}")
    return _SAMPLER_STRATEGIES[name]


def list_samplers() -> list[Any]:
    """List all registered sampler names.

    Returns
    -------
    list
        List of registered sampler names.
    """
    return list(_SAMPLER_STRATEGIES.keys())


# Register built-in strategies
register_sampler("gauss", GaussStrategy)
register_sampler("mc", MCStrategy)
register_sampler("halton", HaltonStrategy)
register_sampler("sobol", SobolStrategy)
