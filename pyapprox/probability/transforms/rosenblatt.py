"""
Rosenblatt transform for general joint distributions.

The Rosenblatt transform converts samples from an arbitrary joint distribution
to independent uniform [0,1] samples using the conditional CDFs:
    u_1 = F_1(x_1)
    u_2 = F_{2|1}(x_2 | x_1)
    u_3 = F_{3|1,2}(x_3 | x_1, x_2)
    ...

Combined with the inverse normal CDF, this gives independent standard normals.
"""

from typing import Any, Callable, Generic, Optional

import numpy as np
from scipy import stats

from pyapprox.optimization.rootfinding.bisection import (
    BisectionSearch,
)
from pyapprox.util.backends.protocols import Array, Backend

# TODO: do we need clip. Also we shoul not be using np.fun only bkd.fun
# so computational graph is not broken for autograd
# TODO: Fix type errors in this file


class RosenblattTransform(Generic[Array]):
    """
    Rosenblatt transform for joint distributions to independent uniform.

    This is the base class that uses numerical approximation of
    conditional CDFs. Specialized distributions should override
    the conditional CDF methods for efficiency.

    The transform maps x -> u where u_i ~ U[0,1] independent.
    Combined with Phi^{-1}, this gives standard normals.

    Parameters
    ----------
    joint_pdf : callable
        Function that evaluates the joint PDF: f(x) -> pdf values.
    nvars : int
        Number of variables.
    bounds : Array, optional
        Bounds for numerical integration. Shape: (2, nvars).
        Default: (-5, 5) for each variable.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The Rosenblatt transform is the multivariate generalization of the
    probability integral transform. It's more general than the Nataf
    transform but requires knowledge of conditional distributions.

    For practical use, specialized implementations should be provided
    for specific distribution families (e.g., Gaussian, copulas).
    """

    def __init__(
        self,
        joint_pdf: callable,
        nvars: int,
        bkd: Backend[Array],
        bounds: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._joint_pdf = joint_pdf
        self._nvars = nvars

        if bounds is None:
            self._bounds = bkd.asarray([[-5.0] * nvars, [5.0] * nvars])
        else:
            self._bounds = bounds

        self._standard_normal = stats.norm(0, 1)

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {samples.shape[0]}"
            )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def map_to_uniform(self, samples: Array) -> Array:
        """
        Transform samples to independent uniform [0,1].

        Uses numerical integration of conditional CDFs.

        Parameters
        ----------
        samples : Array
            Samples from the joint. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Uniform samples. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)

        nsamples = samples.shape[1]
        uniform = self._bkd.zeros((self._nvars, nsamples))

        for i in range(nsamples):
            x = samples[:, i]
            uniform[:, i] = self._compute_conditional_cdfs(x)

        return uniform

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to independent standard normal.

        Parameters
        ----------
        samples : Array
            Samples from the joint. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Standard normal samples. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)
        uniform = self.map_to_uniform(samples)
        # Clip to avoid infinities
        uniform_np = np.clip(self._bkd.to_numpy(uniform), 1e-15, 1.0 - 1e-15)
        return self._bkd.asarray(self._standard_normal.ppf(uniform_np))

    def map_from_uniform(self, uniform_samples: Array) -> Array:
        """
        Transform uniform samples to joint distribution.

        Uses numerical inversion of conditional CDFs.

        Parameters
        ----------
        uniform_samples : Array
            Uniform [0,1] samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Samples from the joint. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(uniform_samples)

        nsamples = uniform_samples.shape[1]
        samples = self._bkd.zeros((self._nvars, nsamples))

        for i in range(nsamples):
            u = uniform_samples[:, i]
            samples[:, i] = self._compute_conditional_inverse_cdfs(u)

        return samples

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform standard normal samples to joint distribution.

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Samples from the joint. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(canonical_samples)
        uniform = self._bkd.asarray(
            self._standard_normal.cdf(self._bkd.to_numpy(canonical_samples))
        )
        return self.map_from_uniform(uniform)

    def _compute_conditional_cdfs(self, x: Array) -> Array:
        """
        Compute conditional CDFs for a single sample.

        F_1(x_1), F_{2|1}(x_2|x_1), F_{3|1,2}(x_3|x_1,x_2), ...

        Uses numerical integration.
        """
        from scipy import integrate

        u = np.zeros(self._nvars)
        bounds = self._bkd.to_numpy(self._bounds)
        x_np = self._bkd.to_numpy(x)

        # TODO: we should pass in a quadrature class or factory that satisfies
        # a proticik so we can swap in different quadrature rules easily
        # to allow user to use one that works best for thier porblem
        for i in range(self._nvars):
            if i == 0:
                # F_1(x_1) = integral_{-inf}^{x_1} f_1(t) dt
                def integrand(t: float) -> float:
                    point = np.zeros(self._nvars)
                    point[0] = t
                    # Integrate out remaining variables
                    for j in range(1, self._nvars):
                        point[j] = 0  # Will be integrated
                    return float(self._marginal_pdf_by_integration(point, 0))

                u[0], _ = integrate.quad(integrand, bounds[0, 0], x_np[0], limit=50)
            else:
                # F_{i|1,...,i-1}(x_i | x_1, ..., x_{i-1})
                # = integral f(x_1,...,x_i,*) / f(x_1,...,x_{i-1},*)
                u[i] = self._conditional_cdf_numerical(x_np, i)

        return self._bkd.asarray(u)

    def _compute_conditional_inverse_cdfs(self, u: Array) -> Array:
        """
        Compute inverse conditional CDFs for a single sample.

        x_1 = F_1^{-1}(u_1)
        x_2 = F_{2|1}^{-1}(u_2 | x_1)
        ...

        Uses BisectionSearch from the rootfinding module.
        """
        x = np.zeros(self._nvars)
        u_np = self._bkd.to_numpy(u)
        bounds = self._bkd.to_numpy(self._bounds)

        for i in range(self._nvars):
            # Create a residual for bisection search
            residual = _CDFInversionResidual(
                bkd=self._bkd,
                target=u_np[i],
                dim=i,
                current_x=x.copy(),
                marginal_cdf=self._marginal_cdf_numerical,
                conditional_cdf=self._conditional_cdf_numerical,
            )

            # TODO: when does root finding fail. Should we throw error
            # or is fallback ok
            try:
                bisection = BisectionSearch(residual)
                search_bounds = self._bkd.asarray([[bounds[0, i]], [bounds[1, i]]]).T
                result = bisection.solve(search_bounds, maxiters=50, atol=1e-10)
                x[i] = float(result[0])
            except RuntimeError:
                # Fallback if root finding fails
                x[i] = (bounds[0, i] + bounds[1, i]) / 2

        return self._bkd.asarray(x)

    def _marginal_pdf_by_integration(self, x: Array, dim: int) -> float:
        """
        Compute marginal PDF by integrating out other variables.

        f_i(x_i) = integral f(x) dx_{-i}
        """

        self._bkd.to_numpy(self._bounds)

        # This is a placeholder - actual implementation needs
        # proper multivariate integration
        x_full = np.array(x)
        return float(self._joint_pdf(self._bkd.asarray(x_full[:, None]))[0])

    def _marginal_cdf_numerical(self, x: float, dim: int) -> float:
        """
        Compute marginal CDF numerically.

        F_i(x) = integral_{-inf}^{x} f_i(t) dt
        """
        from scipy import integrate
        # TODO: Again should allow protocol to be passed in so we
        # can use different integration methods

        bounds = self._bkd.to_numpy(self._bounds)

        def integrand(t: float) -> float:
            point = np.zeros(self._nvars)
            point[dim] = t
            return float(self._marginal_pdf_by_integration(point, dim))

        result, _ = integrate.quad(integrand, bounds[0, dim], x, limit=50)
        return float(result)

    def _conditional_cdf_numerical(self, x: Array, dim: int) -> float:
        """
        Compute conditional CDF numerically.

        F_{dim|1,...,dim-1}(x_dim | x_1,...,x_{dim-1})
        """
        # Simplified: for general case, need proper numerical implementation
        # This is a placeholder that assumes independence
        return self._marginal_cdf_numerical(x[dim], dim)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RosenblattTransform(nvars={self._nvars})"


class _CDFInversionResidual(Generic[Array]):
    """
    Residual for CDF inversion via bisection search.

    Finds x such that F(x) = target, i.e., residual = F(x) - target = 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        target: float,
        dim: int,
        current_x: np.ndarray,
        marginal_cdf: Callable[..., Any],
        conditional_cdf: Callable[..., Any],
    ):
        self._bkd = bkd
        self._target = target
        self._dim = dim
        self._current_x = current_x
        self._marginal_cdf = marginal_cdf
        self._conditional_cdf = conditional_cdf

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def __call__(self, iterate: Array) -> Array:
        """Evaluate residual F(x) - target."""
        xi = float(iterate[0])
        self._current_x[self._dim] = xi

        if self._dim == 0:
            cdf_val = self._marginal_cdf(xi, 0)
        else:
            cdf_val = self._conditional_cdf(self._current_x, self._dim)

        return self._bkd.asarray([cdf_val - self._target])
