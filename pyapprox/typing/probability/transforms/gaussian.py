"""
Gaussian probability transforms.

Provides transforms to/from standard normal space using the
probability integral transform (CDF/inverse CDF).
"""

from typing import Generic, Tuple, List

import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.protocols import MarginalProtocol


class GaussianTransform(Generic[Array]):
    """
    Transform a single marginal to/from standard normal.

    Uses the probability integral transform:
        y = Phi^{-1}(F(x))  (to canonical/standard normal)
        x = F^{-1}(Phi(y))  (from canonical)

    where F is the CDF of the marginal, Phi is standard normal CDF.

    Parameters
    ----------
    marginal : MarginalProtocol[Array]
        The marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import uniform
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import ScipyContinuousMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = ScipyContinuousMarginal(uniform(0, 1), bkd)
    >>> transform = GaussianTransform(marginal, bkd)
    >>> x = np.array([0.5])  # median of uniform
    >>> y = transform.map_to_canonical(x)  # should be 0 (median of normal)
    """

    def __init__(
        self,
        marginal: MarginalProtocol[Array],
        bkd: Backend[Array],
    ):
        if marginal.nvars() != 1:
            raise ValueError(
                f"GaussianTransform requires univariate marginal (nvars=1), "
                f"got nvars={marginal.nvars()}"
            )
        self._bkd = bkd
        self._marginal = marginal
        # Standard normal for inverse CDF
        self._standard_normal = stats.norm(0, 1)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 1)."""
        return 1

    def marginal(self) -> MarginalProtocol[Array]:
        """Return the marginal distribution."""
        return self._marginal

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to standard normal space.

        y = Phi^{-1}(F(x))

        Parameters
        ----------
        samples : Array
            Samples from the marginal. Shape: (nsamples,) or (1, nsamples)

        Returns
        -------
        Array
            Samples in standard normal space. Same shape as input.
        """
        # Get uniform samples via CDF
        probs = self._marginal.cdf(samples)

        # Clip to avoid infinities
        probs_np = np.clip(self._bkd.to_numpy(probs), 1e-15, 1.0 - 1e-15)

        # Transform to standard normal
        return self._bkd.asarray(self._standard_normal.ppf(probs_np))

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform from standard normal space to marginal space.

        x = F^{-1}(Phi(y))

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nsamples,) or (1, nsamples)

        Returns
        -------
        Array
            Samples from the marginal.
        """
        # Get uniform samples via standard normal CDF
        probs = self._bkd.asarray(
            self._standard_normal.cdf(self._bkd.to_numpy(canonical_samples))
        )

        # Transform to marginal space
        return self._marginal.invcdf(probs)

    def map_to_canonical_with_jacobian(
        self, samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform to standard normal with Jacobian.

        The Jacobian is:
            dy/dx = f(x) / phi(y)

        where f is the marginal PDF, phi is standard normal PDF.

        Parameters
        ----------
        samples : Array
            Samples from the marginal. Shape: (nsamples,) or (1, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            canonical : Standard normal samples
            jacobian : dy/dx values. Shape same as samples
        """
        canonical = self.map_to_canonical(samples)

        # f(x) / phi(y)
        marginal_pdf = self._bkd.exp(self._marginal.logpdf(samples))
        normal_pdf = self._bkd.asarray(
            self._standard_normal.pdf(self._bkd.to_numpy(canonical))
        )

        # Avoid division by zero
        jacobian = marginal_pdf / (normal_pdf + 1e-15)

        return canonical, jacobian

    def map_from_canonical_with_jacobian(
        self, canonical_samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform from standard normal with Jacobian.

        The Jacobian is:
            dx/dy = phi(y) / f(x)

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples.

        Returns
        -------
        Tuple[Array, Array]
            samples : Marginal samples
            jacobian : dx/dy values
        """
        samples = self.map_from_canonical(canonical_samples)

        # phi(y) / f(x)
        normal_pdf = self._bkd.asarray(
            self._standard_normal.pdf(self._bkd.to_numpy(canonical_samples))
        )
        marginal_pdf = self._bkd.exp(self._marginal.logpdf(samples))

        jacobian = normal_pdf / (marginal_pdf + 1e-15)

        return samples, jacobian

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianTransform(marginal={self._marginal})"


class IndependentGaussianTransform(Generic[Array]):
    """
    Transform independent marginals to standard normal space.

    Applies GaussianTransform to each dimension independently:
        y_i = Phi^{-1}(F_i(x_i))

    Parameters
    ----------
    marginals : List[MarginalProtocol[Array]]
        List of marginal distributions.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import uniform, beta
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import ScipyContinuousMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [
    ...     ScipyContinuousMarginal(uniform(0, 1), bkd),
    ...     ScipyContinuousMarginal(beta(2, 5), bkd),
    ... ]
    >>> transform = IndependentGaussianTransform(marginals, bkd)
    >>> x = np.array([[0.5], [0.3]])  # Shape: (2, 1)
    >>> y = transform.map_to_canonical(x)  # Standard normal samples
    """

    def __init__(
        self,
        marginals: List[MarginalProtocol[Array]],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._marginals = marginals
        self._nvars = len(marginals)
        self._transforms = [GaussianTransform(m, bkd) for m in marginals]
        self._standard_normal = stats.norm(0, 1)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Return list of marginal distributions."""
        return self._marginals

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to standard normal space.

        Parameters
        ----------
        samples : Array
            Samples from the joint. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Standard normal samples. Shape: (nvars, nsamples)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        canonical_list = []
        for i, transform in enumerate(self._transforms):
            canonical_list.append(transform.map_to_canonical(samples[i]))

        return self._bkd.stack(canonical_list, axis=0)

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform from standard normal space.

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Samples from the joint. Shape: (nvars, nsamples)
        """
        if canonical_samples.ndim == 1:
            canonical_samples = self._bkd.reshape(
                canonical_samples, (self._nvars, 1)
            )

        samples_list = []
        for i, transform in enumerate(self._transforms):
            samples_list.append(
                transform.map_from_canonical(canonical_samples[i])
            )

        return self._bkd.stack(samples_list, axis=0)

    def map_to_canonical_with_jacobian(
        self, samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform to standard normal with Jacobian diagonal.

        For independent marginals, the Jacobian is diagonal:
            dy_i/dx_i = f_i(x_i) / phi(y_i)

        Parameters
        ----------
        samples : Array
            Samples from the joint. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            canonical : Standard normal samples. Shape: (nvars, nsamples)
            jacobian_diag : Diagonal Jacobian. Shape: (nvars, nsamples)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        canonical_list = []
        jacobian_list = []
        for i, transform in enumerate(self._transforms):
            canon, jac = transform.map_to_canonical_with_jacobian(samples[i])
            canonical_list.append(canon)
            jacobian_list.append(jac)

        return (
            self._bkd.stack(canonical_list, axis=0),
            self._bkd.stack(jacobian_list, axis=0),
        )

    def map_from_canonical_with_jacobian(
        self, canonical_samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform from standard normal with Jacobian diagonal.

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            samples : Joint samples. Shape: (nvars, nsamples)
            jacobian_diag : Diagonal Jacobian. Shape: (nvars, nsamples)
        """
        if canonical_samples.ndim == 1:
            canonical_samples = self._bkd.reshape(
                canonical_samples, (self._nvars, 1)
            )

        samples_list = []
        jacobian_list = []
        for i, transform in enumerate(self._transforms):
            samp, jac = transform.map_from_canonical_with_jacobian(
                canonical_samples[i]
            )
            samples_list.append(samp)
            jacobian_list.append(jac)

        return (
            self._bkd.stack(samples_list, axis=0),
            self._bkd.stack(jacobian_list, axis=0),
        )

    def log_det_jacobian_to_canonical(self, samples: Array) -> Array:
        """
        Compute log absolute determinant of Jacobian (to canonical).

        For separable transforms:
            log|det(J)| = sum_i log|dy_i/dx_i|

        Parameters
        ----------
        samples : Array
            Samples from the joint. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log determinant. Shape: (nsamples,)
        """
        _, jacobian_diag = self.map_to_canonical_with_jacobian(samples)
        return self._bkd.sum(
            self._bkd.log(self._bkd.abs(jacobian_diag)), axis=0
        )

    def log_det_jacobian_from_canonical(
        self, canonical_samples: Array
    ) -> Array:
        """
        Compute log absolute determinant of Jacobian (from canonical).

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log determinant. Shape: (nsamples,)
        """
        _, jacobian_diag = self.map_from_canonical_with_jacobian(
            canonical_samples
        )
        return self._bkd.sum(
            self._bkd.log(self._bkd.abs(jacobian_diag)), axis=0
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IndependentGaussianTransform(nvars={self._nvars})"
