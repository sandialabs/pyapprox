"""
Gaussian probability transforms.

Provides transforms to/from standard normal space using the
probability integral transform (CDF/inverse CDF).
"""

from typing import Generic, Tuple, List

import numpy as np
from scipy import stats

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols import MarginalProtocol


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import ScipyContinuousMarginal
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

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (1, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (1, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Expected 1 variable, got {samples.shape[0]}"
            )

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to standard normal space.

        y = Phi^{-1}(F(x))

        Parameters
        ----------
        samples : Array
            Samples from the marginal. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Samples in standard normal space. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with shape (1, nsamples)
        """
        self._validate_input(samples)

        # Get uniform samples via CDF - marginal.cdf expects (1, nsamples)
        probs = self._marginal.cdf(samples)

        # Clip to avoid infinities
        probs_np = np.clip(self._bkd.to_numpy(probs), 1e-15, 1.0 - 1e-15)

        # Transform to standard normal
        result = self._bkd.asarray(self._standard_normal.ppf(probs_np))
        return self._bkd.reshape(result, (1, -1))

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform from standard normal space to marginal space.

        x = F^{-1}(Phi(y))

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Samples from the marginal. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with shape (1, nsamples)
        """
        self._validate_input(canonical_samples)

        # Get uniform samples via standard normal CDF
        probs_np = self._standard_normal.cdf(
            self._bkd.to_numpy(canonical_samples)
        )
        probs = self._bkd.asarray(probs_np)
        probs = self._bkd.reshape(probs, (1, -1))

        # Transform to marginal space - marginal.invcdf expects (1, nsamples)
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
            Samples from the marginal. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            canonical : Standard normal samples. Shape: (1, nsamples)
            jacobian : dy/dx values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with shape (1, nsamples)
        """
        self._validate_input(samples)
        canonical = self.map_to_canonical(samples)

        # f(x) / phi(y)
        marginal_pdf = self._bkd.exp(self._marginal.logpdf(samples))
        normal_pdf = self._bkd.asarray(
            self._standard_normal.pdf(self._bkd.to_numpy(canonical))
        )
        normal_pdf = self._bkd.reshape(normal_pdf, (1, -1))

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
            Standard normal samples. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            samples : Marginal samples. Shape: (1, nsamples)
            jacobian : dx/dy values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with shape (1, nsamples)
        """
        self._validate_input(canonical_samples)
        samples = self.map_from_canonical(canonical_samples)

        # phi(y) / f(x)
        normal_pdf = self._bkd.asarray(
            self._standard_normal.pdf(self._bkd.to_numpy(canonical_samples))
        )
        normal_pdf = self._bkd.reshape(normal_pdf, (1, -1))
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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import ScipyContinuousMarginal
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

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), "
                f"got {samples.ndim}D"
            )
        if samples.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {samples.shape[0]}"
            )

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to standard normal space.

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

        canonical_list = []
        for i, transform in enumerate(self._transforms):
            # Reshape row to (1, nsamples) for univariate transform
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            canon = transform.map_to_canonical(row_2d)
            canonical_list.append(canon[0])  # Back to 1D for stacking

        return self._bkd.stack(canonical_list, axis=0)

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform from standard normal space.

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

        samples_list = []
        for i, transform in enumerate(self._transforms):
            # Reshape row to (1, nsamples) for univariate transform
            row_2d = self._bkd.reshape(canonical_samples[i], (1, -1))
            samp = transform.map_from_canonical(row_2d)
            samples_list.append(samp[0])  # Back to 1D for stacking

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
            Samples from the joint. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            canonical : Standard normal samples. Shape: (nvars, nsamples)
            jacobian_diag : Diagonal Jacobian. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)

        canonical_list = []
        jacobian_list = []
        for i, transform in enumerate(self._transforms):
            # Reshape row to (1, nsamples) for univariate transform
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            canon, jac = transform.map_to_canonical_with_jacobian(row_2d)
            canonical_list.append(canon[0])  # Back to 1D for stacking
            jacobian_list.append(jac[0])

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
            Standard normal samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            samples : Joint samples. Shape: (nvars, nsamples)
            jacobian_diag : Diagonal Jacobian. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(canonical_samples)

        samples_list = []
        jacobian_list = []
        for i, transform in enumerate(self._transforms):
            # Reshape row to (1, nsamples) for univariate transform
            row_2d = self._bkd.reshape(canonical_samples[i], (1, -1))
            samp, jac = transform.map_from_canonical_with_jacobian(row_2d)
            samples_list.append(samp[0])  # Back to 1D for stacking
            jacobian_list.append(jac[0])

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
            Samples from the joint. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log determinant. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        _, jacobian_diag = self.map_to_canonical_with_jacobian(samples)
        result = self._bkd.sum(
            self._bkd.log(self._bkd.abs(jacobian_diag)), axis=0
        )
        return self._bkd.reshape(result, (1, -1))

    def log_det_jacobian_from_canonical(
        self, canonical_samples: Array
    ) -> Array:
        """
        Compute log absolute determinant of Jacobian (from canonical).

        Parameters
        ----------
        canonical_samples : Array
            Standard normal samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log determinant. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        _, jacobian_diag = self.map_from_canonical_with_jacobian(
            canonical_samples
        )
        result = self._bkd.sum(
            self._bkd.log(self._bkd.abs(jacobian_diag)), axis=0
        )
        return self._bkd.reshape(result, (1, -1))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IndependentGaussianTransform(nvars={self._nvars})"
