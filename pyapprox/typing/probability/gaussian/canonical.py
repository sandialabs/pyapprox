"""
Gaussian distributions in canonical (information) form.

The canonical form represents a Gaussian as:
    p(x|h,K) = exp(g + h^T x - 0.5 x^T K x)

where:
- K = precision matrix (inverse covariance)
- h = K @ mean (precision-weighted mean)
- g = normalization constant

This form is efficient for:
- Multiplication of Gaussians (factor products)
- Conditioning on observed variables
- Message passing in graphical models
"""

from typing import Generic, Tuple
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class GaussianCanonicalForm(Generic[Array]):
    """
    Gaussian distribution in canonical (information) form.

    p(x|h,K) = exp(g + h^T x - 0.5 x^T K x)

    Parameters
    ----------
    precision : Array
        Precision matrix K = Cov^{-1}. Shape: (nvars, nvars)
    shift : Array
        Precision-weighted mean h = K @ mean. Shape: (nvars,) or (nvars, 1)
    normalization : float
        Log normalization constant g.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The normalization constant is:
        g = -0.5 * mean^T @ h - 0.5 * nvars * log(2*pi) + 0.5 * log|K|
    """

    def __init__(
        self,
        precision: Array,
        shift: Array,
        normalization: float,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._precision = precision
        self._nvars = precision.shape[0]

        # Ensure shift is 1D
        if shift.ndim == 2:
            shift = shift.flatten()
        self._shift = shift
        self._normalization = normalization

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def precision(self) -> Array:
        """Return the precision matrix K."""
        return self._precision

    def shift(self) -> Array:
        """Return the shift vector h = K @ mean."""
        return self._shift

    def normalization(self) -> float:
        """Return the log normalization constant g."""
        return self._normalization

    @classmethod
    def from_moments(
        cls,
        mean: Array,
        covariance: Array,
        bkd: Backend[Array],
    ) -> "GaussianCanonicalForm[Array]":
        """
        Create canonical form from mean and covariance.

        Parameters
        ----------
        mean : Array
            Mean vector. Shape: (nvars,) or (nvars, 1)
        covariance : Array
            Covariance matrix. Shape: (nvars, nvars)
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        GaussianCanonicalForm
            Gaussian in canonical form.
        """
        if mean.ndim == 2:
            mean = mean.flatten()

        nvars = mean.shape[0]
        precision = bkd.inv(covariance)
        shift = precision @ mean

        # g = -0.5 * m^T h - 0.5 * n * log(2*pi) + 0.5 * log|K|
        sign, logdet = bkd.slogdet(precision)
        normalization = 0.5 * (
            -float(mean @ shift)
            - nvars * math.log(2 * math.pi)
            + float(logdet)
        )

        return cls(precision, shift, normalization, bkd)

    def to_moments(self) -> Tuple[Array, Array]:
        """
        Convert to mean and covariance form.

        Returns
        -------
        mean : Array
            Mean vector. Shape: (nvars,)
        covariance : Array
            Covariance matrix. Shape: (nvars, nvars)
        """
        covariance = self._bkd.inv(self._precision)
        mean = covariance @ self._shift
        return mean, covariance

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate log probability density function.

        log p(x) = g + h^T x - 0.5 x^T K x

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        # h^T x for each sample
        linear_term = self._shift @ samples

        # x^T K x for each sample
        Kx = self._precision @ samples
        quadratic_term = self._bkd.sum(samples * Kx, axis=0)

        result = self._normalization + linear_term - 0.5 * quadratic_term
        return self._bkd.reshape(result, (1, -1))

    def pdf(self, samples: Array) -> Array:
        """Evaluate probability density function."""
        return self._bkd.exp(self.logpdf(samples))

    def multiply(
        self, other: "GaussianCanonicalForm[Array]"
    ) -> "GaussianCanonicalForm[Array]":
        """
        Multiply two Gaussian potentials (unnormalized densities).

        In canonical form, multiplication is simple addition:
            K_new = K1 + K2
            h_new = h1 + h2
            g_new = g1 + g2

        Note: The result is NOT normalized. Use normalize() if needed.

        Parameters
        ----------
        other : GaussianCanonicalForm
            Other Gaussian potential.

        Returns
        -------
        GaussianCanonicalForm
            Product (unnormalized).
        """
        if self._nvars != other.nvars():
            raise ValueError(
                f"Dimension mismatch: {self._nvars} vs {other.nvars()}"
            )

        new_precision = self._precision + other.precision()
        new_shift = self._shift + other.shift()
        new_normalization = self._normalization + other.normalization()

        return GaussianCanonicalForm(
            new_precision, new_shift, new_normalization, self._bkd
        )

    def normalize(self) -> "GaussianCanonicalForm[Array]":
        """
        Renormalize to be a proper probability distribution.

        Returns
        -------
        GaussianCanonicalForm
            Normalized Gaussian with correct normalization constant.
        """
        mean, covariance = self.to_moments()
        return GaussianCanonicalForm.from_moments(mean, covariance, self._bkd)

    def condition(
        self, fixed_indices: Array, values: Array
    ) -> "GaussianCanonicalForm[Array]":
        """
        Compute conditional distribution p(x_remain | x_fixed = values).

        In canonical form, conditioning is efficient:
            h_new = h1 - K12 @ values
            K_new = K11
            g_new = g + h2^T @ values - 0.5 * values^T @ K22 @ values

        Parameters
        ----------
        fixed_indices : Array
            Indices of variables to condition on.
        values : Array
            Values of fixed variables. Shape: (n_fixed,)

        Returns
        -------
        GaussianCanonicalForm
            Conditional distribution.
        """
        if values.ndim == 2:
            values = values.flatten()

        # Determine remaining indices
        all_indices = self._bkd.arange(self._nvars)
        fixed_set = set(int(i) for i in fixed_indices)
        remain_indices = self._bkd.array(
            [int(i) for i in all_indices if int(i) not in fixed_set]
        )

        # Extract blocks using numpy indexing
        remain_np = np.asarray(remain_indices)
        fixed_np = np.asarray(fixed_indices)
        K11 = self._precision[np.ix_(remain_np, remain_np)]
        K12 = self._precision[np.ix_(remain_np, fixed_np)]
        K22 = self._precision[np.ix_(fixed_np, fixed_np)]
        h1 = self._shift[remain_np]
        h2 = self._shift[fixed_np]

        # New canonical parameters
        new_precision = K11
        new_shift = h1 - K12 @ values
        new_normalization = (
            self._normalization
            + float(h2 @ values)
            - 0.5 * float(values @ (K22 @ values))
        )

        return GaussianCanonicalForm(
            new_precision, new_shift, new_normalization, self._bkd
        )

    def marginalize(
        self, marg_indices: Array
    ) -> "GaussianCanonicalForm[Array]":
        """
        Compute marginal distribution by integrating out variables.

        In canonical form, marginalization is complex:
            h_new = h1 - K12 @ K22^{-1} @ h2
            K_new = K11 - K12 @ K22^{-1} @ K21
            g_new = g + 0.5 * (n2*log(2*pi) + log|K22| + h2^T @ K22^{-1} @ h2)

        Parameters
        ----------
        marg_indices : Array
            Indices of variables to marginalize out.

        Returns
        -------
        GaussianCanonicalForm
            Marginal distribution.
        """
        # Determine remaining indices
        all_indices = self._bkd.arange(self._nvars)
        marg_set = set(int(i) for i in marg_indices)
        remain_indices = self._bkd.array(
            [int(i) for i in all_indices if int(i) not in marg_set]
        )

        n_marg = len(marg_indices)

        # Extract blocks using numpy indexing
        remain_np = np.asarray(remain_indices)
        marg_np = np.asarray(marg_indices)
        K11 = self._precision[np.ix_(remain_np, remain_np)]
        K12 = self._precision[np.ix_(remain_np, marg_np)]
        K21 = self._precision[np.ix_(marg_np, remain_np)]
        K22 = self._precision[np.ix_(marg_np, marg_np)]
        h1 = self._shift[remain_np]
        h2 = self._shift[marg_np]

        # K22^{-1}
        K22_inv = self._bkd.inv(K22)

        # New canonical parameters
        new_precision = K11 - K12 @ K22_inv @ K21
        new_shift = h1 - K12 @ K22_inv @ h2

        # Normalization adjustment
        sign, logdet_K22 = self._bkd.slogdet(K22)
        h2_K22inv_h2 = float(h2 @ (K22_inv @ h2))
        new_normalization = (
            self._normalization
            + 0.5
            * (n_marg * math.log(2 * math.pi) + float(logdet_K22) + h2_K22inv_h2)
        )

        return GaussianCanonicalForm(
            new_precision, new_shift, new_normalization, self._bkd
        )

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples.

        Parameters
        ----------
        nsamples : int
            Number of samples.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        mean, covariance = self.to_moments()
        L = self._bkd.cholesky(covariance)
        std_normal = self._bkd.asarray(
            np.random.normal(0, 1, (self._nvars, nsamples))
        )
        return L @ std_normal + mean[:, None]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianCanonicalForm(nvars={self._nvars})"


def compute_normalization(
    mean: Array, shift: Array, precision: Array, bkd: Backend[Array]
) -> float:
    """
    Compute log normalization constant for canonical form.

    g = -0.5 * mean^T @ shift - 0.5 * n * log(2*pi) + 0.5 * log|K|

    Parameters
    ----------
    mean : Array
        Mean vector.
    shift : Array
        Shift vector h = K @ mean.
    precision : Array
        Precision matrix K.
    bkd : Backend[Array]
        Backend.

    Returns
    -------
    float
        Log normalization constant g.
    """
    if mean.ndim == 2:
        mean = mean.flatten()
    if shift.ndim == 2:
        shift = shift.flatten()

    nvars = precision.shape[0]
    sign, logdet = bkd.slogdet(precision)

    return 0.5 * (
        -float(mean @ shift) - nvars * math.log(2 * math.pi) + float(logdet)
    )
