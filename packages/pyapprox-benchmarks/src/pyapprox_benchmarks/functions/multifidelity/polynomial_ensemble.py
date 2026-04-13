"""Polynomial model ensemble for multifidelity benchmarks.

Implements a hierarchy of polynomial models: x^5, x^4, x^3, x^2, x
with decreasing fidelity and cost.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class PolynomialModelFunction(Generic[Array]):
    """Single polynomial model f(x) = x^degree.

    Implements FunctionProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    degree : int
        Polynomial degree.
    """

    def __init__(self, bkd: Backend[Array], degree: int) -> None:
        self._bkd = bkd
        self._degree = degree

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def degree(self) -> int:
        """Return polynomial degree."""
        return self._degree

    def __call__(self, samples: Array) -> Array:
        """Evaluate the polynomial model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (1, nsamples).
        """
        return samples**self._degree

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (1, 1).

        Returns
        -------
        Array
            Jacobian of shape (1, 1).
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian expects single sample with shape (1, 1), "
                f"got shape {sample.shape}"
            )
        x = sample[0, 0]
        return self._bkd.array([[self._degree * x ** (self._degree - 1)]])

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (1, 1).
        vec : Array
            Direction vector of shape (1, 1).

        Returns
        -------
        Array
            HVP result of shape (1, 1).
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"hvp expects single sample with shape (1, 1), got shape {sample.shape}"
            )
        if vec.shape[1] != 1:
            raise ValueError(
                f"hvp expects direction vector with shape (1, 1), got shape {vec.shape}"
            )
        x = sample[0, 0]
        v = vec[0, 0]
        d = self._degree
        if d <= 1:
            hess_val = 0.0
        else:
            hess_val = d * (d - 1) * x ** (d - 2)
        return self._bkd.array([[hess_val * v]])

    def mean(self) -> float:
        """Analytical mean for U[0,1] input.

        Returns
        -------
        float
            Mean value E[x^degree] = 1/(degree+1).
        """
        return 1.0 / (self._degree + 1)

    def variance(self) -> float:
        """Analytical variance for U[0,1] input.

        Returns
        -------
        float
            Variance Var[x^degree] = 1/(2*degree+1) - 1/(degree+1)^2.
        """
        d = self._degree
        return 1.0 / (2 * d + 1) - 1.0 / (d + 1) ** 2


