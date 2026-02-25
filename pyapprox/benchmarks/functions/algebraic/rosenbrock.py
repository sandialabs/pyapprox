"""Rosenbrock function for optimization benchmarks.

The Rosenbrock function is a classic benchmark for nonlinear optimization,
featuring a narrow banana-shaped valley that makes it challenging for
many optimization algorithms.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class RosenbrockFunction(Generic[Array]):
    """Rosenbrock function: f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2].

    The global minimum is at x = (1, 1, ..., 1) with f(x*) = 0.

    This function implements FunctionWithJacobianAndHVPProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    nvars : int
        Number of input variables (must be >= 2).

    References
    ----------
    Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or
    least value of a function."
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
    ) -> None:
        if nvars < 2:
            raise ValueError(f"nvars must be >= 2, got {nvars}")
        self._bkd = bkd
        self._nvars = nvars

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at multiple samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values of shape (1, nsamples).
        """
        bkd = self._bkd
        result = bkd.zeros((1, samples.shape[1]))
        for i in range(self._nvars - 1):
            xi = samples[i:i+1, :]
            xi1 = samples[i+1:i+2, :]
            result = result + 100 * (xi1 - xi**2)**2 + (1 - xi)**2
        return result

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (1, nvars).

        Raises
        ------
        ValueError
            If sample is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )

        bkd = self._bkd
        grad = bkd.zeros((1, self._nvars))

        for i in range(self._nvars - 1):
            xi = sample[i, 0]
            xi1 = sample[i + 1, 0]

            # Gradient contribution for x_i
            # d/dx_i of [100(x_{i+1} - x_i^2)^2] = -400 * x_i * (x_{i+1} - x_i^2)
            # d/dx_i of [(1 - x_i)^2] = -2 * (1 - x_i)
            grad_i = -400 * xi * (xi1 - xi**2) - 2 * (1 - xi)
            grad = bkd.reshape(
                bkd.hstack([
                    grad[0, :i],
                    bkd.asarray([grad[0, i] + grad_i]),
                    grad[0, i+1:]
                ]),
                (1, self._nvars)
            )

            # Gradient contribution for x_{i+1}
            # d/dx_{i+1} of [100(x_{i+1} - x_i^2)^2] = 200 * (x_{i+1} - x_i^2)
            grad_i1 = 200 * (xi1 - xi**2)
            grad = bkd.reshape(
                bkd.hstack([
                    grad[0, :i+1],
                    bkd.asarray([grad[0, i+1] + grad_i1]),
                    grad[0, i+2:] if i+2 < self._nvars else bkd.array([])
                ]),
                (1, self._nvars)
            )

        return grad

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1).

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )
        if vec.ndim != 2 or vec.shape != (self._nvars, 1):
            raise ValueError(
                f"vec must have shape ({self._nvars}, 1), got {vec.shape}"
            )

        bkd = self._bkd
        n = self._nvars

        # Compute Hessian-vector product directly
        # H @ v where H is the Hessian matrix
        result = bkd.zeros((n, 1))

        for i in range(n - 1):
            xi = sample[i, 0]
            xi1 = sample[i + 1, 0]
            vi = vec[i, 0]
            vi1 = vec[i + 1, 0]

            # Hessian elements for the i-th term:
            # H[i,i] += 1200*x_i^2 - 400*x_{i+1} + 2
            # H[i,i+1] = H[i+1,i] = -400*x_i
            # H[i+1,i+1] += 200

            H_ii = 1200 * xi**2 - 400 * xi1 + 2
            H_ii1 = -400 * xi
            H_i1i1 = 200.0

            # Contribution to result[i]
            contrib_i = H_ii * vi + H_ii1 * vi1
            result_i_old = result[i, 0]
            result = bkd.reshape(
                bkd.hstack([
                    result[:i, 0],
                    bkd.asarray([result_i_old + contrib_i]),
                    result[i+1:, 0]
                ]),
                (n, 1)
            )

            # Contribution to result[i+1]
            contrib_i1 = H_ii1 * vi + H_i1i1 * vi1
            result_i1_old = result[i + 1, 0]
            result = bkd.reshape(
                bkd.hstack([
                    result[:i+1, 0],
                    bkd.asarray([result_i1_old + contrib_i1]),
                    result[i+2:, 0] if i + 2 < n else bkd.array([])
                ]),
                (n, 1)
            )

        return result
