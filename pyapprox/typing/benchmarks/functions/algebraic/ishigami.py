"""Ishigami function for sensitivity analysis benchmarks.

The Ishigami function is a standard benchmark for global sensitivity analysis,
featuring non-monotonic behavior and variable interaction effects.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class IshigamiFunction(Generic[Array]):
    """Ishigami function: f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1).

    Standard parameters: a=7, b=0.1
    Standard domain: [-pi, pi]^3

    This function implements FunctionWithJacobianAndHVPProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    a : float, optional
        First parameter (default 7.0).
    b : float, optional
        Second parameter (default 0.1).

    References
    ----------
    Ishigami, T. and Homma, T. (1990). "An importance quantification technique
    in uncertainty analysis for computer models."
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: float = 7.0,
        b: float = 0.1,
    ) -> None:
        self._bkd = bkd
        self._a = a
        self._b = b

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 3

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at multiple samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (3, nsamples).

        Returns
        -------
        Array
            Function values of shape (1, nsamples).
        """
        x1 = samples[0:1, :]
        x2 = samples[1:2, :]
        x3 = samples[2:3, :]
        bkd = self._bkd
        return (
            bkd.sin(x1)
            + self._a * bkd.sin(x2) ** 2
            + self._b * x3**4 * bkd.sin(x1)
        )

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (3, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (1, 3).

        Raises
        ------
        ValueError
            If sample is not 2D with shape (3, 1).
        """
        if sample.ndim != 2 or sample.shape != (3, 1):
            raise ValueError(
                f"sample must have shape (3, 1), got {sample.shape}"
            )

        x1 = sample[0, 0]
        x2 = sample[1, 0]
        x3 = sample[2, 0]
        bkd = self._bkd

        df_dx1 = bkd.cos(x1) * (1 + self._b * x3**4)
        df_dx2 = 2 * self._a * bkd.sin(x2) * bkd.cos(x2)
        df_dx3 = 4 * self._b * x3**3 * bkd.sin(x1)

        return bkd.stack(
            [bkd.asarray([df_dx1, df_dx2, df_dx3])], axis=0
        )

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        More efficient than computing the full Hessian for optimization.

        Parameters
        ----------
        sample : Array
            Single sample of shape (3, 1).
        vec : Array
            Direction vector of shape (3, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (3, 1).

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (3, 1).
        """
        if sample.ndim != 2 or sample.shape != (3, 1):
            raise ValueError(
                f"sample must have shape (3, 1), got {sample.shape}"
            )
        if vec.ndim != 2 or vec.shape != (3, 1):
            raise ValueError(f"vec must have shape (3, 1), got {vec.shape}")

        x1 = sample[0, 0]
        x2 = sample[1, 0]
        x3 = sample[2, 0]
        v1 = vec[0, 0]
        v2 = vec[1, 0]
        v3 = vec[2, 0]
        bkd = self._bkd

        # Hessian elements (computed on-the-fly, not stored)
        # H11 = d^2f/dx1^2 = -sin(x1) * (1 + b*x3^4)
        H11 = -bkd.sin(x1) * (1 + self._b * x3**4)
        # H22 = d^2f/dx2^2 = 2*a*(cos^2(x2) - sin^2(x2)) = 2*a*cos(2*x2)
        H22 = 2 * self._a * (bkd.cos(x2) ** 2 - bkd.sin(x2) ** 2)
        # H33 = d^2f/dx3^2 = 12*b*x3^2*sin(x1)
        H33 = 12 * self._b * x3**2 * bkd.sin(x1)
        # H12 = H21 = 0
        # H13 = H31 = d^2f/dx1dx3 = 4*b*x3^3*cos(x1)
        H13 = 4 * self._b * x3**3 * bkd.cos(x1)
        # H23 = H32 = 0

        # H @ v (symmetric matrix)
        return bkd.reshape(
            bkd.stack(
                (
                    H11 * v1 + H13 * v3,
                    H22 * v2,
                    H13 * v1 + H33 * v3,
                ),
                axis=0,
            ),
            (3, 1),
        )
