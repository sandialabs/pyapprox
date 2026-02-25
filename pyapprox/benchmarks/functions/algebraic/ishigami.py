"""Ishigami function for sensitivity analysis benchmarks.

The Ishigami function is a standard benchmark for global sensitivity analysis,
featuring non-monotonic behavior and variable interaction effects.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

import math
from typing import Generic, List

from pyapprox.util.backends.protocols import Array, Backend


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


class IshigamiSensitivityIndices(Generic[Array]):
    """Analytical sensitivity indices for the Ishigami function.

    Computes exact Sobol indices for the Ishigami function:
    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)

    with x uniformly distributed on [-pi, pi]^3.

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
        self._compute_indices()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 3

    def _compute_indices(self) -> None:
        """Compute and cache all sensitivity indices."""
        bkd = self._bkd
        a = self._a
        b = self._b
        pi = math.pi

        # Mean: E[f] = a/2 (since E[sin(x1)] = 0, E[sin^2(x2)] = 1/2)
        self._mean = bkd.asarray([a / 2.0])

        # Variance components (unnormalized Sobol indices)
        # D_1 = Var(E[f|x1]) = b*pi^4/5 + b^2*pi^8/50 + 1/2
        D_1 = b * pi**4 / 5 + b**2 * pi**8 / 50 + 0.5
        # D_2 = Var(E[f|x2]) = a^2/8
        D_2 = a**2 / 8
        # D_3 = 0 (x3 has no main effect)
        D_3 = 0.0
        # D_12 = 0 (no interaction between x1 and x2)
        D_12 = 0.0
        # D_13 = b^2*pi^8*(1/18 - 1/50)
        D_13 = b**2 * pi**8 * (1.0 / 18 - 1.0 / 50)
        # D_23 = 0 (no interaction between x2 and x3)
        D_23 = 0.0
        # D_123 = 0 (no three-way interaction)
        D_123 = 0.0

        # Total variance
        self._variance = bkd.asarray(
            [D_1 + D_2 + D_3 + D_12 + D_13 + D_23 + D_123]
        )
        var = D_1 + D_2 + D_3 + D_12 + D_13 + D_23 + D_123

        # Main effects (first-order Sobol indices): S_i = D_i / Var(f)
        self._main_effects = bkd.asarray([[D_1 / var], [D_2 / var], [D_3 / var]])

        # Total effects: ST_i = (D_i + sum of interactions involving i) / Var(f)
        # ST_1 = (D_1 + D_12 + D_13 + D_123) / var
        # ST_2 = (D_2 + D_12 + D_23 + D_123) / var
        # ST_3 = (D_3 + D_13 + D_23 + D_123) / var
        ST_1 = (D_1 + D_12 + D_13 + D_123) / var
        ST_2 = (D_2 + D_12 + D_23 + D_123) / var
        ST_3 = (D_3 + D_13 + D_23 + D_123) / var
        self._total_effects = bkd.asarray([[ST_1], [ST_2], [ST_3]])

        # All Sobol indices (main effects and interactions)
        # Order: D_1, D_2, D_3, D_12, D_13, D_23, D_123
        self._sobol_indices = bkd.asarray([
            [D_1 / var],
            [D_2 / var],
            [D_3 / var],
            [D_12 / var],
            [D_13 / var],
            [D_23 / var],
            [D_123 / var],
        ])

        # Interaction indices (binary representation of which variables are involved)
        # Shape: (nvars, nindices)
        self._interaction_indices: List[List[int]] = [
            [0],        # x1
            [1],        # x2
            [2],        # x3
            [0, 1],     # x1, x2
            [0, 2],     # x1, x3
            [1, 2],     # x2, x3
            [0, 1, 2],  # x1, x2, x3
        ]

    def mean(self) -> Array:
        """Return the mean of the Ishigami function.

        Returns
        -------
        Array
            Mean value of shape (1,).
        """
        return self._mean

    def variance(self) -> Array:
        """Return the variance of the Ishigami function.

        Returns
        -------
        Array
            Variance of shape (1,).
        """
        return self._variance

    def main_effects(self) -> Array:
        """Return the main effects (first-order Sobol indices).

        Returns
        -------
        Array
            Main effects of shape (nvars, 1).
        """
        return self._main_effects

    def total_effects(self) -> Array:
        """Return the total effects (total Sobol indices).

        Returns
        -------
        Array
            Total effects of shape (nvars, 1).
        """
        return self._total_effects

    def sobol_indices(self) -> Array:
        """Return all Sobol indices (main effects and interactions).

        Returns
        -------
        Array
            Sobol indices of shape (nindices, 1) where nindices = 7
            for the 3-variable Ishigami function.
            Order: S_1, S_2, S_3, S_12, S_13, S_23, S_123.
        """
        return self._sobol_indices

    def sobol_interaction_indices(self) -> Array:
        """Return the interaction indices as a binary matrix.

        Returns
        -------
        Array
            Binary matrix of shape (nvars, nindices) where entry (i, j)
            is 1 if variable i is involved in interaction j.
        """
        bkd = self._bkd
        nvars = self.nvars()
        nindices = len(self._interaction_indices)
        # Build as list of columns, then stack
        columns = []
        for compressed_index in self._interaction_indices:
            col = [1.0 if i in compressed_index else 0.0 for i in range(nvars)]
            columns.append(bkd.asarray(col))
        return bkd.stack(columns, axis=1)
