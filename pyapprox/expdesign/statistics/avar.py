"""
Smoothed Average Value at Risk (AVaR) statistic.

AVaR (also known as CVaR or Expected Shortfall) is computed without
explicit VaR estimation using a smoothing approach.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.statistics.base import SampleStatistic


class SampleAverageSmoothedAVaR(SampleStatistic[Array], Generic[Array]):
    """
    Compute smoothed Average Value at Risk without explicit VaR estimation.

    The smoothing parameter delta controls accuracy. Larger delta produces
    more accurate estimates.

    Parameters
    ----------
    alpha : float
        Risk level in (0, 1). AVaR_alpha is the expected value given that
        outcomes exceed the alpha quantile.
    bkd : Backend[Array]
        Computational backend.
    delta : float, optional
        Smoothing parameter. Larger values give more accurate estimates.
        Default is 100.
    """

    def __init__(
        self,
        alpha: float,
        bkd: Backend[Array],
        delta: float = 100,
    ) -> None:
        super().__init__(bkd)
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._alpha = bkd.atleast_1d(bkd.asarray(alpha))
        self._delta = delta
        self._lambda = 0.0  # Regularization parameter

    def jacobian_implemented(self) -> bool:
        """Jacobian is implemented."""
        return True

    def _project(self, values: Array, weights: Array) -> Array:
        """
        Project y onto the CVaR risk envelope.

        Uses binary search to find the optimal Lagrange multiplier.

        Parameters
        ----------
        values : Array
            Scaled values to project. Shape: (nsamples,)
        weights : Array
            Probability weights. Shape: (nsamples,)

        Returns
        -------
        Array
            Projection onto CVaR risk envelope. Shape: (nsamples,)
        """
        # Compute all possible kinks
        lbnd = self._bkd.asarray(0.0)
        ubnd = 1.0 / (1.0 - self._alpha)
        dvalues = values / weights

        # Sorted kinks in descending order
        K = self._bkd.flip(
            self._bkd.sort(
                self._bkd.hstack([lbnd - dvalues, ubnd - dvalues])
            )
        )

        nsamp = len(values)

        def residual(x: Array) -> Array:
            """Compute residual for a given Lagrange multiplier."""
            return 1.0 - weights @ self._bkd.maximum(
                lbnd, self._bkd.minimum(ubnd, dvalues + x)
            )

        # Binary search for zero crossing
        ibeg = 0
        imid = nsamp
        iend = 2 * nsamp
        x1 = K[ibeg]
        y1 = residual(x1)

        # Handle edge case when alpha = 0
        if self._bkd.allclose(y1, self._bkd.zeros(1), atol=3e-16):
            if not self._bkd.allclose(
                self._alpha, self._bkd.zeros(1), atol=1e-15
            ):
                raise RuntimeError("Unexpected zero residual at boundary")
            imid = 1
            y1 = y1 * 0 - 3.0e-16

        x2 = K[imid]
        y2 = residual(x2)

        while True:
            if self._bkd.sign(y1) != self._bkd.sign(y2):
                iend = imid
            else:
                ibeg = imid
                x1 = x2
                y1 = y2

            if iend - ibeg == 1:
                imid = iend
            else:
                imid = ibeg + round((iend - ibeg) / 2)

            x2 = K[imid]
            y2 = residual(x2)

            if iend - ibeg == 1:
                break

        # Linear interpolation to find zero
        lam = (y2 * x1 - y1 * x2) / (y2 - y1)

        # Return projection
        return weights * self._bkd.maximum(
            lbnd, self._bkd.minimum(ubnd, dvalues + lam)
        )

    def _check_values_weights_single(
        self, values: Array, weights: Array
    ) -> None:
        """Validate values and weights for single QoI."""
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("values must be a 2D array with single row")
        if values.shape != weights.shape:
            raise ValueError(
                f"values shape {values.shape} must match weights shape "
                f"{weights.shape}"
            )
        total = self._bkd.sum(weights)
        if not self._bkd.allclose(
            self._bkd.ones((1,)), total, atol=1e-15
        ):
            raise ValueError(f"weights must sum to 1, got {total}")

    def _evaluate_single(self, values: Array, weights: Array) -> Array:
        """
        Evaluate AVaR for a single QoI.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (1, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            AVaR value. Shape: scalar
        """
        self._check_values_weights_single(values, weights)

        # Scale values and project
        scaled = weights[0, :] * values[0, :] * self._delta + self._lambda
        proj = self._project(scaled, weights[0, :])

        # Compute smoothed AVaR
        term1 = self._bkd.sum(proj * values[0, :])
        term2 = (
            1.0 / (2.0 * self._delta)
            * ((proj - self._lambda) / weights[0, :])
            @ (proj - self._lambda)
        )
        return term1 - term2

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute smoothed AVaR for each QoI.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            AVaR values. Shape: (nqoi, 1)
        """
        nqoi = values.shape[0]
        result = self._bkd.stack(
            [
                self._evaluate_single(values[ii:ii + 1, :], weights)
                for ii in range(nqoi)
            ]
        )
        return result[:, None]

    def _jacobian_single(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian for a single QoI.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (1, nsamples)
        jac_values : Array
            Jacobians. Shape: (nsamples, nvars)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nvars,)
        """
        scaled = weights[0, :] * values[0, :] * self._delta + self._lambda
        proj = self._project(scaled, weights[0, :])
        return self._bkd.einsum("i,ij->j", proj, jac_values)

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of AVaR.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        jac_values : Array
            Jacobians at samples. Shape: (nqoi, nsamples, nvars)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        nqoi = values.shape[0]
        return self._bkd.stack(
            [
                self._jacobian_single(
                    values[ii:ii + 1, :],
                    jac_values[ii, :, :],
                    weights,
                )
                for ii in range(nqoi)
            ]
        )

    def __repr__(self) -> str:
        alpha_val = float(self._bkd.to_numpy(self._alpha)[0])
        return f"SampleAverageSmoothedAVaR(alpha={alpha_val})"
