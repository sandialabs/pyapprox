"""Tunable model ensemble for multifidelity benchmarks.

This module implements a typing-compatible version of the legacy
TunableModelEnsembleBenchmark, following the same mathematical structure
but using the typing Backend protocol.
"""

import math
from typing import Callable, Generic, List, Optional, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.functions.multifidelity.statistics_mixin import (
    MultifidelityStatisticsMixin,
)


class TunableModelFunction(Generic[Array]):
    """Single model from the tunable ensemble.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    func : Callable
        Function that evaluates the model.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        func: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._func = func

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 2

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (2, nsamples).

        Returns
        -------
        Array
            Values of shape (nqoi, nsamples) = (1, nsamples).
        """
        return self._func(samples)


class TunableModelEnsemble(MultifidelityStatisticsMixin[Array], Generic[Array]):
    """Tunable model ensemble for multifidelity testing.

    Implements an ensemble of 3 models with tunable correlation structure,
    matching the legacy TunableModelEnsembleBenchmark.

    The models are:
    - m0: A0 * (cos(theta0)*x^5 + sin(theta0)*y^5)  [HF]
    - m1: A1 * (cos(theta1)*x^3 + sin(theta1)*y^3) + shift[0]  [MF]
    - m2: A2 * (cos(theta2)*x + sin(theta2)*y) + shift[1]  [LF]

    where A0=sqrt(11), A1=sqrt(7), A2=sqrt(3) ensure unit variance,
    theta0=pi/2, theta2=pi/6 are fixed, and theta1 controls correlation.

    Input domain: [-1, 1]^2 with uniform distribution.

    Parameters
    ----------
    theta1 : float
        Angle controlling the second model's fidelity. Must satisfy
        theta0 > theta1 > theta2 (i.e., pi/6 < theta1 < pi/2).
    bkd : Backend[Array]
        Backend for array operations.
    shifts : Optional[List[float]]
        Shifts applied to the second and third models. Default is [0, 0].
    """

    def __init__(
        self,
        theta1: float,
        bkd: Backend[Array],
        shifts: Optional[List[float]] = None,
    ) -> None:
        self._bkd = bkd
        self._theta1 = theta1

        if shifts is None:
            shifts = [0.0, 0.0]
        self._shifts = shifts

        # Fixed angles
        self._theta0 = math.pi / 2
        self._theta2 = math.pi / 6

        # Coefficients ensuring unit variance
        self._A0 = math.sqrt(11)
        self._A1 = math.sqrt(7)
        self._A2 = math.sqrt(3)

        self._nmodels = 3
        self._nqoi = 1
        self._models = self._create_models()

        # Precompute analytical covariance
        self._cov = self._compute_covariance()

    def _create_models(self) -> List[TunableModelFunction[Array]]:
        """Create the model functions."""

        def m0(samples: Array) -> Array:
            """Highest fidelity model (degree 5 polynomial)."""
            x, y = samples[0, :], samples[1, :]
            result = self._A0 * (
                self._bkd.cos(self._bkd.asarray(self._theta0)) * x**5
                + self._bkd.sin(self._bkd.asarray(self._theta0)) * y**5
            )
            # Typing convention: (nqoi, nsamples) = (1, nsamples)
            return result[None, :]

        def m1(samples: Array) -> Array:
            """Second highest fidelity model (degree 3 polynomial)."""
            x, y = samples[0, :], samples[1, :]
            result = self._A1 * (
                self._bkd.cos(self._bkd.asarray(self._theta1)) * x**3
                + self._bkd.sin(self._bkd.asarray(self._theta1)) * y**3
            ) + self._shifts[0]
            # Typing convention: (nqoi, nsamples) = (1, nsamples)
            return result[None, :]

        def m2(samples: Array) -> Array:
            """Lowest fidelity model (degree 1 polynomial)."""
            x, y = samples[0, :], samples[1, :]
            result = self._A2 * (
                self._bkd.cos(self._bkd.asarray(self._theta2)) * x
                + self._bkd.sin(self._bkd.asarray(self._theta2)) * y
            ) + self._shifts[1]
            # Typing convention: (nqoi, nsamples) = (1, nsamples)
            return result[None, :]

        return [
            TunableModelFunction(self._bkd, m0),
            TunableModelFunction(self._bkd, m1),
            TunableModelFunction(self._bkd, m2),
        ]

    def _compute_covariance(self) -> Array:
        """Compute analytical covariance matrix."""
        bkd = self._bkd
        cov = bkd.eye(3)

        # Convert to numpy for mutation, then back to array
        cov_np = bkd.to_numpy(cov).copy()

        # Cov(m0, m1)
        cov_np[0, 1] = (
            self._A0 * self._A1 / 9 * (
                math.sin(self._theta0) * math.sin(self._theta1)
                + math.cos(self._theta0) * math.cos(self._theta1)
            )
        )
        cov_np[1, 0] = cov_np[0, 1]

        # Cov(m0, m2)
        cov_np[0, 2] = (
            self._A0 * self._A2 / 7 * (
                math.sin(self._theta0) * math.sin(self._theta2)
                + math.cos(self._theta0) * math.cos(self._theta2)
            )
        )
        cov_np[2, 0] = cov_np[0, 2]

        # Cov(m1, m2)
        cov_np[1, 2] = (
            self._A1 * self._A2 / 5 * (
                math.sin(self._theta1) * math.sin(self._theta2)
                + math.cos(self._theta1) * math.cos(self._theta2)
            )
        )
        cov_np[2, 1] = cov_np[1, 2]

        return bkd.asarray(cov_np)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nmodels(self) -> int:
        """Return number of models in ensemble."""
        return self._nmodels

    def nvars(self) -> int:
        """Return number of input variables."""
        return 2

    def nqoi(self) -> int:
        """Return number of QoI per model."""
        return self._nqoi

    def models(self) -> Sequence[TunableModelFunction[Array]]:
        """Return the list of models."""
        return self._models

    def covariance(self) -> Array:
        """Return the analytical covariance matrix.

        Returns
        -------
        Array
            Covariance matrix of shape (nmodels, nmodels).
        """
        return self._cov

    def means(self) -> Array:
        """Return the true means for each model.

        Returns
        -------
        Array
            Means of shape (nmodels, 1).
        """
        return self._bkd.asarray([
            [0.0],
            [self._shifts[0]],
            [self._shifts[1]]
        ])

    def costs(self) -> Array:
        """Return default costs for each model.

        Returns
        -------
        Array
            Costs of shape (nmodels,).
        """
        return 10.0 ** (-self._bkd.arange(self._nmodels))

    def rvs(self, nsamples: int) -> Array:
        """Generate random samples from U[-1, 1]^2.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples of shape (2, nsamples).
        """
        import numpy as np
        samples = np.random.uniform(-1, 1, (2, nsamples))
        return self._bkd.asarray(samples)
