"""Tunable ensemble benchmark — tunable correlation structure."""

import math
from typing import Callable, Generic, List, Optional

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.multifidelity.tunable_ensemble import (
    TunableModelFunction,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)
from pyapprox_benchmarks.statest.statistics_mixin import (
    MultifidelityStatisticsMixin,
)


class TunableEnsembleBenchmark(
    MultifidelityStatisticsMixin[Array], Generic[Array]
):
    """Tunable ensemble benchmark — tunable correlation structure.

    3 models with tunable correlation:
    - m0: A0 * (cos(theta0)*x^5 + sin(theta0)*y^5)  [HF]
    - m1: A1 * (cos(theta1)*x^3 + sin(theta1)*y^3) + shift[0]  [MF]
    - m2: A2 * (cos(theta2)*x   + sin(theta2)*y)   + shift[1]  [LF]

    A0=sqrt(11), A1=sqrt(7), A2=sqrt(3) for unit variance.
    theta0=pi/2, theta2=pi/6 fixed; theta1 tunable.
    Input domain: [-1, 1]^2, uniform distribution.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    theta1 : float
        Angle controlling medium-fidelity correlation (default 1.0).
    shifts : Optional[List[float]]
        Shifts applied to models 1 and 2. Default [0, 0].
    """

    def __init__(
        self,
        bkd: Backend[Array],
        theta1: float = 1.0,
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
        self._models: List[TunableModelFunction[Array]] = self._create_models()

        # Precompute analytical covariance
        self._cov = self._compute_covariance()

        # Build problem
        prior = IndependentJoint(
            [UniformMarginal(-1.0, 1.0, bkd), UniformMarginal(-1.0, 1.0, bkd)],
            bkd,
        )
        domain = BoxDomain(
            _bounds=bkd.array([[-1.0, 1.0], [-1.0, 1.0]]), _bkd=bkd,
        )
        costs = 10.0 ** (-bkd.arange(self._nmodels))
        self._problem = MultifidelityForwardUQProblem(
            "tunable_ensemble_3model",
            list(self._models),
            costs,
            prior,
            description="3-model tunable correlation ensemble",
        )
        self._domain = domain

    def _create_models(self) -> List[TunableModelFunction[Array]]:
        """Create the model functions."""
        bkd = self._bkd

        def m0(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            result = self._A0 * (
                bkd.cos(bkd.asarray(self._theta0)) * x**5
                + bkd.sin(bkd.asarray(self._theta0)) * y**5
            )
            return result[None, :]

        def m1(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            result = (
                self._A1
                * (
                    bkd.cos(bkd.asarray(self._theta1)) * x**3
                    + bkd.sin(bkd.asarray(self._theta1)) * y**3
                )
                + self._shifts[0]
            )
            return result[None, :]

        def m2(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            result = (
                self._A2
                * (
                    bkd.cos(bkd.asarray(self._theta2)) * x
                    + bkd.sin(bkd.asarray(self._theta2)) * y
                )
                + self._shifts[1]
            )
            return result[None, :]

        return [
            TunableModelFunction(bkd, m0),
            TunableModelFunction(bkd, m1),
            TunableModelFunction(bkd, m2),
        ]

    def _compute_covariance(self) -> Array:
        """Compute analytical covariance matrix."""
        bkd = self._bkd
        cov = bkd.eye(3)
        cov_np = bkd.to_numpy(cov).copy()

        cov_np[0, 1] = (
            self._A0 * self._A1 / 9
            * (
                math.sin(self._theta0) * math.sin(self._theta1)
                + math.cos(self._theta0) * math.cos(self._theta1)
            )
        )
        cov_np[1, 0] = cov_np[0, 1]

        cov_np[0, 2] = (
            self._A0 * self._A2 / 7
            * (
                math.sin(self._theta0) * math.sin(self._theta2)
                + math.cos(self._theta0) * math.cos(self._theta2)
            )
        )
        cov_np[2, 0] = cov_np[0, 2]

        cov_np[1, 2] = (
            self._A1 * self._A2 / 5
            * (
                math.sin(self._theta1) * math.sin(self._theta2)
                + math.cos(self._theta1) * math.cos(self._theta2)
            )
        )
        cov_np[2, 1] = cov_np[1, 2]

        return bkd.asarray(cov_np)

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def domain(self) -> BoxDomain[Array]:
        return self._domain

    def ensemble_means(self) -> Array:
        """Analytical means, shape (nmodels, 1)."""
        return self._bkd.asarray(
            [[0.0], [self._shifts[0]], [self._shifts[1]]]
        )

    def ensemble_covariance(self) -> Array:
        """Analytical covariance matrix, shape (nmodels, nmodels)."""
        return self._cov
