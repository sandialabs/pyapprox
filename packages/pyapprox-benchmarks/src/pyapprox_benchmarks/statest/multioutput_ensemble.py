"""Multi-output ensemble benchmark — analytical or numerical statistics."""

import math
from typing import Callable, Generic, List

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelFunction,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)
from pyapprox_benchmarks.statest.statistics_mixin import (
    MultifidelityStatisticsMixin,
)


class MultiOutputEnsembleBenchmark(
    MultifidelityStatisticsMixin[Array], Generic[Array]
):
    """Multi-output ensemble benchmark.

    3 models with 3 QoI each. Analytical covariance for the standard
    variant; numerical quadrature (via mixin) for the PSD variant.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    psd : bool
        Use PSD variant with numerical statistics (default False).

    Reference
    ---------
    Dixon et al. (2024), "Covariance Expressions for Multi-Fidelity Sampling
    with Multi-Output, Multi-Statistic Estimators", SIAM/ASA JUQ.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        psd: bool = False,
    ) -> None:
        self._bkd = bkd
        self._psd = psd
        self._nmodels = 3
        self._nqoi = 3

        if psd:
            self._models = self._create_psd_models()
            name = "psd_multioutput_ensemble_3x3"
        else:
            self._models = self._create_models()
            name = "multioutput_ensemble_3x3"

        prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)
        domain = BoxDomain(_bounds=bkd.array([[0.0, 1.0]]), _bkd=bkd)
        costs = bkd.array([1.0, 0.01, 0.001])
        self._problem = MultifidelityForwardUQProblem(
            name,
            list(self._models),
            costs,
            prior,
            description=(
                f"3-model 3-QoI {'PSD ' if psd else ''}multi-output ensemble"
            ),
        )
        self._domain = domain

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_models(self) -> List[MultiOutputModelFunction[Array]]:
        """Create the standard model functions."""
        bkd = self._bkd

        def f0(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(11) * x**5,
                x**4,
                bkd.sin(2 * math.pi * x),
            ])

        def f1(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(7) * x**3,
                math.sqrt(7) * x**2,
                bkd.cos(2 * math.pi * x + math.pi / 2),
            ])

        def f2(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(3) / 2 * x**2,
                math.sqrt(3) / 2 * x,
                bkd.cos(2 * math.pi * x + math.pi / 4),
            ])

        return [
            MultiOutputModelFunction(bkd, f0, self._nqoi),
            MultiOutputModelFunction(bkd, f1, self._nqoi),
            MultiOutputModelFunction(bkd, f2, self._nqoi),
        ]

    def _create_psd_models(self) -> List[MultiOutputModelFunction[Array]]:
        """Create the PSD model functions with perturbation terms."""
        bkd = self._bkd
        eps0 = 1.0
        eps1 = 1e-1
        eps2 = 1e-2

        def f0(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(11) * x**5,
                x**4 + eps0 * bkd.cos(2.2 * math.pi * x),
                bkd.sin(2 * math.pi * x),
            ])

        def f1(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(7) * x**3,
                math.sqrt(7) * x**2,
                bkd.cos((2 + eps1) * math.pi * x + math.pi / 2),
            ])

        def f2(samples: Array) -> Array:
            x = samples[0, :]
            return bkd.vstack([
                math.sqrt(3) / 2 * x**2 + x,
                math.sqrt(3) / 2 * x
                + bkd.cos(math.pi * x * 2.0 + 2.1) * eps2,
                bkd.cos(2 * math.pi * x + math.pi / 4),
            ])

        return [
            MultiOutputModelFunction(bkd, f0, self._nqoi),
            MultiOutputModelFunction(bkd, f1, self._nqoi),
            MultiOutputModelFunction(bkd, f2, self._nqoi),
        ]

    # ------------------------------------------------------------------
    # Problem / domain access
    # ------------------------------------------------------------------

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def domain(self) -> BoxDomain[Array]:
        return self._domain

    # ------------------------------------------------------------------
    # Statistics (analytical overrides for non-PSD variant)
    # ------------------------------------------------------------------

    def ensemble_means(self) -> Array:
        """Means, shape (nmodels, nqoi)."""
        return self.means()

    def ensemble_covariance(self) -> Array:
        """Covariance matrix, shape (nmodels*nqoi, nmodels*nqoi)."""
        return self.covariance_matrix()

    def means(self) -> Array:
        """Analytical means for standard variant, numerical for PSD."""
        if self._psd:
            return MultifidelityStatisticsMixin.means(self)
        return self._bkd.array([
            [math.sqrt(11) / 6, 1 / 5, 0.0],
            [math.sqrt(7) / 4, math.sqrt(7) / 3, 0.0],
            [1 / (2 * math.sqrt(3)), math.sqrt(3) / 4, 0.0],
        ])

    def covariance_matrix(self) -> Array:
        """Analytical covariance for standard, numerical for PSD."""
        if self._psd:
            return MultifidelityStatisticsMixin.covariance_matrix(self)
        return self._build_analytical_covariance()

    # ------------------------------------------------------------------
    # Subproblem helpers
    # ------------------------------------------------------------------

    def models_subproblem(
        self, model_idx: List[int], qoi_idx: List[int],
    ) -> List[Callable[[Array], Array]]:
        """Get model functions for subset of models and QoI."""
        bkd = self._bkd
        result: List[Callable[[Array], Array]] = []
        for m_idx in model_idx:
            base_model = self._models[m_idx]

            def make_submodel(
                model: MultiOutputModelFunction[Array], qoi: List[int],
            ) -> Callable[[Array], Array]:
                def submodel(samples: Array) -> Array:
                    full_output = model(samples)
                    return bkd.vstack([full_output[q, :] for q in qoi])
                return submodel

            result.append(make_submodel(base_model, qoi_idx))
        return result

    def costs_subproblem(self, model_idx: List[int]) -> Array:
        """Get costs for subset of models."""
        all_costs = self._problem.costs()
        return self._bkd.array([float(all_costs[i]) for i in model_idx])

    def means_subproblem(
        self, model_idx: List[int], qoi_idx: List[int],
    ) -> Array:
        """Get means for subset of models and QoI."""
        all_means = self.means()
        result = []
        for m in model_idx:
            row = [float(all_means[m, q]) for q in qoi_idx]
            result.append(row)
        return self._bkd.array(result)

    # ------------------------------------------------------------------
    # Analytical covariance blocks
    # ------------------------------------------------------------------

    def _build_analytical_covariance(self) -> Array:
        """Build full 9x9 analytical covariance matrix."""
        cov11 = self._cov_block_11()
        cov22 = self._cov_block_22()
        cov33 = self._cov_block_33()
        cov12 = self._cov_block_12()
        cov13 = self._cov_block_13()
        cov23 = self._cov_block_23()

        n = self._nmodels * self._nqoi
        cov = self._bkd.zeros((n, n))
        cov[:3, :3] = cov11
        cov[3:6, 3:6] = cov22
        cov[6:9, 6:9] = cov33
        cov[:3, 3:6] = cov12
        cov[3:6, :3] = cov12.T
        cov[:3, 6:9] = cov13
        cov[6:9, :3] = cov13.T
        cov[3:6, 6:9] = cov23
        cov[6:9, 3:6] = cov23.T
        return cov

    def _cov_block_11(self) -> Array:
        c13 = (
            -math.sqrt(11)
            * (15 - 10 * math.pi**2 + 2 * math.pi**4)
            / (4 * math.pi**5)
        )
        c23 = (3 - math.pi**2) / (2 * math.pi**3)
        return self._bkd.array([
            [25 / 36, math.sqrt(11) / 15.0, c13],
            [math.sqrt(11) / 15.0, 16 / 225, c23],
            [c13, c23, 1 / 2],
        ])

    def _cov_block_22(self) -> Array:
        c13 = math.sqrt(7) * (-3 + 2 * math.pi**2) / (4 * math.pi**3)
        c23 = math.sqrt(7) / (2 * math.pi)
        return self._bkd.array([
            [9 / 16, 7 / 12, c13],
            [7 / 12, 28 / 45, c23],
            [c13, c23, 1 / 2],
        ])

    def _cov_block_33(self) -> Array:
        c13 = math.sqrt(3 / 2) * (1 + math.pi) / (4 * math.pi**2)
        c23 = math.sqrt(3 / 2) / (4 * math.pi)
        return self._bkd.array([
            [1 / 15, 1 / 16, c13],
            [1 / 16, 1 / 16, c23],
            [c13, c23, 1 / 2],
        ])

    def _cov_block_12(self) -> Array:
        c13 = (
            math.sqrt(11)
            * (15 - 10 * math.pi**2 + 2 * math.pi**4)
            / (4 * math.pi**5)
        )
        c31 = math.sqrt(7) * (3 - 2 * math.pi**2) / (4 * math.pi**3)
        return self._bkd.array([
            [5 * math.sqrt(77) / 72, 5 * math.sqrt(77) / 72, c13],
            [
                3 * math.sqrt(7) / 40,
                8 / (15 * math.sqrt(7)),
                (-3 + math.pi**2) / (2 * math.pi**3),
            ],
            [c31, -math.sqrt(7) / (2 * math.pi), -1 / 2],
        ])

    def _cov_block_13(self) -> Array:
        c13 = (
            math.sqrt(11 / 2)
            * (
                15
                + math.pi
                * (-15 + math.pi * (-10 + math.pi * (5 + 2 * math.pi)))
            )
            / (4 * math.pi**5)
        )
        c23 = (-3 + (-1 + math.pi) * math.pi * (3 + math.pi)) / (
            2 * math.sqrt(2) * math.pi**4
        )
        return self._bkd.array([
            [5 * math.sqrt(11 / 3) / 48, 5 * math.sqrt(11 / 3) / 56, c13],
            [4 / (35 * math.sqrt(3)), 1 / (10 * math.sqrt(3)), c23],
            [
                -math.sqrt(3) / (4 * math.pi),
                -math.sqrt(3) / (4 * math.pi),
                -1 / (2 * math.sqrt(2)),
            ],
        ])

    def _cov_block_23(self) -> Array:
        c13 = (
            math.sqrt(7 / 2)
            * (-3 + 3 * math.pi + 2 * math.pi**2)
            / (4 * math.pi**3)
        )
        c23 = math.sqrt(7 / 2) * (1 + math.pi) / (2 * math.pi**2)
        return self._bkd.array([
            [math.sqrt(7 / 3) / 8, 3 * math.sqrt(21) / 80, c13],
            [2 * math.sqrt(7 / 3) / 15, math.sqrt(7 / 3) / 8, c23],
            [
                math.sqrt(3) / (4 * math.pi),
                math.sqrt(3) / (4 * math.pi),
                1 / (2 * math.sqrt(2)),
            ],
        ])
