"""Multi-output model ensemble for multifidelity benchmarks.

Implements an ensemble of 3 models with 3 QoI each, with known
analytical covariance structure for testing multi-output estimators.

Reference: Dixon et al. (2024), SIAM/ASA JUQ
"""

import math
from typing import Callable, Generic, List, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.functions.multifidelity.statistics_mixin import (
    MultifidelityStatisticsMixin,
)


class MultiOutputModelFunction(Generic[Array]):
    """Single multi-output model.

    Implements FunctionProtocol with multiple QoI.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    func : Callable
        Function that evaluates the model.
    nqoi : int
        Number of quantities of interest.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        func: Callable[[Array], Array],
        nqoi: int,
    ) -> None:
        self._bkd = bkd
        self._func = func
        self._nqoi = nqoi

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (nqoi, nsamples).
        """
        return self._func(samples)


class MultiOutputModelEnsemble(MultifidelityStatisticsMixin[Array], Generic[Array]):
    """Ensemble of multi-output models for multifidelity testing.

    3 models with 3 QoI each:
    - f0: [sqrt(11)*x^5, x^4, sin(2*pi*x)]
    - f1: [sqrt(7)*x^3, sqrt(7)*x^2, cos(2*pi*x + pi/2)]
    - f2: [sqrt(3)/2*x^2, sqrt(3)/2*x, cos(2*pi*x + pi/4)]

    Has analytical covariance matrix for U[0,1] input. Overrides the
    mixin's numerical implementations with analytical values for efficiency.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Reference
    ---------
    Dixon et al. (2024), "Covariance Expressions for Multi-Fidelity Sampling
    with Multi-Output, Multi-Statistic Estimators", SIAM/ASA JUQ.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nmodels = 3
        self._nqoi = 3
        self._models = self._create_models()

    def _create_models(self) -> List[MultiOutputModelFunction[Array]]:
        """Create the model functions."""

        def f0(samples: Array) -> Array:
            # samples: (1, nsamples), output: (3, nsamples)
            x = samples[0, :]  # (nsamples,)
            return self._bkd.vstack([
                math.sqrt(11) * x**5,
                x**4,
                self._bkd.sin(2 * math.pi * x),
            ])

        def f1(samples: Array) -> Array:
            x = samples[0, :]
            return self._bkd.vstack([
                math.sqrt(7) * x**3,
                math.sqrt(7) * x**2,
                self._bkd.cos(2 * math.pi * x + math.pi / 2),
            ])

        def f2(samples: Array) -> Array:
            x = samples[0, :]
            return self._bkd.vstack([
                math.sqrt(3) / 2 * x**2,
                math.sqrt(3) / 2 * x,
                self._bkd.cos(2 * math.pi * x + math.pi / 4),
            ])

        return [
            MultiOutputModelFunction(self._bkd, f0, self._nqoi),
            MultiOutputModelFunction(self._bkd, f1, self._nqoi),
            MultiOutputModelFunction(self._bkd, f2, self._nqoi),
        ]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nmodels(self) -> int:
        """Return number of models in ensemble."""
        return self._nmodels

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of QoI per model."""
        return self._nqoi

    def models(self) -> Sequence[MultiOutputModelFunction[Array]]:
        """Return the list of models."""
        return self._models

    def __getitem__(self, idx: int) -> MultiOutputModelFunction[Array]:
        """Get model by index."""
        return self._models[idx]

    def costs(self) -> Array:
        """Return costs of each model.

        Returns
        -------
        Array
            Costs of shape (nmodels,).
        """
        return self._bkd.array([1.0, 0.01, 0.001])

    def means(self) -> Array:
        """Return analytical means of all models.

        Returns
        -------
        Array
            Means of shape (nmodels, nqoi).
        """
        return self._bkd.array([
            [math.sqrt(11) / 6, 1 / 5, 0.0],
            [math.sqrt(7) / 4, math.sqrt(7) / 3, 0.0],
            [1 / (2 * math.sqrt(3)), math.sqrt(3) / 4, 0.0],
        ])

    def _cov_block_11(self) -> Array:
        """Covariance matrix for model 0."""
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
        """Covariance matrix for model 1."""
        c13 = math.sqrt(7) * (-3 + 2 * math.pi**2) / (4 * math.pi**3)
        c23 = math.sqrt(7) / (2 * math.pi)
        return self._bkd.array([
            [9 / 16, 7 / 12, c13],
            [7 / 12, 28 / 45, c23],
            [c13, c23, 1 / 2],
        ])

    def _cov_block_33(self) -> Array:
        """Covariance matrix for model 2."""
        c13 = math.sqrt(3 / 2) * (1 + math.pi) / (4 * math.pi**2)
        c23 = math.sqrt(3 / 2) / (4 * math.pi)
        return self._bkd.array([
            [1 / 15, 1 / 16, c13],
            [1 / 16, 1 / 16, c23],
            [c13, c23, 1 / 2],
        ])

    def _cov_block_12(self) -> Array:
        """Cross-covariance between model 0 and model 1."""
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
        """Cross-covariance between model 0 and model 2."""
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
        """Cross-covariance between model 1 and model 2."""
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

    def covariance_matrix(self) -> Array:
        """Return full covariance matrix for all models and QoI.

        Returns
        -------
        Array
            Covariance matrix of shape (nmodels*nqoi, nmodels*nqoi).
        """
        cov11 = self._cov_block_11()
        cov22 = self._cov_block_22()
        cov33 = self._cov_block_33()
        cov12 = self._cov_block_12()
        cov13 = self._cov_block_13()
        cov23 = self._cov_block_23()

        # Build full covariance matrix
        n = self._nmodels * self._nqoi
        cov = self._bkd.zeros((n, n))

        # Diagonal blocks
        cov[:3, :3] = cov11
        cov[3:6, 3:6] = cov22
        cov[6:9, 6:9] = cov33

        # Off-diagonal blocks
        cov[:3, 3:6] = cov12
        cov[3:6, :3] = cov12.T
        cov[:3, 6:9] = cov13
        cov[6:9, :3] = cov13.T
        cov[3:6, 6:9] = cov23
        cov[6:9, 3:6] = cov23.T

        return cov

    def covariance_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> Array:
        """Extract covariance submatrix for subset of models and QoI.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        Array
            Submatrix of shape (len(model_idx)*len(qoi_idx),
                                len(model_idx)*len(qoi_idx)).
        """
        full_cov = self.covariance_matrix()
        nqoi = self._nqoi

        # Build index array for subproblem
        indices = []
        for m in model_idx:
            for q in qoi_idx:
                indices.append(m * nqoi + q)

        # Extract submatrix
        n = len(indices)
        subcov = self._bkd.zeros((n, n))
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                subcov[i, j] = full_cov[idx_i, idx_j]

        return subcov

    def models_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> List[Callable[[Array], Array]]:
        """Get model functions for subset of models and QoI.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        List[Callable]
            List of model functions that return only selected QoI.
        """
        result = []
        for m_idx in model_idx:
            base_model = self._models[m_idx]

            def make_submodel(model: MultiOutputModelFunction, qoi: List[int]):
                def submodel(samples: Array) -> Array:
                    full_output = model(samples)  # (nqoi, nsamples)
                    return self._bkd.vstack([full_output[q, :] for q in qoi])
                return submodel

            result.append(make_submodel(base_model, qoi_idx))

        return result

    def costs_subproblem(self, model_idx: List[int]) -> Array:
        """Get costs for subset of models.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.

        Returns
        -------
        Array
            Costs of shape (len(model_idx),).
        """
        all_costs = self.costs()
        return self._bkd.array([float(all_costs[i]) for i in model_idx])

    def means_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> Array:
        """Get means for subset of models and QoI.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        Array
            Means of shape (len(model_idx), len(qoi_idx)).
        """
        all_means = self.means()
        result = []
        for m in model_idx:
            row = [float(all_means[m, q]) for q in qoi_idx]
            result.append(row)
        return self._bkd.array(result)


class PSDMultiOutputModelEnsemble(MultiOutputModelEnsemble[Array]):
    """Positive Semi-Definite variant of MultiOutputModelEnsemble.

    This version adds small perturbation terms (eps) to the model functions
    to ensure the resulting covariance matrices are better conditioned.
    This is useful for testing optimization-based estimators where
    ill-conditioned covariance matrices can cause numerical issues.

    3 models with 3 QoI each (perturbed versions):
    - f0: [sqrt(11)*x^5, x^4 + eps0*cos(2.2*pi*x), sin(2*pi*x)]
    - f1: [sqrt(7)*x^3, sqrt(7)*x^2, cos((2+eps1)*pi*x + pi/2)]
    - f2: [sqrt(3)/2*x^2 + x, sqrt(3)/2*x + eps2*cos(2*pi*x + 2.1), cos(2*pi*x + pi/4)]

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Reference
    ---------
    Dixon et al. (2024), "Covariance Expressions for Multi-Fidelity Sampling
    with Multi-Output, Multi-Statistic Estimators", SIAM/ASA JUQ.
    """

    def _create_models(self) -> List[MultiOutputModelFunction[Array]]:
        """Create the PSD model functions with perturbation terms."""
        eps0 = 1.0  # perturbation for f0 qoi 1
        eps1 = 1e-1  # perturbation for f1 qoi 2
        eps2 = 1e-2  # perturbation for f2 qoi 1

        def f0(samples: Array) -> Array:
            # samples: (1, nsamples), output: (3, nsamples)
            x = samples[0, :]  # (nsamples,)
            return self._bkd.vstack([
                math.sqrt(11) * x**5,
                x**4 + eps0 * self._bkd.cos(2.2 * math.pi * x),
                self._bkd.sin(2 * math.pi * x),
            ])

        def f1(samples: Array) -> Array:
            x = samples[0, :]
            return self._bkd.vstack([
                math.sqrt(7) * x**3,
                math.sqrt(7) * x**2,
                self._bkd.cos((2 + eps1) * math.pi * x + math.pi / 2),
            ])

        def f2(samples: Array) -> Array:
            x = samples[0, :]
            return self._bkd.vstack([
                math.sqrt(3) / 2 * x**2 + x,
                math.sqrt(3) / 2 * x
                + self._bkd.cos(math.pi * x * 2.0 + 2.1) * eps2,
                self._bkd.cos(2 * math.pi * x + math.pi / 4),
            ])

        return [
            MultiOutputModelFunction(self._bkd, f0, self._nqoi),
            MultiOutputModelFunction(self._bkd, f1, self._nqoi),
            MultiOutputModelFunction(self._bkd, f2, self._nqoi),
        ]

    # Note: PSD version uses the mixin's numerical covariance_matrix() instead
    # of the parent's analytical version. We need to explicitly skip the
    # parent's analytical override by calling the mixin's method.
    def covariance_matrix(self) -> Array:
        """Use numerical quadrature since no analytical formula exists."""
        return MultifidelityStatisticsMixin.covariance_matrix(self)

    def means(self) -> Array:
        """Use numerical quadrature since no analytical formula exists."""
        return MultifidelityStatisticsMixin.means(self)
