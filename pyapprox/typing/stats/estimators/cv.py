"""Control Variate estimator (two models).

Uses a single low-fidelity model as a control variate to reduce variance.
"""

from typing import Generic, List, Callable, Optional, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.base import AbstractEstimator
from pyapprox.typing.stats.estimators.bootstrap import BootstrapMixin


class CVEstimator(BootstrapMixin[Array], AbstractEstimator[Array], Generic[Array]):
    """Control Variate estimator using one low-fidelity model.

    The CV estimator uses a low-fidelity model as a control variate:
        Q_CV = Q_0 + eta * (mu_1 - Q_1)

    where:
    - Q_0 is the HF sample mean (on shared samples)
    - Q_1 is the LF sample mean (on shared samples)
    - mu_1 is the LF sample mean (on all LF samples)
    - eta is the optimal weight

    The optimal weight is:
        eta = Cov(Q_0, Q_1) / Var(Q_1)

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (2,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> # Set pilot covariance (models correlated with rho=0.9)
    >>> cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0])
    >>> cv = CVEstimator(stat, costs, bkd)
    >>> cv.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        # Ensure we have exactly 2 costs
        costs_np = bkd.to_numpy(costs)
        if costs_np.shape[0] != 2:
            raise ValueError(
                f"CVEstimator requires exactly 2 model costs, got {costs_np.shape[0]}"
            )

        super().__init__(stat, costs, bkd)

        self._weights: Optional[Array] = None
        self._npartition_samples: Optional[Array] = None

    def _compute_optimal_allocation(
        self, target_cost: float
    ) -> tuple:
        """Compute optimal sample allocation.

        Returns
        -------
        n0 : int
            HF-only samples (partition 0)
        n1 : int
            Shared samples (partition 1)
        n2 : int
            LF-only samples (partition 2)
        """
        bkd = self._bkd
        cov = self._stat.cov()
        costs = bkd.to_numpy(self._costs)
        cov_np = bkd.to_numpy(cov)

        c0, c1 = costs[0], costs[1]
        nqoi = self._stat.nqoi()

        # Extract covariance blocks
        sigma00 = cov_np[0, 0] if nqoi == 1 else cov_np[:nqoi, :nqoi]
        sigma01 = cov_np[0, 1] if nqoi == 1 else cov_np[:nqoi, nqoi:2*nqoi]
        sigma11 = cov_np[1, 1] if nqoi == 1 else cov_np[nqoi:2*nqoi, nqoi:2*nqoi]

        # For scalar case
        if nqoi == 1:
            sigma00 = float(sigma00)
            sigma01 = float(sigma01)
            sigma11 = float(sigma11)

            # Correlation coefficient
            rho = sigma01 / np.sqrt(sigma00 * sigma11) if sigma00 > 0 and sigma11 > 0 else 0

            # Optimal ratio r = n1/n0 where n1 = LF samples, n0 = HF samples
            # From optimization: r = sqrt((c0/c1) * (1 - rho^2)) if rho^2 < 1
            rho_sq = rho ** 2
            if rho_sq >= 1:
                # Perfect correlation: use only shared samples
                r = 1.0
            else:
                r = np.sqrt((c0 / c1) * rho_sq / (1 - rho_sq)) if rho_sq > 0 else 1.0

            # Total samples determined by budget
            # Cost = n_shared * (c0 + c1) + n_lf_only * c1
            # For CV: n_shared = n_hf, n_lf = r * n_hf
            # n_lf_only = n_lf - n_shared = (r-1) * n_hf if r > 1, else 0

            # Simplified: all HF on shared, extra LF on partition 2
            # Cost = nhf * c0 + nlf * c1 = nhf * c0 + r * nhf * c1
            nhf = target_cost / (c0 + r * c1)
            nhf = max(int(nhf), self._stat.min_nsamples())

            nlf = int(r * nhf)
            nlf = max(nlf, nhf)  # LF should have at least as many as HF

            # Partitions: P0 = HF only (0), P1 = shared (nhf), P2 = LF only (nlf - nhf)
            n0 = 0  # No HF-only samples in standard CV
            n1 = nhf  # Shared
            n2 = nlf - nhf  # LF only
        else:
            # Multi-QoI case: use trace-based optimization
            # Simplified: use average correlation
            trace_sigma00 = np.trace(sigma00) if sigma00.ndim > 0 else sigma00
            trace_sigma11 = np.trace(sigma11) if sigma11.ndim > 0 else sigma11

            if isinstance(sigma01, np.ndarray) and sigma01.ndim > 0:
                avg_cov = np.mean(np.diag(sigma01))
            else:
                avg_cov = float(sigma01)

            avg_var0 = trace_sigma00 / nqoi
            avg_var1 = trace_sigma11 / nqoi

            rho = avg_cov / np.sqrt(avg_var0 * avg_var1) if avg_var0 > 0 and avg_var1 > 0 else 0
            rho_sq = rho ** 2

            if rho_sq >= 1:
                r = 1.0
            else:
                r = np.sqrt((c0 / c1) * rho_sq / (1 - rho_sq)) if rho_sq > 0 else 1.0

            nhf = target_cost / (c0 + r * c1)
            nhf = max(int(nhf), self._stat.min_nsamples())

            nlf = int(r * nhf)
            nlf = max(nlf, nhf)

            n0 = 0
            n1 = nhf
            n2 = nlf - nhf

        return n0, n1, max(n2, 0)

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples optimally between models.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        n0, n1, n2 = self._compute_optimal_allocation(target_cost)

        # Store partition samples
        self._npartition_samples = self._bkd.asarray(
            [n0, n1, n2], dtype=self._bkd.int64_dtype()
        )

        # Compute samples per model
        nhf = n0 + n1  # HF on P0 and P1
        nlf = n1 + n2  # LF on P1 and P2

        self._nsamples = self._bkd.asarray(
            [nhf, nlf], dtype=self._bkd.int64_dtype()
        )

        # Compute optimal weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute optimal control variate weights."""
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)
        nqoi = self._stat.nqoi()

        # Extract blocks
        if nqoi == 1:
            sigma01 = cov_np[0, 1]
            sigma11 = cov_np[1, 1]
            eta = sigma01 / sigma11 if sigma11 > 0 else 0.0
            self._weights = bkd.asarray([eta])
        else:
            sigma01 = cov_np[:nqoi, nqoi:2*nqoi]
            sigma11 = cov_np[nqoi:2*nqoi, nqoi:2*nqoi]

            # Solve for weights: eta = sigma01 @ sigma11^{-1}
            try:
                eta = np.linalg.solve(sigma11.T, sigma01.T).T
            except np.linalg.LinAlgError:
                eta = np.zeros((nqoi, nqoi))

            # For diagonal case, take diagonal weights
            self._weights = bkd.asarray(np.diag(eta))

    def weights(self) -> Array:
        """Return optimal control variate weights.

        Returns
        -------
        Array
            Weights. Shape: (nqoi,) or (nstats,)
        """
        if self._weights is None:
            raise ValueError("Weights not computed. Call allocate_samples() first.")
        return self._weights

    def npartition_samples(self) -> Array:
        """Return samples per partition.

        Returns
        -------
        Array
            Partition samples [n0, n1, n2]. Shape: (3,)
        """
        if self._npartition_samples is None:
            raise ValueError("Samples not allocated.")
        return self._npartition_samples

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for both models.

        The sample structure is:
        - HF samples: first nhf samples
        - LF samples: first n1 samples are shared with HF, remaining n2 are LF-only

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            [hf_samples, lf_samples]
        """
        nsamples = self.nsamples_per_model()
        nsamples_np = self._bkd.to_numpy(nsamples)
        nhf, nlf = int(nsamples_np[0]), int(nsamples_np[1])

        # Generate all LF samples (includes shared + LF-only)
        all_samples = rvs(nlf)

        # HF uses first nhf samples (shared)
        hf_samples = all_samples[:nhf]

        return [hf_samples, all_samples]

    def __call__(self, values: List[Array]) -> Array:
        """Compute control variate estimate.

        Q_CV = Q_0 + eta * (mu_1 - Q_1)

        where Q_0 and Q_1 are computed on shared samples,
        and mu_1 is the LF mean on all LF samples.

        Parameters
        ----------
        values : List[Array]
            [hf_values, lf_values]
            hf_values: shape (nhf, nqoi) - HF on shared samples
            lf_values: shape (nlf, nqoi) - LF on all samples

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        if len(values) != 2:
            raise ValueError(
                f"CVEstimator expects 2 model outputs, got {len(values)}"
            )

        bkd = self._bkd
        hf_values, lf_values = values

        nhf = hf_values.shape[0]

        # Q_0: HF mean on shared samples
        Q0 = bkd.sum(hf_values, axis=0) / nhf

        # Q_1: LF mean on shared samples (first nhf)
        Q1 = bkd.sum(lf_values[:nhf], axis=0) / nhf

        # mu_1: LF mean on all samples
        nlf = lf_values.shape[0]
        mu1 = bkd.sum(lf_values, axis=0) / nlf

        # Control variate estimate
        eta = self.weights()

        # Q_CV = Q_0 + eta * (mu_1 - Q_1)
        result = Q0 + eta * (mu1 - Q1)

        return result

    def optimized_covariance(self) -> Array:
        """Return covariance of the CV estimator.

        Var(Q_CV) = Var(Q_0) - eta^2 * Var(Q_1) * (1 - n1/nlf)
                  ≈ Var(Q_0) * (1 - rho^2) for optimal eta

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        bkd = self._bkd
        nsamples = self.nsamples_per_model()
        nsamples_np = bkd.to_numpy(nsamples)
        nhf = int(nsamples_np[0])

        npart = self.npartition_samples()
        CF, cf = self._stat.get_cv_discrepancy_covariances(npart)

        # Simplified variance computation
        # Var(Q_CV) ≈ Var(Q_0)/nhf * (1 - rho^2)
        hf_cov = self._stat.high_fidelity_estimator_covariance(nhf)

        eta = bkd.to_numpy(self.weights())
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            # Scalar case
            cov = self._stat.cov()
            cov_np = bkd.to_numpy(cov)
            rho_sq = cov_np[0, 1]**2 / (cov_np[0, 0] * cov_np[1, 1])
            var_reduction = 1 - rho_sq
            return bkd.asarray(bkd.to_numpy(hf_cov) * var_reduction)
        else:
            # Multi-QoI: approximate
            return hf_cov

    def variance_reduction(self) -> float:
        """Return variance reduction factor compared to MC.

        Returns
        -------
        float
            Ratio Var(Q_CV) / Var(Q_MC).
        """
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            rho_sq = cov_np[0, 1]**2 / (cov_np[0, 0] * cov_np[1, 1])
            return 1 - rho_sq
        else:
            # Average variance reduction
            return 0.5  # Placeholder

    def _compute_optimal_weights(self) -> Array:
        """Compute optimal control variate weights.

        This method is called by BootstrapMixin during weight resampling.

        Returns
        -------
        Array
            Optimal weights.
        """
        self._compute_weights()
        return self._weights  # type: ignore

    def _estimate_with_weights(
        self, values_per_model: List[Array], weights: Any
    ) -> Array:
        """Compute estimate using specified weights.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values.
        weights : Any
            Control variate weights. If None, uses stored weights.

        Returns
        -------
        Array
            Estimated statistic.
        """
        if weights is None:
            return self(values_per_model)

        bkd = self._bkd
        hf_values, lf_values = values_per_model

        nhf = hf_values.shape[0]

        # Q_0: HF mean on shared samples
        Q0 = bkd.sum(hf_values, axis=0) / nhf

        # Q_1: LF mean on shared samples (first nhf)
        Q1 = bkd.sum(lf_values[:nhf], axis=0) / nhf

        # mu_1: LF mean on all samples
        nlf = lf_values.shape[0]
        mu1 = bkd.sum(lf_values, axis=0) / nlf

        # Control variate estimate with provided weights
        result = Q0 + weights * (mu1 - Q1)

        return result

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = f"[{int(ns[0])}, {int(ns[1])}]"
        return f"CVEstimator(nsamples={nsamples_str})"
