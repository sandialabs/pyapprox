"""Multi-output mean statistic for multifidelity estimation.

Computes sample means and their covariance structure for control variate
estimators.
"""

from typing import Generic, List, Tuple, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.statistics.base import AbstractStatistic


class MultiOutputMean(AbstractStatistic[Array], Generic[Array]):
    """Multi-output mean statistic.

    Computes the sample mean for multiple quantities of interest and provides
    the covariance structure needed for multifidelity estimation.

    The statistic is:
        mu = E[Q]

    where Q is the model output vector of shape (nqoi,).

    For control variate estimation, we need the covariance between model
    outputs across all models:
        Cov[Q_i, Q_j] for models i, j

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=2, bkd=bkd)
    >>> values = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> stat.sample_estimate(values)  # Returns [3.0, 4.0]
    """

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._means: Optional[Array] = None

    def nstats(self) -> int:
        """Return total number of scalar statistics.

        For mean, nstats = nqoi.
        """
        return self._nqoi

    def min_nsamples(self) -> int:
        """Return minimum number of samples needed.

        For mean, minimum is 1.
        """
        return 1

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample mean.

        Parameters
        ----------
        values : Array
            Model output samples. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Sample mean. Shape: (nqoi,)
        """
        self._validate_values(values, min_samples=1)
        return self._bkd.sum(values, axis=0) / values.shape[0]

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array]:
        """Compute pilot covariance and means from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        cov : Array
            Cross-model covariance. Shape: (nmodels*nqoi, nmodels*nqoi)
        means : Array
            Sample means. Shape: (nmodels, nqoi)
        """
        nmodels = len(pilot_values)

        # Validate all pilot values have same shape
        npilot = pilot_values[0].shape[0]
        for m, vals in enumerate(pilot_values):
            if vals.shape[0] != npilot:
                raise ValueError(
                    f"All pilot samples must have same size. "
                    f"Model 0 has {npilot}, model {m} has {vals.shape[0]}"
                )
            if vals.shape[1] != self._nqoi:
                raise ValueError(
                    f"Model {m} has {vals.shape[1]} QoI, expected {self._nqoi}"
                )

        # Stack all values: shape (npilot, nmodels * nqoi)
        stacked = self._bkd.concatenate(pilot_values, axis=1)

        # Compute covariance
        cov = self._compute_covariance(stacked)

        # Compute means
        means = self._bkd.stack(
            [self._bkd.sum(v, axis=0) / v.shape[0] for v in pilot_values], axis=0
        )

        return cov, means

    def set_pilot_quantities(self, cov: Array, means: Optional[Array] = None) -> None:
        """Set pilot covariance and means directly.

        Parameters
        ----------
        cov : Array
            Cross-model covariance. Shape: (nmodels*nqoi, nmodels*nqoi)
        means : Array, optional
            Sample means. Shape: (nmodels, nqoi)
        """
        self._cov = cov
        self._means = means

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity sample mean.

        The covariance of the sample mean is:
            Cov[mean(Q)] = Var(Q) / n

        Parameters
        ----------
        nhf_samples : int
            Number of high-fidelity samples.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nqoi, nqoi)
        """
        cov = self.cov()

        # Extract HF block (first nqoi x nqoi)
        hf_cov = cov[:self._nqoi, :self._nqoi]

        return hf_cov / nhf_samples

    def _compute_covariance(self, values: Array) -> Array:
        """Compute sample covariance matrix.

        Parameters
        ----------
        values : Array
            Samples. Shape: (nsamples, nvars)

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars)
        """
        # Center the data
        nsamples = values.shape[0]
        mean = self._bkd.sum(values, axis=0) / nsamples
        centered = values - mean

        # Compute covariance with Bessel correction
        cov = centered.T @ centered / (nsamples - 1)

        return cov

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for standard CV estimator.

        For a 2-model CV estimator:
            Q_CV = Q_0 + eta * (mu_1 - Q_1)

        We need:
        - CF: Cov[Q_0, Q_1 - mu_1] = Cov[Q_0, Q_1]
        - cf: Var[Q_1 - mu_1] = Var[Q_1] / n_shared + Var[mu_1] terms

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in each partition. Shape: (npartitions,)

        Returns
        -------
        CF : Array
            Covariance between HF estimator and LF control.
            Shape: (nqoi, nqoi)
        cf : Array
            Variance of LF control variate estimator.
            Shape: (nqoi, nqoi)
        """
        cov = self.cov()
        nqoi = self._nqoi

        # For 2-model CV: partitions are [HF only, HF+LF]
        # n0 = HF only samples, n1 = shared samples
        n0 = npartition_samples[0]
        n1 = npartition_samples[1]

        # Extract covariance blocks
        C00 = cov[:nqoi, :nqoi]  # Var(Q_0)
        C01 = cov[:nqoi, nqoi:2*nqoi]  # Cov(Q_0, Q_1)
        C11 = cov[nqoi:2*nqoi, nqoi:2*nqoi]  # Var(Q_1)

        # CF = Cov[mean_0, mean_1] on shared samples
        # Covariance of means is Cov/n
        CF = C01 / n1

        # cf = Var[mean_1] on LF samples
        # LF samples = n1 (shared), so Var[mean_1] = C11/n1
        cf = C11 / n1

        return CF, cf

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator.

        For ACV, we compute covariances between the high-fidelity estimator
        and control variate discrepancies based on the allocation matrix.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix A. A[i,j] = 1 if model i is evaluated
            on partition j. Shape: (nmodels, npartitions)
        npartition_samples : Array
            Number of samples in each partition. Shape: (npartitions,)

        Returns
        -------
        CF : Array
            Covariance between HF estimator and controls.
            Shape: (nqoi, nqoi * (nmodels-1))
        cf : Array
            Covariance of control variate estimators.
            Shape: (nqoi * (nmodels-1), nqoi * (nmodels-1))
        """
        cov = self.cov()
        nqoi = self._nqoi
        bkd = self._bkd

        allocation_mat_np = bkd.to_numpy(allocation_mat)
        npartition_samples_np = bkd.to_numpy(npartition_samples)

        nmodels = allocation_mat_np.shape[0]
        ncontrols = nmodels - 1

        # Initialize output matrices
        CF_np = np.zeros((nqoi, nqoi * ncontrols))
        cf_np = np.zeros((nqoi * ncontrols, nqoi * ncontrols))

        cov_np = bkd.to_numpy(cov)

        # Extract covariance blocks
        def get_cov_block(i: int, j: int) -> np.ndarray:
            """Get Cov(Q_i, Q_j) block."""
            return cov_np[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]

        # For each control (model m > 0), compute covariances
        # Control m uses: mu_m (estimated on partitions where only m is run)
        #                 Q_m (estimated on partitions where 0 and m are run)

        for m in range(1, nmodels):
            # Find partitions for mu_m (m only) and Q_m (0 and m)
            mu_parts = []  # partitions where model m runs but not model 0
            q_parts = []   # partitions where both models 0 and m run

            for p in range(allocation_mat_np.shape[1]):
                if allocation_mat_np[m, p] == 1:
                    if allocation_mat_np[0, p] == 1:
                        q_parts.append(p)
                    else:
                        mu_parts.append(p)

            # Samples for each part
            n_mu = sum(npartition_samples_np[p] for p in mu_parts)
            n_q = sum(npartition_samples_np[p] for p in q_parts)

            if n_q == 0:
                continue

            # CF[m-1] = Cov[mean_0, Delta_m] where Delta_m = mu_m - Q_m
            # = Cov[mean_0, mu_m] - Cov[mean_0, Q_m]
            # Since mu_m and mean_0 are on disjoint samples: Cov[mean_0, mu_m] = 0
            # Cov[mean_0, Q_m] = Cov(Q_0, Q_m) / n_q (on shared samples)
            C0m = get_cov_block(0, m)
            CF_np[:, (m-1)*nqoi:m*nqoi] = -C0m / n_q

        # cf: Covariance between control variates
        for m in range(1, nmodels):
            for k in range(1, nmodels):
                # Find overlap in partitions
                Cmm = get_cov_block(m, m)
                Cmk = get_cov_block(m, k)
                Ckk = get_cov_block(k, k)

                # Compute sample sizes for each term
                # This is a simplification; full ACV has more complex structure
                n_shared = 0
                for p in range(allocation_mat_np.shape[1]):
                    if (allocation_mat_np[m, p] == 1 and
                        allocation_mat_np[k, p] == 1):
                        n_shared += npartition_samples_np[p]

                if n_shared > 0:
                    if m == k:
                        # Var[Delta_m] = Var[mu_m] + Var[Q_m]
                        n_mu_m = sum(
                            npartition_samples_np[p]
                            for p in range(allocation_mat_np.shape[1])
                            if (allocation_mat_np[m, p] == 1 and
                                allocation_mat_np[0, p] == 0)
                        )
                        n_q_m = sum(
                            npartition_samples_np[p]
                            for p in range(allocation_mat_np.shape[1])
                            if (allocation_mat_np[m, p] == 1 and
                                allocation_mat_np[0, p] == 1)
                        )
                        var_mu = Cmm / max(n_mu_m, 1) if n_mu_m > 0 else 0
                        var_q = Cmm / max(n_q_m, 1) if n_q_m > 0 else 0
                        cf_np[(m-1)*nqoi:m*nqoi, (k-1)*nqoi:k*nqoi] = var_mu + var_q
                    else:
                        cf_np[(m-1)*nqoi:m*nqoi, (k-1)*nqoi:k*nqoi] = Cmk / n_shared

        return bkd.asarray(CF_np), bkd.asarray(cf_np)

    def get_npartition_samples(
        self, allocation_mat: Array, nsamples_per_model: Array
    ) -> Array:
        """Compute samples per partition from samples per model.

        Given the allocation matrix A and desired samples per model,
        compute the samples in each partition.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix. Shape: (nmodels, npartitions)
        nsamples_per_model : Array
            Number of samples for each model. Shape: (nmodels,)

        Returns
        -------
        Array
            Number of samples in each partition. Shape: (npartitions,)
        """
        # This requires solving a linear system A @ npartition = nsamples
        # For ACV structures, this is typically determined by the
        # allocation algorithm
        bkd = self._bkd
        A = bkd.to_numpy(allocation_mat)
        n = bkd.to_numpy(nsamples_per_model)

        # Use least squares if system is overdetermined
        npartition, _, _, _ = np.linalg.lstsq(A, n, rcond=None)

        # Round to integers and ensure non-negative
        npartition = np.maximum(np.round(npartition), 0).astype(np.int64)

        return bkd.asarray(npartition)

    def __repr__(self) -> str:
        has_cov = self._cov is not None
        return f"MultiOutputMean(nqoi={self._nqoi}, has_pilot_cov={has_cov})"
