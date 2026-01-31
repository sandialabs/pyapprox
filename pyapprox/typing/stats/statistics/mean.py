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

        This implementation matches the legacy pyapprox.multifidelity.stats
        computation using Gmat and gvec multipliers derived from sample
        intersection counts.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix A. Shape: (nmodels, 2*nmodels)
            - Rows (k): independent partitions k = 0, ..., M-1
            - Columns (j): sample sets Z₀*, Z₀, Z₁*, Z₁, ...
              - Column 2m: Z_m* (starred set for model m)
              - Column 2m+1: Z_m (unstarred set for model m)
        npartition_samples : Array
            Number of samples in each partition. Shape: (nmodels,)

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

        nmodels = allocation_mat.shape[0]
        ncontrols = nmodels - 1

        # Compute Gmat and gvec multipliers using legacy formula
        Gmat, gvec = self._get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples
        )

        # Compute CF and cf using the multipliers
        # For nqoi=1: CF[j-1] = Cov(Q_0, Q_j) * gvec[j-1]
        #             cf[i-1, j-1] = Cov(Q_i, Q_j) * Gmat[i-1, j-1]
        CF = bkd.zeros((nqoi, nqoi * ncontrols))
        cf = bkd.zeros((nqoi * ncontrols, nqoi * ncontrols))

        for j in range(1, nmodels):
            # CF block for control j
            C0j = cov[0*nqoi:(0+1)*nqoi, j*nqoi:(j+1)*nqoi]
            CF[:, (j-1)*nqoi:j*nqoi] = C0j * gvec[j-1]

        for i in range(1, nmodels):
            for j in range(1, nmodels):
                Cij = cov[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]
                cf[(i-1)*nqoi:i*nqoi, (j-1)*nqoi:j*nqoi] = Cij * Gmat[i-1, j-1]

        return CF, cf

    def _get_nsamples_intersect(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Array:
        """Compute sample intersection counts between sets.

        Returns
        -------
        nsamples_intersect : Array (2*nmodels, 2*nmodels)
            The i,j entry contains:
            - |Z_i* ∩ Z_j*| when i%2==0 and j%2==0
            - |Z_i ∩ Z_j*| when i%2==1 and j%2==0
            - |Z_i* ∩ Z_j| when i%2==0 and j%2==1
            - |Z_i ∩ Z_j| when i%2==1 and j%2==1
        """
        bkd = self._bkd
        nmodels = allocation_mat.shape[0]

        # nsubset_samples[k, j] = p[k] * A[k, j]
        nsubset_samples = bkd.reshape(npartition_samples, (-1, 1)) * allocation_mat

        # Build intersection matrix row by row
        rows = []
        for ii in range(2 * nmodels):
            # Find partitions where column ii is active
            active_mask = allocation_mat[:, ii] == 1
            # Sum contributions from those partitions for all columns
            row_vals = []
            for jj in range(2 * nmodels):
                val = bkd.sum(bkd.where(active_mask, nsubset_samples[:, jj],
                                        bkd.zeros_like(nsubset_samples[:, jj])))
                row_vals.append(bkd.reshape(val, (1,)))
            rows.append(bkd.concatenate(row_vals))

        return bkd.stack(rows, axis=0)

    def _get_nsamples_subset(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Array:
        """Get the number of samples in each sample subset.

        Returns
        -------
        nsamples_subset : Array (2*nmodels,)
            The number of samples in Z_i* (even indices) and Z_i (odd indices)
        """
        bkd = self._bkd
        # nsamples_subset[j] = sum_k p[k] * A[k, j]
        return bkd.dot(allocation_mat.T, npartition_samples)

    def _get_acv_mean_discrepancy_covariances_multipliers(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Compute Gmat and gvec multipliers for ACV discrepancy covariances.

        This matches the legacy _get_acv_mean_discrepancy_covariances_multipliers
        function from pyapprox.multifidelity.stats.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        npartition_samples : Array
            Number of samples in each partition. Shape: (nmodels,)

        Returns
        -------
        Gmat : Array
            Multiplier matrix for cf computation. Shape: (nmodels-1, nmodels-1)
        gvec : Array
            Multiplier vector for CF computation. Shape: (nmodels-1,)
        """
        bkd = self._bkd
        nmodels = allocation_mat.shape[0]

        if bkd.any_bool(npartition_samples < 0):
            raise RuntimeError(
                f"An entry in npartition_samples was negative: {npartition_samples}"
            )

        nsamples_intersect = self._get_nsamples_intersect(
            allocation_mat, npartition_samples
        )
        nsamples_subset = self._get_nsamples_subset(
            allocation_mat, npartition_samples
        )

        # Build gvec as a list then concatenate
        gvec_list = []
        # Build Gmat as a list of rows then stack
        Gmat_rows = []

        for ii in range(1, nmodels):
            # gvec[ii-1] computation
            # Term 1: n_intersect[2*ii, 1] / (n_subset[2*ii] * n_subset[1])
            # Term 2: n_intersect[2*ii+1, 1] / (n_subset[2*ii+1] * n_subset[1])
            denom1 = nsamples_subset[2 * ii] * nsamples_subset[1]
            denom2 = nsamples_subset[2 * ii + 1] * nsamples_subset[1]

            # Avoid division by zero
            safe_denom1 = bkd.where(denom1 > 0, denom1, bkd.ones_like(denom1))
            safe_denom2 = bkd.where(denom2 > 0, denom2, bkd.ones_like(denom2))

            term1 = bkd.where(
                denom1 > 0,
                nsamples_intersect[2 * ii, 1] / safe_denom1,
                bkd.zeros_like(denom1)
            )
            term2 = bkd.where(
                denom2 > 0,
                nsamples_intersect[2 * ii + 1, 1] / safe_denom2,
                bkd.zeros_like(denom2)
            )
            gvec_list.append(bkd.reshape(term1 - term2, (1,)))

            # Build Gmat row
            Gmat_row = []
            for jj in range(1, nmodels):
                # Gmat[ii-1, jj-1] computation (4 terms)
                d00 = nsamples_subset[2 * ii] * nsamples_subset[2 * jj]
                d01 = nsamples_subset[2 * ii] * nsamples_subset[2 * jj + 1]
                d10 = nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj]
                d11 = nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj + 1]

                safe_d00 = bkd.where(d00 > 0, d00, bkd.ones_like(d00))
                safe_d01 = bkd.where(d01 > 0, d01, bkd.ones_like(d01))
                safe_d10 = bkd.where(d10 > 0, d10, bkd.ones_like(d10))
                safe_d11 = bkd.where(d11 > 0, d11, bkd.ones_like(d11))

                t00 = bkd.where(d00 > 0, nsamples_intersect[2*ii, 2*jj] / safe_d00, bkd.zeros_like(d00))
                t01 = bkd.where(d01 > 0, nsamples_intersect[2*ii, 2*jj+1] / safe_d01, bkd.zeros_like(d01))
                t10 = bkd.where(d10 > 0, nsamples_intersect[2*ii+1, 2*jj] / safe_d10, bkd.zeros_like(d10))
                t11 = bkd.where(d11 > 0, nsamples_intersect[2*ii+1, 2*jj+1] / safe_d11, bkd.zeros_like(d11))

                Gmat_row.append(bkd.reshape(t00 - t01 - t10 + t11, (1,)))
            Gmat_rows.append(bkd.concatenate(Gmat_row))

        gvec = bkd.concatenate(gvec_list)
        Gmat = bkd.stack(Gmat_rows, axis=0)

        return Gmat, gvec

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

        # Use least squares if system is overdetermined
        npartition, _, _, _ = bkd.lstsq(allocation_mat, nsamples_per_model)

        # Round to integers and ensure non-negative
        npartition = bkd.maximum(bkd.round(npartition), bkd.asarray(0.0))

        return npartition

    def __repr__(self) -> str:
        has_cov = self._cov is not None
        return f"MultiOutputMean(nqoi={self._nqoi}, has_pilot_cov={has_cov})"
