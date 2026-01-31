"""Multi-output variance statistic for multifidelity estimation.

Computes sample variances and their covariance structure for control variate
estimators.
"""

from typing import Generic, List, Tuple, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.statistics.base import AbstractStatistic


class MultiOutputVariance(AbstractStatistic[Array], Generic[Array]):
    """Multi-output variance statistic.

    Computes the sample variance for multiple quantities of interest and provides
    the covariance structure needed for multifidelity estimation.

    The statistic is:
        sigma^2 = Var[Q] = E[(Q - E[Q])^2]

    For control variate estimation, we need the covariance of variance estimators
    across all models, which depends on fourth-order moments.

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
    >>> stat = MultiOutputVariance(nqoi=1, bkd=bkd)
    >>> values = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> stat.sample_estimate(values)  # Returns [2.5] (sample variance)
    """

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._means: Optional[Array] = None
        self._variances: Optional[Array] = None
        # Fourth-order central moments needed for variance of variance
        self._fourth_moments: Optional[Array] = None
        # W matrix (fourth-order cross-moments) for variance statistics
        self._W: Optional[Array] = None

    def nstats(self) -> int:
        """Return total number of scalar statistics.

        For variance, nstats = nqoi.
        """
        return self._nqoi

    def nmodels(self) -> int:
        """Return number of models from pilot covariance.

        Returns
        -------
        int
            Number of models inferred from covariance shape.

        Raises
        ------
        ValueError
            If pilot quantities have not been set.
        """
        if self._cov is None:
            raise ValueError("Pilot quantities not set. Call set_pilot_quantities first.")
        return self._cov.shape[0] // self._nqoi

    def min_nsamples(self) -> int:
        """Return minimum number of samples needed.

        For variance, minimum is 2 (need at least 2 for unbiased estimator).
        """
        return 2

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample variance.

        Uses Bessel's correction (n-1) for unbiased estimation.

        Parameters
        ----------
        values : Array
            Model output samples. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Sample variance. Shape: (nqoi,)
        """
        self._validate_values(values, min_samples=2)
        nsamples = values.shape[0]
        mean = self._bkd.sum(values, axis=0) / nsamples
        centered = values - mean
        variance = self._bkd.sum(centered ** 2, axis=0) / (nsamples - 1)
        return variance

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array]:
        """Compute pilot quantities for variance estimation.

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        cov : Array
            Covariance of QoI across models. Shape: (nmodels*nqoi, nmodels*nqoi)
        W : Array
            Fourth-order cross-moment matrix for variance estimation.
        """
        nmodels = len(pilot_values)
        bkd = self._bkd

        # Validate all pilot values have same shape
        npilot = pilot_values[0].shape[0]
        for m, vals in enumerate(pilot_values):
            if vals.shape[0] != npilot:
                raise ValueError(
                    f"All pilot samples must have same size. "
                    f"Model 0 has {npilot}, model {m} has {vals.shape[0]}"
                )

        # Stack all pilot values
        pilot_all = bkd.hstack(pilot_values)
        cov = bkd.cov(pilot_all, rowvar=False, ddof=1)

        # Compute W matrix (fourth-order cross-moments)
        W = self._compute_W_from_pilot(pilot_all, nmodels)

        return cov, W

    def _compute_W_entry(
        self, pilot_values_ii: Array, pilot_values_jj: Array
    ) -> Array:
        """Compute a single block W[ii][jj] of the W matrix.

        This computes the fourth-order cross-moment between models ii and jj.
        For single model (ii==jj) with 1 qoi, this is the kurtosis.

        Parameters
        ----------
        pilot_values_ii : Array
            Pilot values for model ii. Shape: (npilot, nqoi)
        pilot_values_jj : Array
            Pilot values for model jj. Shape: (npilot, nqoi)

        Returns
        -------
        W_entry : Array
            Fourth-order cross-moment block. Shape: (nqoi^2, nqoi^2)
        """
        bkd = self._bkd
        nqoi = pilot_values_ii.shape[1]
        npilot = pilot_values_ii.shape[0]

        # Center the values
        means_ii = bkd.sum(pilot_values_ii, axis=0) / npilot
        means_jj = bkd.sum(pilot_values_jj, axis=0) / npilot
        centered_ii = pilot_values_ii - means_ii
        centered_jj = pilot_values_jj - means_jj

        # Compute outer products for each sample: shape (npilot, nqoi, nqoi)
        # Then flatten to (npilot, nqoi^2)
        centered_sq_ii = bkd.einsum(
            "nk,nl->nkl", centered_ii, centered_ii
        )
        centered_sq_ii = bkd.reshape(centered_sq_ii, (npilot, -1))

        centered_sq_jj = bkd.einsum(
            "nk,nl->nkl", centered_jj, centered_jj
        )
        centered_sq_jj = bkd.reshape(centered_sq_jj, (npilot, -1))

        # Compute means of the squared centered values
        centered_sq_ii_mean = bkd.sum(centered_sq_ii, axis=0) / npilot
        centered_sq_jj_mean = bkd.sum(centered_sq_jj, axis=0) / npilot

        # Compute the cross-covariance of the squared centered values
        centered_sq_ii_centered = centered_sq_ii - centered_sq_ii_mean
        centered_sq_jj_centered = centered_sq_jj - centered_sq_jj_mean

        # Outer product across samples: (npilot, nqoi^2, nqoi^2)
        cross = bkd.einsum(
            "nk,nl->nkl",
            centered_sq_ii_centered,
            centered_sq_jj_centered
        )
        cross = bkd.reshape(cross, (npilot, -1))

        # Sum and normalize
        W_entry = bkd.sum(cross, axis=0) / npilot
        W_entry = bkd.reshape(W_entry, (nqoi * nqoi, nqoi * nqoi))

        return W_entry

    def _compute_W_from_pilot(self, pilot_values: Array, nmodels: int) -> Array:
        """Compute W matrix from pilot samples.

        This matches the legacy _get_W_from_pilot function.
        W is a block matrix where W[ii][jj] contains fourth-order cross-moments
        between models ii and jj.

        Parameters
        ----------
        pilot_values : Array
            Stacked pilot values. Shape: (npilot, nmodels*nqoi)
        nmodels : int
            Number of models.

        Returns
        -------
        W : Array
            Fourth-order cross-moment matrix. Shape: (nmodels*nqoi^2, nmodels*nqoi^2)
        """
        bkd = self._bkd
        nqoi = self._nqoi

        # Build W as a block matrix
        W_blocks = [[None for _ in range(nmodels)] for _ in range(nmodels)]

        for ii in range(nmodels):
            pilot_ii = pilot_values[:, ii * nqoi:(ii + 1) * nqoi]
            W_blocks[ii][ii] = self._compute_W_entry(pilot_ii, pilot_ii)

            for jj in range(ii + 1, nmodels):
                pilot_jj = pilot_values[:, jj * nqoi:(jj + 1) * nqoi]
                W_blocks[ii][jj] = self._compute_W_entry(pilot_ii, pilot_jj)
                W_blocks[jj][ii] = W_blocks[ii][jj].T

        # Assemble the block matrix
        rows = []
        for ii in range(nmodels):
            row = bkd.hstack([W_blocks[ii][jj] for jj in range(nmodels)])
            rows.append(row)
        W = bkd.vstack(rows)

        return W

    def set_pilot_quantities(
        self,
        cov: Array,
        W: Optional[Array] = None,
    ) -> None:
        """Set pilot quantities directly.

        Parameters
        ----------
        cov : Array
            Covariance of QoI across models. Shape: (nmodels*nqoi, nmodels*nqoi)
        W : Array, optional
            Fourth-order cross-moment matrix for variance estimation.
        """
        self._cov = cov
        self._W = W

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity sample variance.

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
        nqoi = self._nqoi

        # Extract HF block (first nqoi x nqoi)
        hf_cov = cov[:nqoi, :nqoi]

        # Scale by sample size (covariance of variance estimators scales as 1/n)
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
        nsamples = values.shape[0]
        mean = self._bkd.sum(values, axis=0) / nsamples
        centered = values - mean
        cov = centered.T @ centered / (nsamples - 1)
        return cov

    # =========================================================================
    # Helper methods for allocation matrix computations
    # These match the pattern in MultiOutputMean
    # =========================================================================

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
        """Compute Gmat and gvec multipliers for mean discrepancy covariances.

        This matches the legacy _get_acv_mean_discrepancy_covariances_multipliers.
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

    def _get_acv_variance_discrepancy_covariances_multipliers(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Compute Hmat and hvec multipliers for variance discrepancy covariances.

        This matches the legacy _get_acv_variance_discrepancy_covariances_multipliers
        from Equation 3.14 of Dixon et al.
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

        Hmat_rows = []
        hvec_list = []

        N0 = nsamples_subset[1]  # N_0 (unstarred set for model 0)

        for ii in range(1, nmodels):
            Nis_0 = nsamples_intersect[2 * ii, 1]      # N_{0 ∩ i*}
            Ni_0 = nsamples_intersect[2 * ii + 1, 1]   # N_{0 ∩ i}
            Nis = nsamples_subset[2 * ii]             # N_{i*}
            Ni = nsamples_subset[2 * ii + 1]          # N_{i}

            # hvec[ii-1] computation
            # Avoid division by zero with safe denominators
            denom1 = N0 * (N0 - 1) * Nis * (Nis - 1)
            denom2 = N0 * (N0 - 1) * Ni * (Ni - 1)

            safe_denom1 = bkd.where(denom1 > 0, denom1, bkd.ones_like(denom1))
            safe_denom2 = bkd.where(denom2 > 0, denom2, bkd.ones_like(denom2))

            term1 = bkd.where(
                denom1 > 0,
                Nis_0 * (Nis_0 - 1) / safe_denom1,
                bkd.zeros_like(denom1)
            )
            term2 = bkd.where(
                denom2 > 0,
                Ni_0 * (Ni_0 - 1) / safe_denom2,
                bkd.zeros_like(denom2)
            )
            hvec_list.append(bkd.reshape(term1 - term2, (1,)))

            # Build Hmat row
            Hmat_row = []
            for jj in range(1, nmodels):
                Nis_js = nsamples_intersect[2 * ii, 2 * jj]        # N_{i* ∩ j*}
                Ni_j = nsamples_intersect[2 * ii + 1, 2 * jj + 1]  # N_{i ∩ j}
                Ni_js = nsamples_intersect[2 * ii + 1, 2 * jj]     # N_{i ∩ j*}
                Nis_j = nsamples_intersect[2 * ii, 2 * jj + 1]     # N_{i* ∩ j}
                Njs = nsamples_subset[2 * jj]                      # N_{j*}
                Nj = nsamples_subset[2 * jj + 1]                   # N_{j}

                # Four terms for Hmat[ii-1, jj-1]
                d00 = Nis * (Nis - 1) * Njs * (Njs - 1)
                d01 = Nis * (Nis - 1) * Nj * (Nj - 1)
                d10 = Ni * (Ni - 1) * Njs * (Njs - 1)
                d11 = Ni * (Ni - 1) * Nj * (Nj - 1)

                safe_d00 = bkd.where(d00 > 0, d00, bkd.ones_like(d00))
                safe_d01 = bkd.where(d01 > 0, d01, bkd.ones_like(d01))
                safe_d10 = bkd.where(d10 > 0, d10, bkd.ones_like(d10))
                safe_d11 = bkd.where(d11 > 0, d11, bkd.ones_like(d11))

                t00 = bkd.where(d00 > 0, Nis_js * (Nis_js - 1) / safe_d00, bkd.zeros_like(d00))
                t01 = bkd.where(d01 > 0, Nis_j * (Nis_j - 1) / safe_d01, bkd.zeros_like(d01))
                t10 = bkd.where(d10 > 0, Ni_js * (Ni_js - 1) / safe_d10, bkd.zeros_like(d10))
                t11 = bkd.where(d11 > 0, Ni_j * (Ni_j - 1) / safe_d11, bkd.zeros_like(d11))

                Hmat_row.append(bkd.reshape(t00 - t01 - t10 + t11, (1,)))
            Hmat_rows.append(bkd.concatenate(Hmat_row))

        hvec = bkd.concatenate(hvec_list)
        Hmat = bkd.stack(Hmat_rows, axis=0)

        return Hmat, hvec

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for CV estimator of variance with known LF stats.

        For CV with shared samples, all multipliers are 1/n (Gmat, gvec)
        and 1/(n*(n-1)) (Hmat, hvec).

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in the single shared partition. Shape: (1,)

        Returns
        -------
        CF : Array
            Covariance of discrepancies. Shape: (nqoi*(nmodels-1), nqoi*(nmodels-1))
        cf : Array
            Covariance between HF estimator and discrepancies.
            Shape: (nqoi, nqoi*(nmodels-1))
        """
        cov = self.cov()
        bkd = self._bkd
        nqoi = self._nqoi
        nmodels = cov.shape[0] // nqoi
        ncontrols = nmodels - 1

        n = npartition_samples[0]

        # For CV with shared samples
        Gmat = bkd.full((ncontrols, ncontrols), 1.0 / n)
        gvec = bkd.full((ncontrols,), 1.0 / n)
        Hmat = bkd.full((ncontrols, ncontrols), 1.0 / (n * (n - 1)))
        hvec = bkd.full((ncontrols,), 1.0 / (n * (n - 1)))

        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        """Compute discrepancy covariances from multipliers.

        Uses the formula for covariance of variance estimators which involves
        both the standard covariance (scaled by Gmat/gvec) and the fourth-order
        moments (scaled by Hmat/hvec).

        For now, this is a simplified implementation that uses only the
        covariance structure (similar to mean). A full implementation would
        incorporate the W matrix (fourth-order moments).
        """
        cov = self.cov()
        bkd = self._bkd
        nqoi = self._nqoi
        nmodels = cov.shape[0] // nqoi
        ncontrols = nmodels - 1

        # Build CF (delta-delta covariance) and cf (HF-delta covariance)
        CF = bkd.zeros((nqoi * ncontrols, nqoi * ncontrols))
        cf = bkd.zeros((nqoi, nqoi * ncontrols))

        for j in range(1, nmodels):
            # cf block for control j (HF-delta covariance)
            C0j = cov[0*nqoi:(0+1)*nqoi, j*nqoi:(j+1)*nqoi]
            cf[:, (j-1)*nqoi:j*nqoi] = C0j * gvec[j-1]

        for i in range(1, nmodels):
            for j in range(1, nmodels):
                # CF block for delta-delta covariance
                Cij = cov[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]
                CF[(i-1)*nqoi:i*nqoi, (j-1)*nqoi:j*nqoi] = Cij * Gmat[i-1, j-1]

        return CF, cf

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator of variance.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix A. Shape: (nmodels, 2*nmodels)
            - Rows (k): independent partitions k = 0, ..., M-1
            - Columns (j): sample sets Z₀*, Z₀, Z₁*, Z₁, ...
        npartition_samples : Array
            Number of samples in each partition. Shape: (nmodels,)

        Returns
        -------
        CF : Array
            Covariance of control variate estimators (delta-delta covariance).
            Shape: (nqoi * (nmodels-1), nqoi * (nmodels-1))
        cf : Array
            Covariance between HF estimator and controls (HF-delta covariance).
            Shape: (nqoi, nqoi * (nmodels-1))
        """
        # Compute both mean and variance multipliers
        Gmat, gvec = self._get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples
        )
        Hmat, hvec = self._get_acv_variance_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples
        )

        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_npartition_samples(
        self, allocation_mat: Array, nsamples_per_model: Array
    ) -> Array:
        """Compute samples per partition from samples per model."""
        bkd = self._bkd
        A = bkd.to_numpy(allocation_mat)
        n = bkd.to_numpy(nsamples_per_model)
        npartition, _, _, _ = np.linalg.lstsq(A, n, rcond=None)
        npartition = np.maximum(np.round(npartition), 0).astype(np.int64)
        return bkd.asarray(npartition)

    def __repr__(self) -> str:
        has_cov = self._cov is not None
        return f"MultiOutputVariance(nqoi={self._nqoi}, has_pilot_cov={has_cov})"
