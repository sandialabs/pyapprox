"""Multi-output mean and variance joint statistic for multifidelity estimation.

Computes both sample mean and variance jointly and their covariance structure
for control variate estimators.
"""

from typing import Generic, List, Tuple, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.statistics.base import AbstractStatistic
from pyapprox.typing.stats.statistics.covariance_utils import (
    compute_W_from_pilot,
    compute_B_from_pilot,
    compute_covariance_from_pilot,
)


class MultiOutputMeanAndVariance(AbstractStatistic[Array], Generic[Array]):
    """Multi-output mean and variance joint statistic.

    Computes both the sample mean and variance for multiple quantities of
    interest and provides the covariance structure needed for multifidelity
    estimation.

    The statistics are:
        mu = E[Q]
        sigma^2 = Var[Q]

    The total number of statistics is 2 * nqoi (nqoi means + nqoi variances).
    Statistics are ordered as [mean_0, ..., mean_{nqoi-1}, var_0, ..., var_{nqoi-1}].

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
    >>> stat = MultiOutputMeanAndVariance(nqoi=1, bkd=bkd)
    >>> values = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> result = stat.sample_estimate(values)
    >>> # Returns [3.0, 2.5] (mean=3, variance=2.5)
    """

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._means: Optional[Array] = None
        self._variances: Optional[Array] = None

    def nstats(self) -> int:
        """Return total number of scalar statistics.

        For mean+variance, nstats = 2*nqoi.
        """
        return 2 * self._nqoi

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
        # For mean+variance, cov has shape (2*nmodels*nqoi, 2*nmodels*nqoi)
        return self._cov.shape[0] // (2 * self._nqoi)

    def min_nsamples(self) -> int:
        """Return minimum number of samples needed.

        For variance, minimum is 2.
        """
        return 2

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample mean and variance.

        Parameters
        ----------
        values : Array
            Model output samples. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Sample mean and variance. Shape: (2*nqoi,)
            Ordered as [mean_0, ..., mean_{nqoi-1}, var_0, ..., var_{nqoi-1}]
        """
        self._validate_values(values, min_samples=2)
        nsamples = values.shape[0]
        bkd = self._bkd

        mean = bkd.sum(values, axis=0) / nsamples
        centered = values - mean
        variance = bkd.sum(centered ** 2, axis=0) / (nsamples - 1)

        # Concatenate mean and variance
        return bkd.concatenate([mean, variance], axis=0)

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array, Array]:
        """Compute pilot quantities for mean and variance estimation.

        The covariance structure includes:
        - Covariance between means across models
        - Covariance between variances across models
        - Covariance between means and variances

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        cov : Array
            Full covariance of mean+variance estimators.
            Shape: (2*nmodels*nqoi, 2*nmodels*nqoi)
        means : Array
            Sample means. Shape: (nmodels, nqoi)
        variances : Array
            Sample variances. Shape: (nmodels, nqoi)
        """
        nmodels = len(pilot_values)
        nqoi = self._nqoi
        bkd = self._bkd

        # Validate
        npilot = pilot_values[0].shape[0]
        for m, vals in enumerate(pilot_values):
            if vals.shape[0] != npilot:
                raise ValueError(
                    f"All pilot samples must have same size. "
                    f"Model 0 has {npilot}, model {m} has {vals.shape[0]}"
                )
            if vals.shape[1] != nqoi:
                raise ValueError(
                    f"Model {m} has {vals.shape[1]} QoI, expected {nqoi}"
                )

        # Compute means and variances
        means_list = []
        vars_list = []

        for vals in pilot_values:
            n = vals.shape[0]
            mean = bkd.sum(vals, axis=0) / n
            centered = vals - mean
            var = bkd.sum(centered ** 2, axis=0) / (n - 1)
            means_list.append(mean)
            vars_list.append(var)

        means = bkd.stack(means_list, axis=0)
        variances = bkd.stack(vars_list, axis=0)

        # Build the joint statistics array for covariance computation
        # For each sample, compute [Q, (Q-mu)^2] for all models
        # Shape: (npilot, 2 * nmodels * nqoi)

        joint_stats_list = []
        for m, vals in enumerate(pilot_values):
            # Centered values
            mean_m = means_list[m]
            centered = vals - mean_m

            # Linear terms (for mean)
            joint_stats_list.append(vals)
            # Quadratic terms (for variance)
            joint_stats_list.append(centered ** 2)

        # Reorder: [means_model0, ..., means_modelM, vars_model0, ..., vars_modelM]
        # First all means, then all variances
        means_parts = joint_stats_list[::2]  # Even indices: means
        vars_parts = joint_stats_list[1::2]  # Odd indices: variances

        joint_stats = bkd.concatenate(means_parts + vars_parts, axis=1)

        # Compute covariance
        cov = self._compute_covariance(joint_stats)

        return cov, means, variances

    def compute_pilot_quantities_with_covariance_matrices(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array, Array]:
        """Compute pilot quantities including W and B matrices.

        This method returns quantities compatible with the legacy interface
        for variance estimation requiring fourth-order moment structures.

        The matrices returned are:
        - cov: Cross-model covariance of outputs (nmodels*nqoi, nmodels*nqoi)
        - W: Cov[(f-E[f])^{⊗2}, (g-E[g])^{⊗2}] (nmodels*nqoi^2, nmodels*nqoi^2)
        - B: Cov[f, (g-E[g])^{⊗2}] (nmodels*nqoi, nmodels*nqoi^2)

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        cov : Array
            Cross-model covariance. Shape: (nmodels*nqoi, nmodels*nqoi)
        W : Array
            Covariance of centered Kronecker products.
            Shape: (nmodels*nqoi^2, nmodels*nqoi^2)
        B : Array
            Cross-covariance of means and variance estimators.
            Shape: (nmodels*nqoi, nmodels*nqoi^2)
        """
        nqoi = self._nqoi
        bkd = self._bkd

        # Validate
        npilot = pilot_values[0].shape[0]
        for m, vals in enumerate(pilot_values):
            if vals.shape[0] != npilot:
                raise ValueError(
                    f"All pilot samples must have same size. "
                    f"Model 0 has {npilot}, model {m} has {vals.shape[0]}"
                )
            if vals.shape[1] != nqoi:
                raise ValueError(
                    f"Model {m} has {vals.shape[1]} QoI, expected {nqoi}"
                )

        # Compute using covariance_utils functions
        cov = compute_covariance_from_pilot(pilot_values, bkd)
        W = compute_W_from_pilot(pilot_values, bkd)
        B = compute_B_from_pilot(pilot_values, bkd)

        return cov, W, B

    def set_pilot_quantities(
        self,
        cov: Array,
        means: Optional[Array] = None,
        variances: Optional[Array] = None,
    ) -> None:
        """Set pilot quantities directly.

        Parameters
        ----------
        cov : Array
            Full covariance. Shape: (2*nmodels*nqoi, 2*nmodels*nqoi)
        means : Array, optional
            Sample means. Shape: (nmodels, nqoi)
        variances : Array, optional
            Sample variances. Shape: (nmodels, nqoi)
        """
        self._cov = cov
        self._means = means
        self._variances = variances

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity mean+variance estimator.

        Parameters
        ----------
        nhf_samples : int
            Number of high-fidelity samples.

        Returns
        -------
        Array
            Covariance matrix. Shape: (2*nqoi, 2*nqoi)
        """
        cov = self.cov()
        nqoi = self._nqoi

        # Extract HF block: first nqoi means + first nqoi variances
        # Covariance structure: [means_all_models, vars_all_models]
        # HF is model 0, so indices are [0:nqoi] for HF mean
        # and [nmodels*nqoi : nmodels*nqoi + nqoi] for HF var

        # For simplicity, assume cov is structured as:
        # [HF_mean, LF1_mean, ..., HF_var, LF1_var, ...]
        # We need the 2x2 block for [HF_mean, HF_var]

        hf_mean_idx = slice(0, nqoi)

        # Need to find where variances start - this depends on nmodels
        # From cov shape, infer nmodels
        total_stats = cov.shape[0]
        # total_stats = 2 * nmodels * nqoi
        nmodels = total_stats // (2 * nqoi)
        hf_var_idx = slice(nmodels * nqoi, nmodels * nqoi + nqoi)

        # Build HF covariance block
        hf_cov = self._bkd.zeros((2 * nqoi, 2 * nqoi))

        cov_np = self._bkd.to_numpy(cov)
        hf_cov_np = np.zeros((2 * nqoi, 2 * nqoi))

        # Mean-mean block
        hf_cov_np[:nqoi, :nqoi] = cov_np[hf_mean_idx, hf_mean_idx]
        # Mean-var block
        hf_cov_np[:nqoi, nqoi:] = cov_np[hf_mean_idx, hf_var_idx]
        # Var-mean block
        hf_cov_np[nqoi:, :nqoi] = cov_np[hf_var_idx, hf_mean_idx]
        # Var-var block
        hf_cov_np[nqoi:, nqoi:] = cov_np[hf_var_idx, hf_var_idx]

        return self._bkd.asarray(hf_cov_np / nhf_samples)

    def _compute_covariance(self, values: Array) -> Array:
        """Compute sample covariance matrix."""
        nsamples = values.shape[0]
        mean = self._bkd.sum(values, axis=0) / nsamples
        centered = values - mean
        cov = centered.T @ centered / (nsamples - 1)
        return cov

    # =========================================================================
    # Helper methods for allocation matrix computations
    # These match the pattern in MultiOutputMean and MultiOutputVariance
    # =========================================================================

    def _get_nsamples_intersect(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Array:
        """Compute sample intersection counts between sets.

        Returns
        -------
        nsamples_intersect : Array (2*nmodels, 2*nmodels)
            The i,j entry contains the sample count in the intersection of sets i and j.
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
        """Compute Gmat and gvec multipliers for mean discrepancy covariances."""
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

        gvec_list = []
        Gmat_rows = []

        for ii in range(1, nmodels):
            denom1 = nsamples_subset[2 * ii] * nsamples_subset[1]
            denom2 = nsamples_subset[2 * ii + 1] * nsamples_subset[1]

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
        """Compute Hmat and hvec multipliers for variance discrepancy covariances."""
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
            Nis_0 = nsamples_intersect[2 * ii, 1]
            Ni_0 = nsamples_intersect[2 * ii + 1, 1]
            Nis = nsamples_subset[2 * ii]
            Ni = nsamples_subset[2 * ii + 1]

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

            Hmat_row = []
            for jj in range(1, nmodels):
                Nis_js = nsamples_intersect[2 * ii, 2 * jj]
                Ni_j = nsamples_intersect[2 * ii + 1, 2 * jj + 1]
                Ni_js = nsamples_intersect[2 * ii + 1, 2 * jj]
                Nis_j = nsamples_intersect[2 * ii, 2 * jj + 1]
                Njs = nsamples_subset[2 * jj]
                Nj = nsamples_subset[2 * jj + 1]

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

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        """Compute discrepancy covariances from multipliers.

        For mean+variance, this combines mean and variance structures.
        """
        cov = self.cov()
        bkd = self._bkd
        nqoi = self._nqoi

        total_stats = cov.shape[0]
        nmodels = total_stats // (2 * nqoi)
        ncontrols = nmodels - 1
        nstats_per_model = 2 * nqoi

        cov_np = bkd.to_numpy(cov)

        def get_model_indices(m: int) -> Tuple[slice, slice]:
            """Get mean and var indices for model m."""
            mean_idx = slice(m * nqoi, (m + 1) * nqoi)
            var_idx = slice(nmodels * nqoi + m * nqoi, nmodels * nqoi + (m + 1) * nqoi)
            return mean_idx, var_idx

        def get_cov_block(m: int, k: int) -> np.ndarray:
            """Get full 2*nqoi x 2*nqoi covariance block between models m and k."""
            m_mean, m_var = get_model_indices(m)
            k_mean, k_var = get_model_indices(k)

            block = np.zeros((2 * nqoi, 2 * nqoi))
            block[:nqoi, :nqoi] = cov_np[m_mean, k_mean]
            block[:nqoi, nqoi:] = cov_np[m_mean, k_var]
            block[nqoi:, :nqoi] = cov_np[m_var, k_mean]
            block[nqoi:, nqoi:] = cov_np[m_var, k_var]
            return block

        # Build CF (delta-delta covariance) and cf (HF-delta covariance)
        CF_np = np.zeros((nstats_per_model * ncontrols, nstats_per_model * ncontrols))
        cf_np = np.zeros((nstats_per_model, nstats_per_model * ncontrols))

        Gmat_np = bkd.to_numpy(Gmat)
        gvec_np = bkd.to_numpy(gvec)

        for j in range(1, nmodels):
            # cf block for control j (HF-delta covariance)
            C0j = get_cov_block(0, j)
            start = (j - 1) * nstats_per_model
            end = j * nstats_per_model
            cf_np[:, start:end] = C0j * gvec_np[j - 1]

        for i in range(1, nmodels):
            for j in range(1, nmodels):
                # CF block for delta-delta covariance
                Cij = get_cov_block(i, j)
                i_start = (i - 1) * nstats_per_model
                i_end = i * nstats_per_model
                j_start = (j - 1) * nstats_per_model
                j_end = j * nstats_per_model
                CF_np[i_start:i_end, j_start:j_end] = Cij * Gmat_np[i - 1, j - 1]

        return bkd.asarray(CF_np), bkd.asarray(cf_np)

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for CV estimator of mean+variance.

        For a multi-model CV estimator with M models, all using shared samples.

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in the single shared partition. Shape: (1,)

        Returns
        -------
        CF : Array
            Covariance of discrepancies.
            Shape: (2*nqoi*(nmodels-1), 2*nqoi*(nmodels-1))
        cf : Array
            Covariance between HF estimator and discrepancies.
            Shape: (2*nqoi, 2*nqoi*(nmodels-1))
        """
        cov = self.cov()
        bkd = self._bkd
        nqoi = self._nqoi

        n = npartition_samples[0]

        total_stats = cov.shape[0]
        nmodels = total_stats // (2 * nqoi)
        ncontrols = nmodels - 1

        # For CV with shared samples
        Gmat = bkd.full((ncontrols, ncontrols), 1.0 / n)
        gvec = bkd.full((ncontrols,), 1.0 / n)
        Hmat = bkd.full((ncontrols, ncontrols), 1.0 / (n * (n - 1)))
        hvec = bkd.full((ncontrols,), 1.0 / (n * (n - 1)))

        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator of mean+variance.

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
            Shape: (2*nqoi * (nmodels-1), 2*nqoi * (nmodels-1))
        cf : Array
            Covariance between HF estimator and controls (HF-delta covariance).
            Shape: (2*nqoi, 2*nqoi * (nmodels-1))
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
        return f"MultiOutputMeanAndVariance(nqoi={self._nqoi}, has_pilot_cov={has_cov})"
