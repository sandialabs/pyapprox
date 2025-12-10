"""Multi-output mean and variance joint statistic for multifidelity estimation.

Computes both sample mean and variance jointly and their covariance structure
for control variate estimators.
"""

from typing import Generic, List, Tuple, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.statistics.base import AbstractStatistic


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

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for CV estimator of mean+variance.

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in each partition.

        Returns
        -------
        CF : Array
            Shape: (2*nqoi, 2*nqoi)
        cf : Array
            Shape: (2*nqoi, 2*nqoi)
        """
        cov = self.cov()
        nqoi = self._nqoi
        bkd = self._bkd

        n1 = npartition_samples[1]

        # Infer nmodels
        total_stats = cov.shape[0]
        nmodels = total_stats // (2 * nqoi)

        cov_np = bkd.to_numpy(cov)

        # HF indices
        hf_mean = slice(0, nqoi)
        hf_var = slice(nmodels * nqoi, nmodels * nqoi + nqoi)

        # LF indices (model 1)
        lf_mean = slice(nqoi, 2 * nqoi)
        lf_var = slice(nmodels * nqoi + nqoi, nmodels * nqoi + 2 * nqoi)

        # Build CF and cf blocks
        CF_np = np.zeros((2 * nqoi, 2 * nqoi))
        cf_np = np.zeros((2 * nqoi, 2 * nqoi))

        # CF: Cov(HF, LF)
        CF_np[:nqoi, :nqoi] = cov_np[hf_mean, lf_mean]
        CF_np[:nqoi, nqoi:] = cov_np[hf_mean, lf_var]
        CF_np[nqoi:, :nqoi] = cov_np[hf_var, lf_mean]
        CF_np[nqoi:, nqoi:] = cov_np[hf_var, lf_var]

        # cf: Cov(LF, LF)
        cf_np[:nqoi, :nqoi] = cov_np[lf_mean, lf_mean]
        cf_np[:nqoi, nqoi:] = cov_np[lf_mean, lf_var]
        cf_np[nqoi:, :nqoi] = cov_np[lf_var, lf_mean]
        cf_np[nqoi:, nqoi:] = cov_np[lf_var, lf_var]

        return bkd.asarray(CF_np / n1), bkd.asarray(cf_np / n1)

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator of mean+variance."""
        cov = self.cov()
        nqoi = self._nqoi
        bkd = self._bkd

        allocation_mat_np = bkd.to_numpy(allocation_mat)
        npartition_samples_np = bkd.to_numpy(npartition_samples)

        nmodels = allocation_mat_np.shape[0]
        ncontrols = nmodels - 1
        nstats_per_model = 2 * nqoi

        CF_np = np.zeros((nstats_per_model, nstats_per_model * ncontrols))
        cf_np = np.zeros(
            (nstats_per_model * ncontrols, nstats_per_model * ncontrols)
        )

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

        # Build CF: Cov(HF, LF_m) for m = 1, ..., nmodels-1
        for m in range(1, nmodels):
            q_parts = []
            for p in range(allocation_mat_np.shape[1]):
                if allocation_mat_np[m, p] == 1 and allocation_mat_np[0, p] == 1:
                    q_parts.append(p)

            n_q = sum(npartition_samples_np[p] for p in q_parts)
            if n_q == 0:
                continue

            C0m = get_cov_block(0, m)
            start = (m - 1) * nstats_per_model
            end = m * nstats_per_model
            CF_np[:, start:end] = -C0m / n_q

        # Build cf: Cov(LF_m, LF_k) for m, k = 1, ..., nmodels-1
        for m in range(1, nmodels):
            for k in range(1, nmodels):
                n_shared = 0
                for p in range(allocation_mat_np.shape[1]):
                    if allocation_mat_np[m, p] == 1 and allocation_mat_np[k, p] == 1:
                        n_shared += npartition_samples_np[p]

                if n_shared > 0:
                    Cmk = get_cov_block(m, k)
                    m_start = (m - 1) * nstats_per_model
                    m_end = m * nstats_per_model
                    k_start = (k - 1) * nstats_per_model
                    k_end = k * nstats_per_model
                    cf_np[m_start:m_end, k_start:k_end] = Cmk / n_shared

        return bkd.asarray(CF_np), bkd.asarray(cf_np)

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
