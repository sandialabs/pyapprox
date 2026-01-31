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
    ) -> Tuple[Array, Array, Array, Array]:
        """Compute pilot quantities for variance estimation.

        For variance estimation, we need:
        - Means and variances of each model
        - Fourth central moments (for variance of variance estimator)
        - Covariance of variance estimators across models

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        cov : Array
            Covariance of variance estimators. Shape: (nmodels*nqoi, nmodels*nqoi)
        means : Array
            Sample means. Shape: (nmodels, nqoi)
        variances : Array
            Sample variances. Shape: (nmodels, nqoi)
        fourth_moments : Array
            Fourth central moments. Shape: (nmodels, nqoi)
        """
        nmodels = len(pilot_values)
        nqoi = self._nqoi
        bkd = self._bkd

        # Validate all pilot values have same shape
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

        # Compute means and variances for each model
        means_list = []
        vars_list = []
        fourth_list = []

        for vals in pilot_values:
            n = vals.shape[0]
            mean = bkd.sum(vals, axis=0) / n
            centered = vals - mean
            var = bkd.sum(centered ** 2, axis=0) / (n - 1)
            fourth = bkd.sum(centered ** 4, axis=0) / n

            means_list.append(mean)
            vars_list.append(var)
            fourth_list.append(fourth)

        means = bkd.stack(means_list, axis=0)
        variances = bkd.stack(vars_list, axis=0)
        fourth_moments = bkd.stack(fourth_list, axis=0)

        # Compute covariance of variance estimators
        # For large n, Var(s^2) ≈ (mu_4 - sigma^4) / n + 2*sigma^4 / (n-1)
        # where mu_4 is the fourth central moment

        # Compute squared deviations for each model
        # Shape: (npilot, nmodels * nqoi)
        sq_devs_list = []
        for vals in pilot_values:
            mean = bkd.sum(vals, axis=0) / npilot
            centered = vals - mean
            sq_devs_list.append(centered ** 2)

        sq_devs = bkd.concatenate(sq_devs_list, axis=1)

        # Covariance of squared deviations (relates to covariance of variances)
        cov = self._compute_covariance(sq_devs)

        # Scale by (n-1)^2 for variance of sample variance
        # Var(s^2_i, s^2_j) = Cov((X-mu)^2, (Y-mu)^2) / (n-1)
        # but this is already accounted for in sample covariance with n-1

        return cov, means, variances, fourth_moments

    def set_pilot_quantities(
        self,
        cov: Array,
        means: Optional[Array] = None,
        variances: Optional[Array] = None,
        fourth_moments: Optional[Array] = None,
    ) -> None:
        """Set pilot quantities directly.

        Parameters
        ----------
        cov : Array
            Covariance of variance estimators. Shape: (nmodels*nqoi, nmodels*nqoi)
        means : Array, optional
            Sample means. Shape: (nmodels, nqoi)
        variances : Array, optional
            Sample variances. Shape: (nmodels, nqoi)
        fourth_moments : Array, optional
            Fourth central moments. Shape: (nmodels, nqoi)
        """
        self._cov = cov
        self._means = means
        self._variances = variances
        self._fourth_moments = fourth_moments

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity sample variance.

        The variance of the sample variance is approximately:
            Var(s^2) ≈ (mu_4 - (n-3)/(n-1)*sigma^4) / n

        where mu_4 is the fourth central moment.

        For simplicity, we use the asymptotic approximation:
            Var(s^2) ≈ mu_4/n - sigma^4*(n-3)/(n*(n-1))

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

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for CV estimator of variance with known LF stats.

        For a multi-model CV estimator with M models:
            s^2_CV = s^2_0 + sum_{m=1}^{M-1} eta_m * (sigma^2_m - s^2_m)

        where sigma^2_m are known LF variances. All models use shared samples.

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

        # Infer nmodels from covariance matrix
        nmodels = cov.shape[0] // nqoi

        # For CV with shared samples, all entries are 1/n
        n = npartition_samples[0]
        ncontrols = nmodels - 1

        # Build CF and cf using same pattern as mean
        CF = bkd.zeros((nqoi * ncontrols, nqoi * ncontrols))
        cf = bkd.zeros((nqoi, nqoi * ncontrols))

        for i in range(1, nmodels):
            for j in range(1, nmodels):
                Cij = cov[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]
                CF[(i-1)*nqoi:i*nqoi, (j-1)*nqoi:j*nqoi] = Cij / n

        for j in range(1, nmodels):
            C0j = cov[0*nqoi:(0+1)*nqoi, j*nqoi:(j+1)*nqoi]
            cf[:, (j-1)*nqoi:j*nqoi] = C0j / n

        return CF, cf

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator of variance.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix. Shape: (nmodels, npartitions)
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

        CF_np = np.zeros((nqoi, nqoi * ncontrols))
        cf_np = np.zeros((nqoi * ncontrols, nqoi * ncontrols))

        cov_np = bkd.to_numpy(cov)

        def get_cov_block(i: int, j: int) -> np.ndarray:
            return cov_np[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]

        for m in range(1, nmodels):
            q_parts = []
            for p in range(allocation_mat_np.shape[1]):
                if allocation_mat_np[m, p] == 1 and allocation_mat_np[0, p] == 1:
                    q_parts.append(p)

            n_q = sum(npartition_samples_np[p] for p in q_parts)

            if n_q == 0:
                continue

            C0m = get_cov_block(0, m)
            CF_np[:, (m-1)*nqoi:m*nqoi] = -C0m / n_q

        for m in range(1, nmodels):
            for k in range(1, nmodels):
                Cmk = get_cov_block(m, k)

                n_shared = 0
                for p in range(allocation_mat_np.shape[1]):
                    if allocation_mat_np[m, p] == 1 and allocation_mat_np[k, p] == 1:
                        n_shared += npartition_samples_np[p]

                if n_shared > 0:
                    cf_np[(m-1)*nqoi:m*nqoi, (k-1)*nqoi:k*nqoi] = Cmk / n_shared

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
        return f"MultiOutputVariance(nqoi={self._nqoi}, has_pilot_cov={has_cov})"
