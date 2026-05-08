"""
Ensemble methods for quantifying uncertainty in GP sensitivity indices.

This module provides the **refit algorithm** (``sobol_distribution_refit``)
for computing the distribution of Sobol indices across GP posterior
realizations.  It samples GP realizations at selected quadrature points Z,
fits a new GP at Z for each realization, and computes kernel-integral Sobol
indices.  Accurate for both sparse and dense training.

The algorithm returns ``Dict[int, Array]`` mapping variable index to an
array of Sobol index samples, shape ``(n_realizations,)``.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.util.backends.protocols import Array, Backend


class SobolThresholdSelector(Generic[Array]):
    """Select sample points Z where GP posterior std exceeds a threshold.

    Generates a large Sobol candidate set, evaluates GP posterior std σ*(z)
    at each candidate, keeps only those with σ*(z) > threshold, then returns
    the first ``n_points`` that pass.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        Fitted GP.
    marginals : List[MarginalProtocol[Array]]
        Marginal distributions (one per input dimension).
    bkd : Backend[Array]
        Numerical backend.
    oversampling_factor : int
        Candidate pool size = oversampling_factor * n_points.
    std_threshold_fraction : float
        Keep candidates with σ* > fraction * max(σ*).

    # TODO: Future improvement — replace Sobol+threshold with pivoted
    # Cholesky of Σ*(C,C). The pivoted Cholesky greedily selects points
    # that maximally reduce posterior variance, giving an optimal
    # space-filling design.
    # Protocol: start with candidate pool C (large Sobol set, excluding
    # ε-balls around training points), run pivoted Cholesky on Σ*(C,C)
    # with X_train as initial pivot set, take first n_sample_points
    # pivots beyond X_train as Z.
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        marginals: List[MarginalProtocol[Array]],
        bkd: Backend[Array],
        oversampling_factor: int = 5,
        std_threshold_fraction: float = 1e-6,
    ) -> None:
        self._gp = gp
        self._marginals = marginals
        self._bkd = bkd
        self._oversampling_factor = oversampling_factor
        self._std_threshold_fraction = std_threshold_fraction

    def select(
        self, n_points: int, seed: Optional[int] = None
    ) -> Array:
        """Return sample points Z of shape (nvars, n_points)."""
        from pyapprox.probability.joint.independent import (
            IndependentJoint,
        )
        from pyapprox.util.sampling.sobol import SobolSampler

        bkd = self._bkd
        nvars = len(self._marginals)

        # Generate candidate pool
        n_candidates = self._oversampling_factor * n_points
        dist = IndependentJoint(self._marginals, bkd)
        sampler = SobolSampler(
            nvars, bkd, distribution=dist, scramble=True, seed=seed,
        )
        candidates, _ = sampler.sample(n_candidates)

        # Evaluate posterior std
        post_std = self._gp.predict_std(candidates)  # (nqoi, n_cand)
        if post_std.ndim == 2:
            post_std = bkd.reshape(post_std, (-1,))

        # Filter by threshold
        max_std = bkd.max(post_std)
        threshold = self._std_threshold_fraction * max_std
        mask = post_std > threshold

        # Gather passing candidates
        # Use argsort trick to select indices where mask is True
        mask_float = mask * 1.0
        # Sort descending by mask value (1s first), then take first n
        sort_idx = bkd.argsort(-mask_float)
        n_passing = int(float(bkd.to_numpy(bkd.sum(mask_float))))
        n_select = min(n_points, n_passing)

        if n_select < n_points:
            # Not enough candidates passed — use all that did,
            # pad with remaining highest-std candidates
            sort_by_std = bkd.argsort(-post_std)
            selected_idx = sort_by_std[:n_points]
        else:
            selected_idx = sort_idx[:n_points]

        return candidates[:, selected_idx]


class GaussianProcessEnsemble(Generic[Array]):
    """
    Ensemble methods for GP sensitivity index uncertainty quantification.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process.
    gp_sensitivity : GaussianProcessSensitivity[Array]
        Sensitivity calculator with kernel integral matrices.

    Examples
    --------
    >>> ensemble = GaussianProcessEnsemble(gp, gp_sensitivity)
    >>> S_refit = ensemble.sobol_distribution_refit(
    ...     n_realizations=200, n_sample_points=300
    ... )
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        gp_sensitivity: GaussianProcessSensitivity[Array],
    ) -> None:
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before creating "
                "GaussianProcessEnsemble"
            )

        self._gp = gp
        self._sensitivity = gp_sensitivity
        self._bkd = gp.bkd()
        self._nvars = gp.nvars()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def gp(self) -> PredictiveGPProtocol[Array]:
        """Return the Gaussian Process."""
        return self._gp

    def sensitivity(self) -> GaussianProcessSensitivity[Array]:
        """Return the sensitivity calculator."""
        return self._sensitivity

    def sample_realizations(
        self,
        sample_points: Array,
        n_realizations: int,
        seed: Optional[int] = None,
    ) -> Array:
        """
        Sample GP realizations at given points.

        Uses the reparameterization trick:
            f^(r) = μ* + L @ ε^(r)

        Parameters
        ----------
        sample_points : Array
            Points at which to sample, shape (nvars, n_points).
        n_realizations : int
            Number of GP realizations to sample.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        realizations : Array
            Shape (n_realizations, n_points).
        """
        bkd = self._bkd
        n_points = sample_points.shape[1]

        # Posterior mean
        mean = self._gp.predict(sample_points)
        if mean.shape[0] != 1:
            raise NotImplementedError(
                "sample_realizations currently only "
                "supports single-output GPs (nqoi=1)"
            )
        mean = bkd.reshape(mean, (-1,))

        # Posterior covariance
        cov = self._gp.predict_covariance(sample_points)
        cov = 0.5 * (cov + cov.T)
        cov = cov + 1e-10 * bkd.eye(n_points)

        # Cholesky
        try:
            L = bkd.cholesky(cov)
        except Exception:
            eigenvalues, eigenvectors = bkd.eigh(cov)
            eigenvalues = eigenvalues * (eigenvalues > 0)
            L = eigenvectors * bkd.sqrt(eigenvalues)

        # Random samples
        if seed is not None:
            np.random.seed(seed + 12345)

        epsilon_np = np.random.randn(n_points, n_realizations)
        epsilon = bkd.asarray(epsilon_np.tolist())

        realizations_T = L @ epsilon + bkd.reshape(mean, (-1, 1))
        return realizations_T.T

    def _compute_sobol_from_alpha(
        self,
        alpha_1d: Array,
        tau_K: Array,
        sens: GaussianProcessSensitivity[Array],
    ) -> Tuple[Dict[int, Array], Array]:
        """Compute Sobol indices from a weight vector.

        Returns (main_effects, total_var) where main_effects maps
        variable index to S_i scalar.
        """
        bkd = self._bkd
        nvars = sens.nvars()

        # η = αᵀ τ_K
        eta = alpha_1d @ tau_K

        # ζ = αᵀ P_K α (full conditioning)
        index_all = bkd.asarray([1.0] * nvars)
        zeta, _ = sens._compute_zeta_p(index_all, alpha_1d=alpha_1d)

        # Var_X[μ*] = ζ - η²
        total_var = zeta - eta * eta
        total_var = total_var * (total_var >= 0.0)

        main_effects: Dict[int, Array] = {}
        for i in range(nvars):
            index_list = [0.0] * nvars
            index_list[i] = 1.0
            index_i = bkd.asarray(index_list)

            zeta_p, _ = sens._compute_zeta_p(
                index_i, alpha_1d=alpha_1d
            )
            V_i = zeta_p - eta * eta
            V_i = V_i * (V_i >= 0.0)

            S_i = V_i / (total_var + 1e-15)
            S_i = S_i * (S_i >= 0.0)
            S_i = S_i * (S_i <= 1.0) + (S_i > 1.0) * 1.0
            main_effects[i] = S_i

        return main_effects, total_var

    def sobol_distribution_refit(
        self,
        n_realizations: int,
        n_sample_points: int,
        selector: Optional[SobolThresholdSelector[Array]] = None,
        seed: Optional[int] = None,
    ) -> Dict[int, Array]:
        """
        Sobol index distribution via refit at new quadrature points.

        1. Select n_sample_points points Z via selector
        2. Sample GP realizations at Z
        3. For each realization, solve for α and compute Sobol indices
           using kernel integral matrices at Z

        Parameters
        ----------
        n_realizations : int
            Number of posterior realizations.
        n_sample_points : int
            Number of quadrature points Z.
        selector : SobolThresholdSelector, optional
            Point selector. Default: SobolThresholdSelector with
            marginals from the integral calculator.
        seed : int, optional
            Random seed.

        Returns
        -------
        Dict[int, Array]
            Variable index → shape (n_realizations,) Sobol samples.
        """
        from pyapprox.surrogates.gaussianprocess import (
            ExactGaussianProcess,
        )
        from pyapprox.surrogates.gaussianprocess.statistics import (
            GaussianProcessStatistics,
            SeparableKernelIntegralCalculator,
        )

        bkd = self._bkd
        calc = self._sensitivity._calc

        # Select sample points
        if selector is None:
            selector = SobolThresholdSelector(
                self._gp, calc.marginals(), bkd
            )
        Z = selector.select(n_sample_points, seed=seed)

        # Sample realizations at Z
        realizations = self.sample_realizations(
            Z, n_realizations, seed=seed
        )

        # Create a new GP at Z with same kernel, fixed hyperparameters
        # Clone kernel structure
        kernel = self._gp.kernel()
        gp_z = ExactGaussianProcess(
            kernel, nvars=self._nvars, bkd=bkd, nugget=1e-10,
        )
        gp_z.hyp_list().set_all_inactive()

        # Fit once to get Cholesky of k(Z,Z) + nugget*I
        # Use first realization as dummy y — Cholesky depends only on Z
        y_dummy = bkd.reshape(realizations[0, :], (1, -1))
        gp_z.fit(Z, y_dummy)

        # Create kernel integral calculator at Z
        # Need quadrature bases for the marginals
        from pyapprox.surrogates.sparsegrids.basis_factory import (
            create_basis_factories,
        )

        marginals = calc.marginals()
        factories = create_basis_factories(marginals, bkd, "gauss")
        bases: List[Any] = [f.create_basis() for f in factories]
        for b in bases:
            b.set_nterms(50)

        calc_z = SeparableKernelIntegralCalculator(
            gp_z, bases, marginals, bkd=bkd
        )
        stats_z = GaussianProcessStatistics(gp_z, calc_z)
        sens_z = GaussianProcessSensitivity(stats_z)

        # Get Cholesky and tau_K from Z-based calculator
        chol = gp_z.cholesky()
        tau_K = calc_z.tau_K()

        # Compute Sobol indices for each realization
        result: Dict[int, list[Array]] = {
            i: [] for i in range(self._nvars)
        }

        for r in range(n_realizations):
            f_r = realizations[r, :]  # shape (n_sample_points,)
            f_r_col = bkd.reshape(f_r, (-1, 1))

            # α^(r) = (k(Z,Z)+nugget·I)⁻¹ f^(r)(Z)
            alpha_r = bkd.reshape(chol.solve(f_r_col), (-1,))

            effects, _ = self._compute_sobol_from_alpha(
                alpha_r, tau_K, sens_z
            )
            for i in range(self._nvars):
                result[i].append(effects[i])

        return {i: bkd.stack(v) for i, v in result.items()}

