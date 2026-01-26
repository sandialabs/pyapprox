"""
Monte Carlo ensemble for quantifying uncertainty in GP sensitivity indices.

This module provides the GaussianProcessEnsemble class which samples GP realizations
to compute the distribution of sensitivity indices, enabling uncertainty quantification
beyond the point estimates provided by the analytical formulas.

Mathematical Background
-----------------------
The analytical formulas from GaussianProcessSensitivity give E[S_i], the expected
value of the Sobol index under GP uncertainty. However, S_i = V_i/γ_f is a ratio
of random variables, so E[S_i] ≠ E[V_i]/E[γ_f] (Jensen's inequality).

To quantify the full distribution of S_i, we:
1. Sample GP realizations: f^(r)(z) ~ GP posterior
2. For each realization, compute:
   - μ_f^(r) = ∫ f^(r)(z) p(z) dz  (mean)
   - κ_f^(r) = ∫ [f^(r)(z)]² p(z) dz  (second moment)
   - γ_f^(r) = κ_f^(r) - [μ_f^(r)]²  (variance)
   - V_i^(r) = conditional variance for variable i
   - S_i^(r) = V_i^(r) / γ_f^(r)
3. Return the empirical distribution of S_i^(r)

Critical Implementation Note
----------------------------
Quadrature points must differ from training points! At training points,
the GP posterior variance is zero (exact interpolation), so all realizations
have identical values there. This gives artificially zero variation in the
computed statistics.
"""

from typing import Generic, Optional, Tuple, Dict

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.typing.expdesign.protocols.quadrature import (
    QuadratureSamplerProtocol,
)


class GaussianProcessEnsemble(Generic[Array]):
    """
    Monte Carlo ensemble for GP sensitivity index uncertainty quantification.

    This class samples GP realizations and computes the distribution of Sobol
    sensitivity indices, providing uncertainty quantification beyond point
    estimates.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process.
    gp_sensitivity : GaussianProcessSensitivity[Array]
        Sensitivity calculator for computing E[S_i] (used for validation).
    sampler : QuadratureSamplerProtocol[Array], optional
        Sampler for generating integration points. If None, uses SobolSampler
        with the same distribution as used in gp_sensitivity.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    ...     GaussianProcessEnsemble,
    ...     GaussianProcessSensitivity,
    ... )
    >>> # Assume gp is fitted, gp_sensitivity is created
    >>> ensemble = GaussianProcessEnsemble(gp, gp_sensitivity)
    >>> S_dist = ensemble.compute_sobol_distribution(
    ...     n_realizations=1000,
    ...     n_sample_points=500
    ... )
    >>> S_0_samples = S_dist[0]  # Array of S_0 values across realizations
    >>> S_0_mean = bkd.mean(S_0_samples)
    >>> S_0_std = bkd.std(S_0_samples)

    Notes
    -----
    The sample points used for Monte Carlo integration are generated using
    the provided sampler (default: Sobol sequence). These points must differ
    from the GP training points to ensure non-degenerate variance estimates.
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        gp_sensitivity: GaussianProcessSensitivity[Array],
        sampler: Optional[QuadratureSamplerProtocol[Array]] = None,
    ) -> None:
        # Validate GP is fitted
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before creating GaussianProcessEnsemble"
            )

        self._gp = gp
        self._sensitivity = gp_sensitivity
        self._bkd = gp.bkd()
        self._nvars = gp.nvars()

        # Store sampler (or create default later when needed)
        self._sampler = sampler

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

    def _generate_sample_points(
        self, n_points: int, seed: Optional[int] = None
    ) -> Tuple[Array, Array]:
        """
        Generate sample points for Monte Carlo integration.

        Uses the provided sampler or creates a default SobolSampler.

        Parameters
        ----------
        n_points : int
            Number of integration points.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        points : Array
            Sample points, shape (nvars, n_points).
        weights : Array
            Quadrature weights, shape (n_points,).
        """
        if self._sampler is not None:
            self._sampler.reset()
            return self._sampler.sample(n_points)

        # Create default SobolSampler with uniform distribution on domain
        # Get marginals from the integral calculator
        from pyapprox.typing.expdesign.quadrature.sobol import SobolSampler
        from pyapprox.typing.probability.joint.independent import (
            IndependentJoint,
        )

        # Get marginals from the integral calculator
        calc = self._sensitivity._calc
        marginals = calc.marginals()

        # Create joint distribution
        dist = IndependentJoint(marginals, self._bkd)

        # Create Sobol sampler
        sampler = SobolSampler(
            self._nvars,
            self._bkd,
            distribution=dist,
            scramble=True,
            seed=seed,
        )

        return sampler.sample(n_points)

    def sample_realizations(
        self,
        n_realizations: int,
        n_sample_points: int,
        seed: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Sample GP realizations at Monte Carlo integration points.

        For each realization r, samples f^(r)(z_j) at integration points z_j.
        Uses the reparameterization trick:
            f^(r) = μ* + L @ ε^(r)
        where μ* is the posterior mean, L = chol(Σ*) is the Cholesky of the
        posterior covariance, and ε^(r) ~ N(0, I).

        Parameters
        ----------
        n_realizations : int
            Number of GP realizations to sample.
        n_sample_points : int
            Number of Monte Carlo integration points.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        realizations : Array
            GP realizations, shape (n_realizations, n_sample_points).
            realizations[r, j] = f^(r)(z_j)
        sample_points : Array
            Integration points, shape (nvars, n_sample_points).
        weights : Array
            Quadrature weights, shape (n_sample_points,).
            For MC, weights are uniform: 1/n_sample_points.

        Notes
        -----
        The sample points are generated independently of the training points
        to ensure the GP posterior has non-zero variance, which is necessary
        for meaningful ensemble statistics.
        """
        bkd = self._bkd

        # Generate integration points
        sample_points, weights = self._generate_sample_points(
            n_sample_points, seed=seed
        )

        # Get posterior mean and covariance at sample points
        # predict returns (n_sample_points, nqoi), we want (n_sample_points,)
        mean = self._gp.predict(sample_points)  # Shape: (n_sample_points, nqoi)
        if mean.shape[1] != 1:
            raise NotImplementedError(
                "sample_realizations currently only supports single-output GPs (nqoi=1)"
            )
        mean = bkd.reshape(mean, (-1,))  # Shape: (n_sample_points,)

        # Get posterior covariance
        cov = self._gp.predict_covariance(sample_points)  # (n_sample_points, n_sample_points)

        # Ensure symmetry and add small nugget for numerical stability
        cov = 0.5 * (cov + cov.T)
        nugget = 1e-10 * bkd.eye(n_sample_points)
        cov = cov + nugget

        # Cholesky decomposition for sampling
        # f = μ + L @ ε where ε ~ N(0, I)
        try:
            L = bkd.cholesky(cov)  # Lower triangular
        except Exception:
            # Fall back to SVD-based sampling if Cholesky fails
            # Use eigendecomposition: Σ = V Λ V^T
            eigenvalues, eigenvectors = bkd.eigh(cov)
            # Clamp negative eigenvalues
            eigenvalues = eigenvalues * (eigenvalues > 0)
            # L = V @ sqrt(Λ)
            L = eigenvectors * bkd.sqrt(eigenvalues)

        # Generate random samples
        if seed is not None:
            np.random.seed(seed + 12345)  # Offset to avoid correlation with sampler

        # Standard normal samples: shape (n_sample_points, n_realizations)
        epsilon_np = np.random.randn(n_sample_points, n_realizations)
        epsilon = bkd.asarray(epsilon_np.tolist())

        # Transform: f = μ + L @ ε
        # mean shape: (n_sample_points,), broadcast to all realizations
        # L @ epsilon: (n_sample_points, n_sample_points) @ (n_sample_points, n_realizations)
        #            = (n_sample_points, n_realizations)
        realizations_T = L @ epsilon + bkd.reshape(mean, (-1, 1))

        # Transpose to (n_realizations, n_sample_points)
        realizations = realizations_T.T

        return realizations, sample_points, weights

    def _compute_variance_from_samples(
        self, values: Array, weights: Array
    ) -> Array:
        """
        Compute weighted variance from samples.

        Var[f] = E[f²] - E[f]²

        Parameters
        ----------
        values : Array
            Function values, shape (n_samples,) or (n_realizations, n_samples).
        weights : Array
            Quadrature weights, shape (n_samples,).

        Returns
        -------
        variance : Array
            Variance estimate. Scalar if values is 1D, shape (n_realizations,) if 2D.
        """
        bkd = self._bkd

        if values.ndim == 1:
            # Single set of values
            mean = bkd.sum(weights * values)
            second_moment = bkd.sum(weights * values * values)
            variance = second_moment - mean * mean
            return variance
        else:
            # Multiple realizations: (n_realizations, n_samples)
            # Compute mean and second moment for each realization
            # weights shape: (n_samples,)
            mean = values @ weights  # (n_realizations,)
            second_moment = (values * values) @ weights  # (n_realizations,)
            variance = second_moment - mean * mean
            return variance

    def _compute_conditional_variance_mc(
        self,
        realizations: Array,
        sample_points: Array,
        weights: Array,
        var_index: int,
    ) -> Array:
        """
        Compute conditional variance V_i for each realization using MC.

        V_i = E_{z_~i}[Var_{z_i}[f | z_~i]]

        This is approximated by:
        1. Grouping sample points by their z_~i values
        2. Computing variance within each group
        3. Averaging across groups

        For MC with Sobol points, we use a simpler approach:
        V_i ≈ Var[E[f | z_i]] = E[E[f | z_i]²] - E[E[f | z_i]]²

        This is computed via ANOVA decomposition using the Sobol approach.

        Parameters
        ----------
        realizations : Array
            GP realizations, shape (n_realizations, n_sample_points).
        sample_points : Array
            Integration points, shape (nvars, n_sample_points).
        weights : Array
            Quadrature weights, shape (n_sample_points,).
        var_index : int
            Index of the variable for which to compute V_i.

        Returns
        -------
        V_i : Array
            Conditional variance for each realization, shape (n_realizations,).
        """
        bkd = self._bkd
        n_realizations = realizations.shape[0]
        n_samples = realizations.shape[1]

        # For a proper Sobol decomposition, we would need a special sample design
        # (e.g., Saltelli's method). Here we use a simpler approximation:
        # Sort samples by z_i and compute variance of conditional means.

        # Get the values for variable i
        z_i = sample_points[var_index, :]  # Shape: (n_samples,)

        # Sort samples by z_i
        sort_idx = bkd.argsort(z_i)
        z_i_sorted = z_i[sort_idx]

        # Group samples into bins and compute conditional mean within each bin
        # Use a reasonable number of bins
        n_bins = min(20, n_samples // 5)
        if n_bins < 2:
            # Not enough samples for reliable estimation
            return bkd.zeros((n_realizations,))

        # Compute bin edges
        bin_edges_np = np.linspace(
            float(bkd.to_numpy(bkd.min(z_i))),
            float(bkd.to_numpy(bkd.max(z_i))) + 1e-10,
            n_bins + 1,
        )
        bin_edges = bkd.asarray(bin_edges_np.tolist())

        # Compute V_i for each realization
        V_i_list = []
        for r in range(n_realizations):
            f_r = realizations[r, :]  # Shape: (n_samples,)
            f_r_sorted = f_r[sort_idx]

            # Compute conditional means E[f | z_i in bin_b]
            cond_means = []
            cond_weights = []
            for b in range(n_bins):
                # Find samples in this bin
                in_bin = (z_i_sorted >= bin_edges[b]) & (z_i_sorted < bin_edges[b + 1])
                # Convert boolean mask to float: True -> 1.0, False -> 0.0
                in_bin_float = in_bin * 1.0
                n_in_bin = int(bkd.to_numpy(bkd.sum(in_bin_float)))
                if n_in_bin > 0:
                    # Compute mean within bin
                    f_in_bin = f_r_sorted * in_bin_float
                    cond_mean = bkd.sum(f_in_bin) / n_in_bin
                    cond_means.append(cond_mean)
                    cond_weights.append(float(n_in_bin) / n_samples)

            if len(cond_means) < 2:
                V_i_list.append(bkd.asarray(0.0))
                continue

            # Compute variance of conditional means
            cond_means_arr = bkd.stack(cond_means)
            cond_weights_arr = bkd.asarray(cond_weights)
            cond_weights_arr = cond_weights_arr / bkd.sum(cond_weights_arr)

            mean_of_cond_means = bkd.sum(cond_weights_arr * cond_means_arr)
            V_i_r = bkd.sum(
                cond_weights_arr * (cond_means_arr - mean_of_cond_means) ** 2
            )
            V_i_list.append(V_i_r)

        V_i = bkd.stack(V_i_list)
        return V_i

    def compute_sobol_distribution(
        self,
        n_realizations: int,
        n_sample_points: int,
        seed: Optional[int] = None,
    ) -> Dict[int, Array]:
        """
        Compute the distribution of Sobol indices across GP realizations.

        For each realization r:
        1. Compute μ_f^(r) = ∫ f^(r)(z) p(z) dz
        2. Compute κ_f^(r) = ∫ [f^(r)(z)]² p(z) dz
        3. Compute γ_f^(r) = κ_f^(r) - [μ_f^(r)]²
        4. Compute V_i^(r) = conditional variance for variable i
        5. Compute S_i^(r) = V_i^(r) / γ_f^(r)

        Parameters
        ----------
        n_realizations : int
            Number of GP realizations to sample.
        n_sample_points : int
            Number of Monte Carlo integration points.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping variable index i to array of S_i values
            across realizations. Each array has shape (n_realizations,).

        Examples
        --------
        >>> S_dist = ensemble.compute_sobol_distribution(1000, 500)
        >>> S_0_samples = S_dist[0]
        >>> print(f"E[S_0] = {bkd.mean(S_0_samples):.4f}")
        >>> print(f"Std[S_0] = {bkd.std(S_0_samples):.4f}")
        """
        bkd = self._bkd

        # Sample realizations
        realizations, sample_points, weights = self.sample_realizations(
            n_realizations, n_sample_points, seed=seed
        )

        # Compute total variance γ_f^(r) for each realization
        # γ_f^(r) = E[f²] - E[f]² = κ_f - μ_f²
        gamma_f = self._compute_variance_from_samples(realizations, weights)

        # Avoid division by zero - clamp small variances
        gamma_f = bkd.maximum(gamma_f, bkd.asarray(1e-15))

        # Compute V_i for each variable and each realization
        sobol_distribution: Dict[int, Array] = {}
        for i in range(self._nvars):
            V_i = self._compute_conditional_variance_mc(
                realizations, sample_points, weights, i
            )
            # S_i = V_i / γ_f
            S_i = V_i / gamma_f
            # Clamp to [0, 1]
            S_i = bkd.maximum(S_i, bkd.asarray(0.0))
            S_i = bkd.minimum(S_i, bkd.asarray(1.0))
            sobol_distribution[i] = S_i

        return sobol_distribution
