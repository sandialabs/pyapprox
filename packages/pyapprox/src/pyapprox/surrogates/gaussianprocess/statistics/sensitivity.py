"""
Gaussian Process sensitivity analysis module.

This module provides the GaussianProcessSensitivity class which computes
Sobol sensitivity indices from fitted Gaussian Processes.

Sobol indices quantify the contribution of each input variable (or set of
variables) to the total variance of the output. They are computed by
integrating conditional variances over the input space.

Key quantities:
- Main effect V_i: Variance explained by variable i alone
- Total effect V_Ti: Variance NOT explained when all except i are fixed
- Sobol indices: S_i = V_i / Var[f], T_i = V_Ti / Var[f]

Mathematical Background
-----------------------
For a GP with posterior mean μ(x) and variance γ(x), the conditional variance
when variables in index set p are fixed is:

E[γ_f^(p)] = ζ_p + s² v_p² - η² - s² ς²

where:
- ζ_p = y^T A^{-1} P_p A^{-1} y
- v_p² = u_p - Tr[P_p A^{-1}]
- η and ς² are unchanged (depend on τ, not P)

The main effect index for variable i uses p = {i} (only z_i conditioned).
The total effect index for variable i uses p = {all except i}.
"""

from typing import Dict, Generic, Optional, Union

from pyapprox.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.moments import (
    GaussianProcessStatistics,
)
from pyapprox.util.backends.protocols import Array, Backend


class GaussianProcessSensitivity(Generic[Array]):
    """
    Compute Sobol sensitivity indices from a fitted Gaussian Process.

    This class wraps GaussianProcessStatistics and provides methods to compute
    first-order (main effect) and total effect Sobol indices.

    Sobol indices quantify:
    - S_i (main effect): Fraction of variance due to variable i alone
    - T_i (total effect): Fraction of variance involving variable i

    The indices satisfy:
    - 0 ≤ S_i ≤ 1, 0 ≤ T_i ≤ 1
    - Σ_i S_i ≤ 1 (equality for additive functions)
    - T_i ≥ S_i
    - Σ_i T_i ≥ 1 (equality for additive functions)

    Parameters
    ----------
    gp_stats : GaussianProcessStatistics[Array]
        A GaussianProcessStatistics object with computed moments.

    Examples
    --------
    >>> from pyapprox.surrogates.gaussianprocess.statistics import (
    ...     SeparableKernelIntegralCalculator,
    ...     GaussianProcessStatistics,
    ...     GaussianProcessSensitivity,
    ... )
    >>> # Assume gp is a fitted GP with separable kernel
    >>> calc = SeparableKernelIntegralCalculator(gp, bases, bkd=bkd)
    >>> stats = GaussianProcessStatistics(gp, calc)
    >>> sens = GaussianProcessSensitivity(stats)
    >>> main_effects = sens.main_effect_indices()
    >>> total_effects = sens.total_effect_indices()
    """

    def __init__(
        self,
        gp_stats: GaussianProcessStatistics[Array],
    ):
        self._stats = gp_stats
        self._bkd = gp_stats.bkd()

        # Get the integral calculator from stats
        # - must be SeparableKernelIntegralCalculator
        # to access the 1D kernel components for nvars
        calc = gp_stats._calc
        if not isinstance(calc, SeparableKernelIntegralCalculator):
            raise TypeError(
                f"gp_stats._calc must be SeparableKernelIntegralCalculator, "
                f"got {type(calc).__name__}"
            )
        self._calc: SeparableKernelIntegralCalculator[Array] = calc

        # Get number of input variables from the GP
        self._nvars = len(self._calc._kernels_1d)

        # Cache for computed quantities (Array or Dict[int, Array])
        self._cache: Dict[str, Union[Array, Dict[int, Array]]] = {}

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def _get_kernel_variance(self) -> Array:
        """
        Get the kernel variance hyperparameter (s²).

        Returns
        -------
        s2 : Array
            Kernel variance (scalar).
        """
        return self._stats._get_kernel_variance()

    def _get_alpha_1d(self) -> Array:
        """Return fitted alpha as 1D, shape (n_train,)."""
        alpha = self._stats._gp.alpha()  # Shape: (nqoi, n_train)
        if alpha.shape[0] == 1:
            return self._bkd.reshape(alpha, (-1,))
        raise NotImplementedError(
            "Currently only supports single-output GPs (nqoi=1)"
        )

    def _get_P_K_p(self, index: Array) -> Array:
        """Return s⁴ · conditional_P(index), shape (n_train, n_train)."""
        s2 = self._get_kernel_variance()
        s4 = s2 * s2
        return s4 * self._calc.conditional_P(index)

    def _compute_zeta_p(
        self,
        index: Array,
        alpha_1d: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Compute ζ_p = αᵀ P_K_p α for a given conditioning index.

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
        alpha_1d : Array, optional
            Weight vector of shape (n_train,). When None, uses
            the fitted GP alpha.

        Returns
        -------
        zeta_p : Array
            Scalar ζ_p = αᵀ P_K_p α.
        P_K_p : Array
            Scaled conditional kernel integral matrix,
            shape (n_train, n_train).
        """
        P_K_p = self._get_P_K_p(index)

        if alpha_1d is None:
            alpha_1d = self._get_alpha_1d()

        # ζ_p = αᵀ P_K_p α
        P_K_p_alpha = P_K_p @ alpha_1d
        zeta_p = alpha_1d @ P_K_p_alpha

        return zeta_p, P_K_p

    def _conditional_variance_scaled(self, index: Array) -> Array:
        """Compute conditional variance in scaled (internal) space.

        E[γ_f^(p)]_scaled = ζ_p + v_p² - η² - var_mu

        All quantities are in scaled (kernel) space.

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
            index[k] = 1: dimension k is CONDITIONED ON
            index[k] = 0: dimension k is INTEGRATED OUT

        Returns
        -------
        Array
            Conditional variance in scaled space (scalar, non-negative).
        """
        zeta_p, P_K_p = self._compute_zeta_p(index)

        # u_K_p for integrated posterior variance term
        s2 = self._get_kernel_variance()
        u_C_p = self._calc.conditional_u(index)
        u_K_p = s2 * u_C_p

        # v_p² = u_K_p - tr[P_K_p A⁻¹]
        A_inv_P_K_p = self._stats._solve(P_K_p)
        trace_P_K_p_A_inv = self._bkd.trace(A_inv_P_K_p)
        v_p_sq = u_K_p - trace_P_K_p_A_inv

        # Use scaled versions (no output transform applied)
        eta = self._stats._input_mean_of_posterior_mean_scaled()
        var_mu = self._stats._gp_variance_of_posterior_mean_scaled()

        # E[γ_f^(p)] = ζ_p + v_p² - η² - var_mu
        cond_var = zeta_p + v_p_sq - eta * eta - var_mu

        # Ensure non-negative (numerical stability)
        cond_var = cond_var * (cond_var >= 0.0)

        return cond_var

    def conditional_variance(self, index: Array) -> Array:
        """
        Compute the conditional variance E[γ_f^(p)].

        This computes the expected GP variance when variables in the index set
        p are conditioned on (fixed) and the remaining variables are integrated
        out. Returns values in the original output space if an output
        transform is set on the GP.

        E[γ_f^(p)] = ζ_p + s² v_p² - η² - s² ς²

        where:
        - ζ_p = y^T A^{-1} P_p A^{-1} y
        - v_p² = u_p - Tr[P_p A^{-1}]
        - η = E[μ_f] (mean of mean)
        - ς² = u - τ^T A^{-1} τ

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
            index[k] = 1: dimension k is CONDITIONED ON
            index[k] = 0: dimension k is INTEGRATED OUT

        Returns
        -------
        Array
            Conditional variance (scalar, non-negative).

        Examples
        --------
        # Main effect: only variable 0 conditioned
        index = bkd.asarray([1.0, 0.0, 0.0])  # nvars=3
        V_0 = sens.conditional_variance(index)

        # Total effect complement: all except variable 0 conditioned
        index = bkd.asarray([0.0, 1.0, 1.0])  # nvars=3
        V_not_0 = sens.conditional_variance(index)
        """
        cond_var_scaled = self._conditional_variance_scaled(index)

        transform = self._stats._get_output_transform()
        if transform is None:
            return cond_var_scaled

        # V_p_orig = σ_y² * V_p_scaled
        sigma_y_sq = transform.scale()[0] ** 2
        return sigma_y_sq * cond_var_scaled

    def main_effect_indices(self) -> Dict[int, Array]:
        """
        Compute main effect (first-order) Sobol indices.

        The main effect index S_i measures the fraction of variance
        explained by variable i alone:

        S_i = V_i / E[γ_f]

        where V_i is the conditional variance when only z_i is conditioned on.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping variable index to its main effect Sobol index.
            Keys are 0, 1, ..., nvars-1.

        Examples
        --------
        >>> main_effects = sens.main_effect_indices()
        >>> S_0 = main_effects[0]  # Main effect of variable 0
        """
        if 'main_effect_indices' in self._cache:
            cached = self._cache['main_effect_indices']
            assert isinstance(cached, dict)
            return cached

        # Get total variance E[γ_f]
        total_variance = self._stats.input_mean_of_posterior_variance()

        indices: Dict[int, Array] = {}
        for i in range(self._nvars):
            # Create index vector: only variable i is conditioned on
            # index[i] = 1, all others = 0
            index_list = [0.0] * self._nvars
            index_list[i] = 1.0
            index = self._bkd.asarray(index_list)

            # V_i = conditional_variance with only z_i conditioned
            V_i = self.conditional_variance(index)

            # S_i = V_i / E[γ_f]
            # Handle division by zero (if total variance is zero)
            S_i = V_i / (total_variance + 1e-15)

            # Clamp to [0, 1] for numerical stability
            S_i = S_i * (S_i >= 0.0)
            S_i = S_i * (S_i <= 1.0) + (S_i > 1.0) * 1.0

            indices[i] = S_i

        self._cache['main_effect_indices'] = indices
        return indices

    def total_effect_indices(self) -> Dict[int, Array]:
        """
        Compute total effect Sobol indices.

        The total effect index T_i measures the fraction of variance
        that involves variable i (including interactions):

        T_i = V_Ti / E[γ_f] = 1 - V_~i / E[γ_f]

        where V_~i is the conditional variance when all variables except z_i
        are conditioned on.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping variable index to its total effect Sobol index.
            Keys are 0, 1, ..., nvars-1.

        Examples
        --------
        >>> total_effects = sens.total_effect_indices()
        >>> T_0 = total_effects[0]  # Total effect of variable 0
        """
        if 'total_effect_indices' in self._cache:
            cached = self._cache['total_effect_indices']
            assert isinstance(cached, dict)
            return cached

        # Get total variance E[γ_f]
        total_variance = self._stats.input_mean_of_posterior_variance()

        indices: Dict[int, Array] = {}
        for i in range(self._nvars):
            # Create index vector: all variables except i are conditioned on
            # index[i] = 0, all others = 1
            index_list = [1.0] * self._nvars
            index_list[i] = 0.0
            index = self._bkd.asarray(index_list)

            # V_~i = conditional_variance with all except z_i conditioned
            V_not_i = self.conditional_variance(index)

            # T_i = 1 - V_~i / E[γ_f]
            # Note: V_Ti = E[γ_f] - V_~i, so T_i = V_Ti / E[γ_f] = 1 - V_~i / E[γ_f]
            T_i = 1.0 - V_not_i / (total_variance + 1e-15)

            # Clamp to [0, 1] for numerical stability
            T_i = T_i * (T_i >= 0.0)
            T_i = T_i * (T_i <= 1.0) + (T_i > 1.0) * 1.0

            indices[i] = T_i

        self._cache['total_effect_indices'] = indices
        return indices

    # ===== Posterior-mean Sobol indices =====

    def _conditional_variance_of_posterior_mean_scaled(
        self, index: Array
    ) -> Array:
        """Compute Var_X[μ*(X_p)] in scaled space.

        ζ_p - η², where ζ_p = αᵀ P_K_p α.

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        zeta_p, _ = self._compute_zeta_p(index)
        eta = self._stats._input_mean_of_posterior_mean_scaled()
        cond_var = zeta_p - eta * eta
        return cond_var * (cond_var >= 0.0)

    def conditional_variance_of_posterior_mean(self, index: Array) -> Array:
        """
        Compute the conditional variance of the posterior mean.

        Var_X[E[μ*(X) | X_p]] = ζ_p - η²

        This is the variance explained by conditioning on variables in
        the index set p, treating μ* as a deterministic function.
        Returns values in the original output space if an output
        transform is set on the GP.

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
            index[k] = 1: dimension k is CONDITIONED ON
            index[k] = 0: dimension k is INTEGRATED OUT

        Returns
        -------
        Array
            Conditional variance (scalar, non-negative).
        """
        cond_var_scaled = (
            self._conditional_variance_of_posterior_mean_scaled(index)
        )

        transform = self._stats._get_output_transform()
        if transform is None:
            return cond_var_scaled

        sigma_y_sq = transform.scale()[0] ** 2
        return sigma_y_sq * cond_var_scaled

    def main_effect_indices_of_posterior_mean(self) -> Dict[int, Array]:
        """
        Compute main effect Sobol indices of the posterior mean.

        S_i = Var_X[E[μ*(X) | X_i]] / Var_X[μ*(X)]

        These treat the posterior mean μ* as a deterministic surrogate,
        stripping epistemic uncertainty from the sensitivity analysis.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping variable index to its main effect Sobol index.
        """
        if 'main_effect_indices_pm' in self._cache:
            cached = self._cache['main_effect_indices_pm']
            assert isinstance(cached, dict)
            return cached

        total_var = self._stats.input_variance_of_posterior_mean()

        indices: Dict[int, Array] = {}
        for i in range(self._nvars):
            index_list = [0.0] * self._nvars
            index_list[i] = 1.0
            index = self._bkd.asarray(index_list)

            V_i = self.conditional_variance_of_posterior_mean(index)
            S_i = V_i / (total_var + 1e-15)

            # Clamp to [0, 1]
            S_i = S_i * (S_i >= 0.0)
            S_i = S_i * (S_i <= 1.0) + (S_i > 1.0) * 1.0

            indices[i] = S_i

        self._cache['main_effect_indices_pm'] = indices
        return indices

    def total_effect_indices_of_posterior_mean(self) -> Dict[int, Array]:
        """
        Compute total effect Sobol indices of the posterior mean.

        T_i = 1 - Var_X[E[μ*(X) | X_{~i}]] / Var_X[μ*(X)]

        These treat the posterior mean μ* as a deterministic surrogate,
        stripping epistemic uncertainty from the sensitivity analysis.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping variable index to its total effect Sobol index.
        """
        if 'total_effect_indices_pm' in self._cache:
            cached = self._cache['total_effect_indices_pm']
            assert isinstance(cached, dict)
            return cached

        total_var = self._stats.input_variance_of_posterior_mean()

        indices: Dict[int, Array] = {}
        for i in range(self._nvars):
            index_list = [1.0] * self._nvars
            index_list[i] = 0.0
            index = self._bkd.asarray(index_list)

            V_not_i = self.conditional_variance_of_posterior_mean(index)
            T_i = 1.0 - V_not_i / (total_var + 1e-15)

            # Clamp to [0, 1]
            T_i = T_i * (T_i >= 0.0)
            T_i = T_i * (T_i <= 1.0) + (T_i > 1.0) * 1.0

            indices[i] = T_i

        self._cache['total_effect_indices_pm'] = indices
        return indices
