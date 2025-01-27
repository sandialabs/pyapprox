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

from typing import Generic, Dict, Union
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.statistics.moments import (
    GaussianProcessStatistics,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)


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
    >>> from pyapprox.typing.surrogates.gaussianprocess.statistics import (
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

        # Get the integral calculator from stats - must be SeparableKernelIntegralCalculator
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

    def conditional_variance(self, index: Array) -> Array:
        """
        Compute the conditional variance E[γ_f^(p)].

        This computes the expected GP variance when variables in the index set
        p are conditioned on (fixed) and the remaining variables are integrated
        out.

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
        # Scale C-integrals to K-quantities
        s2 = self._get_kernel_variance()
        s4 = s2 * s2
        P_C_p = self._calc.conditional_P(index)  # C-quantity
        u_C_p = self._calc.conditional_u(index)  # C-quantity
        P_K_p = s4 * P_C_p
        u_K_p = s2 * u_C_p

        # Get alpha = A^{-1} y from GP
        alpha = self._stats._gp.alpha()  # Shape: (nqoi, n_train)

        # For single output, squeeze to 1D
        if alpha.shape[0] == 1:
            alpha_1d = self._bkd.reshape(alpha, (-1,))  # Shape: (n_train,)
        else:
            raise NotImplementedError(
                "conditional_variance currently only supports single-output GPs (nqoi=1)"
            )

        # ζ_p = αᵀ P_K_p α
        P_K_p_alpha = P_K_p @ alpha_1d
        zeta_p = alpha_1d @ P_K_p_alpha

        # v_p² = u_K_p - tr[P_K_p A⁻¹]
        A_inv_P_K_p = self._stats._solve(P_K_p)
        trace_P_K_p_A_inv = self._bkd.trace(A_inv_P_K_p)
        v_p_sq = u_K_p - trace_P_K_p_A_inv

        # η (cached)
        eta = self._stats.mean_of_mean()

        # var_mu = u_K - τ_Kᵀ A⁻¹ τ_K (cached from variance_of_mean)
        var_mu = self._stats.variance_of_mean()

        # E[γ_f^(p)] = ζ_p + v_p² - η² - var_mu
        cond_var = zeta_p + v_p_sq - eta * eta - var_mu

        # Ensure non-negative (numerical stability)
        cond_var = cond_var * (cond_var >= 0.0)

        return cond_var

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
        total_variance = self._stats.mean_of_variance()

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
        total_variance = self._stats.mean_of_variance()

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
