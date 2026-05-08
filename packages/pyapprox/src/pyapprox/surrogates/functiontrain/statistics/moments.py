"""Analytical moments computation for PCE FunctionTrain."""

from typing import Generic, Optional

from pyapprox.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.util.backends.protocols import Array, Backend


class FunctionTrainMoments(Generic[Array]):
    """Compute analytical moments from PCE FunctionTrain.

    Uses orthonormality: E[φ_ℓ] = δ_{ℓ0}, E[φ_ℓ φ_m] = δ_{ℓm}

    Parameters
    ----------
    pce_ft : PCEFunctionTrain[Array]
        PCE-validated FunctionTrain.

    Raises
    ------
    TypeError
        If pce_ft is not a PCEFunctionTrain instance.

    Notes
    -----
    Complexity:
    - Mean: O(d · r²)
    - Variance: O(d · r⁴ · p) where p = nterms

    For r > 10 or p > 20, Kronecker products may become large.

    Warning
    -------
    Assumes all basis expansions use orthonormal polynomials (e.g., Legendre,
    Hermite, Laguerre). Results are mathematically incorrect for non-orthonormal
    bases such as monomials.
    """

    def __init__(self, pce_ft: PCEFunctionTrain[Array]) -> None:
        if not isinstance(pce_ft, PCEFunctionTrain):
            raise TypeError(f"Expected PCEFunctionTrain, got {type(pce_ft).__name__}")
        self._pce_ft = pce_ft
        self._bkd = pce_ft.bkd()
        self._mean_cache: Optional[Array] = None
        self._second_moment_cache: Optional[Array] = None

    def pce_ft(self) -> PCEFunctionTrain[Array]:
        """Access underlying PCEFunctionTrain."""
        return self._pce_ft

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def mean(self) -> Array:
        """E[f] = Θ_1^{(0)} · Θ_2^{(0)} · ... · Θ_d^{(0)}.

        Product of expected (constant-term) cores.

        Returns
        -------
        Array
            Mean value. Shape: (1,) - scalar as 1D array.
        """
        if self._mean_cache is None:
            self._mean_cache = self._compute_mean()
        return self._mean_cache

    def _compute_mean(self) -> Array:
        """Product of expected cores."""
        cores = self._pce_ft.pce_cores()
        result = cores[0].expected_core()
        for core in cores[1:]:
            result = self._bkd.dot(result, core.expected_core())
        # Result shape: (1, 1) for boundary ranks → reshape to (1,)
        return self._bkd.reshape(result, (1,))

    def second_moment(self) -> Array:
        """E[f²] = Π_k M_k where M_k = E[F_k ⊗ F_k].

        Product of expected Kronecker cores.

        Returns
        -------
        Array
            Second moment. Shape: (1,) - scalar as 1D array.
        """
        if self._second_moment_cache is None:
            self._second_moment_cache = self._compute_second_moment()
        return self._second_moment_cache

    def _compute_second_moment(self) -> Array:
        """Product of expected Kronecker cores."""
        cores = self._pce_ft.pce_cores()
        result = cores[0].expected_kron_core()
        for core in cores[1:]:
            result = self._bkd.dot(result, core.expected_kron_core())
        # Result shape: (1, 1) for boundary ranks → reshape to (1,)
        return self._bkd.reshape(result, (1,))

    def variance(self) -> Array:
        """Var[f] = E[f²] - E[f]².

        Returns
        -------
        Array
            Variance. Shape: (1,) - scalar as 1D array (non-negative).
        """
        return self.second_moment() - self.mean() ** 2

    def std(self) -> Array:
        """Standard deviation σ = √Var[f].

        Returns
        -------
        Array
            Standard deviation. Shape: (1,) - scalar as 1D array.
        """
        return self._bkd.sqrt(self.variance())
