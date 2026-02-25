"""PCE coefficient extraction from FunctionTrain cores."""

from typing import Dict, Generic, List, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore


class PCEFunctionTrainCore(Generic[Array]):
    """Extract PCE coefficient matrices from a FunctionTrain core.

    Assembles coefficients from univariate expansions into matrices
    Θ^{(ℓ)} of shape (r_left, r_right) for each basis index ℓ.

    Parameters
    ----------
    core : FunctionTrainCore[Array]
        The FunctionTrain core to wrap.

    Raises
    ------
    TypeError
        If any univariate expansion doesn't support get_coefficients().
    ValueError
        If expansions have different nterms or nqoi != 1.

    Warning
    -------
    Assumes all basis expansions use orthonormal polynomials (e.g., Legendre,
    Hermite, Laguerre). Results are mathematically incorrect for non-orthonormal
    bases such as monomials.

    Notes
    -----
    Currently only supports nqoi=1. Multi-QoI support may be added later.

    All univariate expansions in the core must have the same nterms. This
    excludes FunctionTrain structures mixing ConstantExpansion (nterms=1)
    with PCE (nterms>1) in the same core (e.g., additive structure).
    """

    def __init__(self, core: FunctionTrainCore[Array]) -> None:
        self._core = core
        self._bkd = core.bkd()
        self._validate_pce_structure()
        # Caches
        self._theta_cache: Dict[int, Array] = {}
        self._expected_kron_cache: Optional[Array] = None
        self._delta_kron_cache: Optional[Array] = None
        self._mean_kron_cache: Optional[Array] = None

    def _validate_pce_structure(self) -> None:
        """Verify all univariate expansions are compatible PCE."""
        r_left, r_right = self._core.ranks()
        nterms_values: List[int] = []

        for ii in range(r_left):
            for jj in range(r_right):
                bexp = self._core.get_basisexp(ii, jj)

                # Check has get_coefficients
                if not hasattr(bexp, "get_coefficients"):
                    raise TypeError(
                        f"Expansion at ({ii}, {jj}) doesn't support "
                        "get_coefficients(). All expansions must be PCE-like."
                    )

                # Check nqoi == 1
                if bexp.nqoi() != 1:
                    raise ValueError(
                        f"Expansion at ({ii}, {jj}) has nqoi={bexp.nqoi()}, "
                        "but only nqoi=1 is currently supported."
                    )

                nterms_values.append(bexp.nterms())

        # All nterms must match
        unique_nterms = set(nterms_values)
        if len(unique_nterms) > 1:
            raise ValueError(
                f"Expansions have inconsistent nterms: {unique_nterms}. "
                "All univariate expansions in a core must have the same nterms. "
                "Note: ConstantExpansion has nterms=1, which is incompatible "
                "with PCE expansions having nterms>1."
            )

        self._nterms = nterms_values[0] if nterms_values else 0

    def core(self) -> FunctionTrainCore[Array]:
        """Return the underlying FunctionTrainCore."""
        return self._core

    def nterms(self) -> int:
        """Number of basis terms in each univariate expansion."""
        return self._nterms

    def ranks(self) -> Tuple[int, int]:
        """Return (r_left, r_right)."""
        return self._core.ranks()

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def coefficient_matrix(self, basis_idx: int) -> Array:
        """Assemble Θ_k^{(ℓ)} ∈ R^{r_left × r_right} for basis index ℓ.

        Θ_k^{(ℓ)}[i,j] = coefficient of basis ℓ in univariate function (i,j).

        Parameters
        ----------
        basis_idx : int
            Basis index (0 = constant term, 1 = linear, etc.)

        Returns
        -------
        Array
            Coefficient matrix. Shape: (r_left, r_right)

        Raises
        ------
        IndexError
            If basis_idx is out of bounds [0, nterms).
        """
        if basis_idx < 0 or basis_idx >= self._nterms:
            raise IndexError(
                f"basis_idx {basis_idx} out of bounds [0, {self._nterms})"
            )

        if basis_idx in self._theta_cache:
            return self._theta_cache[basis_idx]

        r_left, r_right = self._core.ranks()

        # Build as list of lists, then stack (backend-agnostic)
        rows = []
        for ii in range(r_left):
            row = []
            for jj in range(r_right):
                coef = self._core.get_basisexp(ii, jj).get_coefficients()
                # coef shape: (nterms, 1) for nqoi=1
                row.append(coef[basis_idx, 0])
            # Stack row elements into 1D array
            rows.append(self._bkd.stack(row))
        # Stack rows into 2D array
        theta = self._bkd.stack(rows)

        self._theta_cache[basis_idx] = theta
        return theta

    def expected_core(self) -> Array:
        """Return E[F_k] = Θ_k^{(0)} (constant term coefficients).

        Returns
        -------
        Array
            Expected core matrix. Shape: (r_left, r_right)
        """
        return self.coefficient_matrix(0)

    def expected_kron_core(self) -> Array:
        """Return M_k = Σ_ℓ Θ_k^{(ℓ)} ⊗ Θ_k^{(ℓ)} ∈ R^{r_left² × r_right²}.

        This is E[F_k ⊗ F_k] using orthonormality.

        Returns
        -------
        Array
            Expected Kronecker product. Shape: (r_left², r_right²)
        """
        if self._expected_kron_cache is not None:
            return self._expected_kron_cache

        r_left, r_right = self._core.ranks()
        M = self._bkd.zeros((r_left * r_left, r_right * r_right))

        for ell in range(self._nterms):
            theta = self.coefficient_matrix(ell)
            kron = self._bkd.kron(theta, theta)
            M = M + kron

        self._expected_kron_cache = M
        return M

    def delta_kron_core(self) -> Array:
        """Return ΔM_k = Σ_{ℓ≥1} Θ_k^{(ℓ)} ⊗ Θ_k^{(ℓ)} (excludes constant).

        This is M_k - M_k^{(0)}, the non-constant contribution.

        Returns
        -------
        Array
            Delta Kronecker product. Shape: (r_left², r_right²)
        """
        if self._delta_kron_cache is not None:
            return self._delta_kron_cache

        r_left, r_right = self._core.ranks()
        delta_M = self._bkd.zeros((r_left * r_left, r_right * r_right))

        for ell in range(1, self._nterms):  # Skip ℓ=0
            theta = self.coefficient_matrix(ell)
            kron = self._bkd.kron(theta, theta)
            delta_M = delta_M + kron

        self._delta_kron_cache = delta_M
        return delta_M

    def mean_kron_core(self) -> Array:
        """Return M_k^{(0)} = Θ_k^{(0)} ⊗ Θ_k^{(0)}.

        This is the constant-term Kronecker product.

        Returns
        -------
        Array
            Mean Kronecker product. Shape: (r_left², r_right²)
        """
        if self._mean_kron_cache is not None:
            return self._mean_kron_cache

        theta0 = self.coefficient_matrix(0)
        self._mean_kron_cache = self._bkd.kron(theta0, theta0)
        return self._mean_kron_cache
