"""Polynomial Chaos Expansion (PCE) implementation.

A PCE represents a function as a linear combination of orthonormal
polynomial basis functions: f(x) ≈ Σ_i c_i ψ_i(x), where ψ_i are
orthonormal with respect to the input probability measure.
"""

from typing import Generic, List, Optional, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    LinearSystemSolverProtocol,
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.typing.surrogates.affine.basis import (
    OrthonormalPolynomialBasis,
)
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.expansions.base import BasisExpansion
from pyapprox.typing.surrogates.affine.univariate.factory import (
    create_basis_1d,
)
from pyapprox.typing.surrogates.affine.univariate.transformed import (
    TransformedBasis1D,
    NativeBasis1D,
)
from pyapprox.typing.surrogates.affine.expansions.pce_arithmetic import (
    add_pce,
    add_constant_to_pce,
    multiply_pce,
    multiply_pce_by_constant,
    pce_power,
)


class PolynomialChaosExpansion(BasisExpansion[Array], Generic[Array]):
    """Polynomial Chaos Expansion with orthonormal polynomial basis.

    PCE exploits orthonormality for efficient computation of statistics
    (mean, variance, Sobol indices) directly from coefficients.

    Parameters
    ----------
    basis : OrthonormalPolynomialBasis[Array]
        Orthonormal polynomial basis.
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest. Default: 1.
    solver : LinearSystemSolverProtocol[Array], optional
        Solver for fitting.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability import UniformMarginal
    >>> from pyapprox.typing.surrogates.affine.univariate.factory import create_basis_1d
    >>> from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
    >>> from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
    >>> bases_1d = [create_basis_1d(m, bkd) for m in marginals]
    >>> indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
    >>> basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    >>> pce = PolynomialChaosExpansion(basis, bkd)
    """

    def __init__(
        self,
        basis: OrthonormalPolynomialBasis[Array],
        bkd: Backend[Array],
        nqoi: int = 1,
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ):
        if not isinstance(basis, OrthonormalPolynomialBasis):
            raise TypeError(
                f"basis must be OrthonormalPolynomialBasis, "
                f"got {type(basis).__name__}"
            )
        super().__init__(basis, bkd, nqoi, solver)

    def get_indices(self) -> Array:
        """Return multi-indices. Shape: (nvars, nterms)."""
        return self._basis.get_indices()

    def _get_constant_index(self) -> int:
        """Return index of the constant basis function."""
        indices = self.get_indices()
        # Constant term has all zeros
        index_sums = self._bkd.sum(indices, axis=0)
        const_mask = index_sums == 0
        const_indices = self._bkd.nonzero(const_mask)
        if len(const_indices[0]) == 0:
            raise ValueError("Basis does not contain constant term")
        return int(const_indices[0][0])

    def mean(self) -> Array:
        """Compute mean from PCE coefficients.

        For orthonormal polynomials, E[f] = c_0 (coefficient of constant term).

        Returns
        -------
        Array
            Mean values. Shape: (nqoi,)
        """
        const_idx = self._get_constant_index()
        return self._coef[const_idx, :]

    def variance(self) -> Array:
        """Compute variance from PCE coefficients.

        For orthonormal polynomials: Var[f] = Σ_{i≠0} c_i²

        Returns
        -------
        Array
            Variance values. Shape: (nqoi,)
        """
        const_idx = self._get_constant_index()
        # Sum of squared coefficients excluding constant
        coef_sq = self._coef ** 2
        total = self._bkd.sum(coef_sq, axis=0)
        # Subtract constant term squared
        return total - coef_sq[const_idx, :]

    def std(self) -> Array:
        """Compute standard deviation from PCE coefficients.

        Returns
        -------
        Array
            Standard deviation values. Shape: (nqoi,)
        """
        return self._bkd.sqrt(self.variance())

    def covariance(self) -> Array:
        """Compute covariance matrix between QoIs.

        For orthonormal polynomials: Cov[f_i, f_j] = Σ_{k≠0} c_{k,i} c_{k,j}

        Returns
        -------
        Array
            Covariance matrix. Shape: (nqoi, nqoi)
        """
        const_idx = self._get_constant_index()
        # coef: (nterms, nqoi)
        # Remove constant term
        coef_nonconstant = self._bkd.concatenate([
            self._coef[:const_idx, :],
            self._coef[const_idx+1:, :]
        ], axis=0)
        # Cov = coef^T @ coef
        return self._bkd.dot(coef_nonconstant.T, coef_nonconstant)

    def total_sobol_indices(self) -> Array:
        """Compute total Sobol sensitivity indices.

        The total Sobol index T_i measures the total contribution of
        variable i (including interactions) to the output variance.

        Returns
        -------
        Array
            Total Sobol indices. Shape: (nvars, nqoi)
        """
        indices = self.get_indices()  # (nvars, nterms)
        var = self.variance()  # (nqoi,)
        coef = self._coef

        # Avoid division by zero
        var_safe = self._bkd.where(
            var > 0, var, self._bkd.ones_like(var)
        )

        total_indices = self._bkd.zeros((self.nvars(), self.nqoi()))

        for dd in range(self.nvars()):
            # Terms that depend on variable dd
            depends_on_dd = indices[dd, :] > 0
            # Sum of squared coefficients for those terms
            coef_sq = coef ** 2
            mask = self._bkd.asarray(depends_on_dd, dtype=self._bkd.default_dtype())
            contribution = self._bkd.sum(
                coef_sq * self._bkd.reshape(mask, (-1, 1)),
                axis=0
            )
            total_indices[dd, :] = contribution / var_safe

        # Set to zero where variance is zero
        total_indices = self._bkd.where(
            self._bkd.reshape(var, (1, -1)) > 0,
            total_indices,
            self._bkd.zeros_like(total_indices)
        )
        return total_indices

    def main_effect_sobol_indices(self) -> Array:
        """Compute main effect (first-order) Sobol indices.

        The main effect S_i measures the contribution of variable i
        alone (no interactions) to the output variance.

        Returns
        -------
        Array
            Main effect Sobol indices. Shape: (nvars, nqoi)
        """
        indices = self.get_indices()  # (nvars, nterms)
        var = self.variance()  # (nqoi,)
        coef = self._coef

        # Avoid division by zero
        var_safe = self._bkd.where(
            var > 0, var, self._bkd.ones_like(var)
        )

        main_indices = self._bkd.zeros((self.nvars(), self.nqoi()))

        for dd in range(self.nvars()):
            # Terms that depend ONLY on variable dd (no interactions)
            depends_on_dd = indices[dd, :] > 0
            # Sum of all indices equals index for variable dd (no other vars)
            index_sum = self._bkd.sum(indices, axis=0)
            other_vars_zero = index_sum == indices[dd, :]
            main_effect_terms = depends_on_dd & other_vars_zero

            coef_sq = coef ** 2
            mask = self._bkd.asarray(
                main_effect_terms, dtype=self._bkd.default_dtype()
            )
            contribution = self._bkd.sum(
                coef_sq * self._bkd.reshape(mask, (-1, 1)),
                axis=0
            )
            main_indices[dd, :] = contribution / var_safe

        # Set to zero where variance is zero
        main_indices = self._bkd.where(
            self._bkd.reshape(var, (1, -1)) > 0,
            main_indices,
            self._bkd.zeros_like(main_indices)
        )
        return main_indices

    def fit_via_projection(
        self,
        quad_samples: Array,
        values: Array,
        quad_weights: Array,
    ) -> None:
        """Fit PCE via spectral projection using quadrature.

        For orthonormal polynomials: c_i = E[f(x)ψ_i(x)] ≈ Σ_j w_j f(x_j)ψ_i(x_j)

        Parameters
        ----------
        quad_samples : Array
            Quadrature sample points. Shape: (nvars, nsamples)
        values : Array
            Function values at quadrature points. Shape: (nqoi, nsamples) or
            (nsamples,) for nqoi=1.
        quad_weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        # Evaluate basis at quadrature points
        basis_vals = self._basis(quad_samples)  # (nsamples, nterms)

        # Compute coefficients via projection
        # c_i = Σ_j w_j f(x_j) ψ_i(x_j)
        # values: (nqoi, nsamples) -> transpose to (nsamples, nqoi)
        weighted_values = values.T * self._bkd.reshape(quad_weights, (-1, 1))
        coef = basis_vals.T @ weighted_values  # (nterms, nqoi)
        self.set_coefficients(coef)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(
        self, other: Union["PolynomialChaosExpansion[Array]", float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        if isinstance(other, (float, int)):
            return add_constant_to_pce(self, float(other))
        if isinstance(other, PolynomialChaosExpansion):
            return add_pce(self, other, sign=1.0)
        return NotImplemented

    def __radd__(
        self, other: Union["PolynomialChaosExpansion[Array]", float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        return self.__add__(other)

    def __sub__(
        self, other: Union["PolynomialChaosExpansion[Array]", float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        if isinstance(other, (float, int)):
            return add_constant_to_pce(self, -float(other))
        if isinstance(other, PolynomialChaosExpansion):
            return add_pce(self, other, sign=-1.0)
        return NotImplemented

    def __rsub__(
        self, other: Union[float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        if isinstance(other, (float, int)):
            result = multiply_pce_by_constant(self, -1.0)
            return add_constant_to_pce(result, float(other))
        return NotImplemented

    def __mul__(
        self, other: Union["PolynomialChaosExpansion[Array]", float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        if isinstance(other, (float, int)):
            return multiply_pce_by_constant(self, float(other))
        if isinstance(other, PolynomialChaosExpansion):
            return multiply_pce(self, other)
        return NotImplemented

    def __rmul__(
        self, other: Union["PolynomialChaosExpansion[Array]", float, int]
    ) -> "PolynomialChaosExpansion[Array]":
        return self.__mul__(other)

    def __pow__(self, order: int) -> "PolynomialChaosExpansion[Array]":
        return pce_power(self, order)

    def __repr__(self) -> str:
        return (
            f"PolynomialChaosExpansion(nvars={self.nvars()}, "
            f"nterms={self.nterms()}, nqoi={self.nqoi()})"
        )


_PhysicalDomainBasis1D = Union[TransformedBasis1D[Array], NativeBasis1D[Array]]


def create_pce(
    bases_1d: List[_PhysicalDomainBasis1D],
    max_level: int,
    bkd: Backend[Array],
    pnorm: float = 1.0,
    nqoi: int = 1,
    solver: Optional[LinearSystemSolverProtocol[Array]] = None,
) -> PolynomialChaosExpansion[Array]:
    """Create a PCE with hyperbolic index set.

    Parameters
    ----------
    bases_1d : List[PhysicalDomainBasis1DProtocol[Array]]
        Physical-domain univariate bases (TransformedBasis1D or NativeBasis1D).
        Use create_basis_1d(marginal, bkd) to create these.
    max_level : int
        Maximum polynomial level.
    bkd : Backend[Array]
        Computational backend.
    pnorm : float
        p-norm for hyperbolic index selection. Default: 1.0 (total degree).
    nqoi : int
        Number of quantities of interest. Default: 1.
    solver : LinearSystemSolverProtocol[Array], optional
        Solver for fitting.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        The PCE object.
    """
    nvars = len(bases_1d)
    indices = compute_hyperbolic_indices(nvars, max_level, pnorm, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return PolynomialChaosExpansion(basis, bkd, nqoi, solver)


def get_basis_from_marginal(
    marginal,
    bkd: Backend[Array],
) -> _PhysicalDomainBasis1D:
    """Get physical-domain basis for a marginal distribution.

    Uses the marginal registry to select the optimal polynomial family
    and wraps it with the appropriate transform for physical-domain samples.

    This is the primary API for creating 1D bases from marginals.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    TransformedBasis1D or NativeBasis1D
        Physical-domain basis that accepts samples from marginal's support.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability import UniformMarginal, BetaMarginal
    >>> bkd = NumpyBkd()
    >>> # Uniform on [0, 2]
    >>> basis = get_basis_from_marginal(UniformMarginal(0.0, 2.0, bkd), bkd)
    >>> # Beta(2, 5) on [0.5, 1.5]
    >>> basis = get_basis_from_marginal(BetaMarginal(2.0, 5.0, bkd, lb=0.5, ub=1.5), bkd)
    """
    return create_basis_1d(marginal, bkd)


def create_pce_from_marginals(
    marginals: List,
    max_level: int,
    bkd: Backend[Array],
    pnorm: float = 1.0,
    nqoi: int = 1,
    solver: Optional[LinearSystemSolverProtocol[Array]] = None,
) -> PolynomialChaosExpansion[Array]:
    """Create PCE from marginal distributions.

    Automatically selects optimal orthonormal polynomial for each marginal
    using the Askey scheme. Supports both continuous and discrete marginals.
    Wraps polynomials with appropriate transforms for physical-domain samples.

    Parameters
    ----------
    marginals : List of marginals
        Univariate marginal distributions for each variable.
    max_level : int
        Maximum polynomial level.
    bkd : Backend[Array]
        Computational backend.
    pnorm : float
        p-norm for hyperbolic index selection. Default: 1.0
    nqoi : int
        Number of quantities of interest. Default: 1
    solver : LinearSystemSolverProtocol, optional
        Solver for fitting.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        PCE with physical-domain polynomial bases.
    """
    bases_1d = [get_basis_from_marginal(m, bkd) for m in marginals]
    return create_pce(bases_1d, max_level, bkd, pnorm, nqoi, solver)
